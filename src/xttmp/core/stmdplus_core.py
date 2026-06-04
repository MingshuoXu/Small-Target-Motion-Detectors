import torch
from torch.nn import functional as F

from .base_core import BaseCore
from ..util.create_kernel import create_T1_kernels
from ..util.compute_module import AreaNMS
from ..util.compute_module import compute_response


class ContrastPathway(BaseCore):
    """ContrastPathway class for ApgSTMD - PyTorch Version."""

    def __init__(self):
        """Constructor method."""
        super().__init__()
        
        self.theta = torch.tensor([0, torch.pi/4, torch.pi/2, 3*torch.pi/4])
        self.alpha2 = 1.5
        self.eta = 3
        self.sizeT1 = 11
        
        self.register_buffer('T1_kernel', torch.empty(0))

    def setup(self):
        """Initialization method."""
        # 假设 create_T1_kernels 返回的是 4 个 kernel 的 List 或 NumPy array
        _T1_kernel = create_T1_kernels(len(self.theta), self.alpha2, self.eta, self.sizeT1)
        self.T1_kernel.data = _T1_kernel

    def forward(self, x):
        """
        Processing method.
        retinaOpt: 可以是 (H, W) 或 (1, 1, H, W) 的 PyTorch Tensor
        """
        
        # out 的形状是 (1, 4, H, W)
        self.Opt = F.conv2d(x, self.T1_kernel, padding='same')

        return self.Opt
    

class MushroomBody(BaseCore):
    # MushroomBody class for STMDPlus - Ultimate Vectorized PyTorch Version

    def __init__(self):
        super().__init__()
        self.nms_size = 5

        self.DBSCANDist = 5.0
        self.lenDBSCAN = 100
        self.SDThres = 5.0
        
        self.torch_nms = None
        
        # ================= 终极优化：预分配的全张量状态机 =================
        self.C = None           # 通道数 (将在第一帧自动推断)
        self.trackID = None     # 张量: [N, D] 坐标
        self.trackInfo = None   # 张量: [N, C, lenDBSCAN] (固定大小环形缓冲区)
        self.trackLens = None   # 张量: [N] (记录当前轨迹有效长度)
        self.trackPtr = None    # 张量: [N] (记录环形缓冲区的写入指针)

    def setup(self):
        self.torch_nms = AreaNMS(self.nms_size)

    def forward(self, lobulaOpt, contrast_tensor):
        device = lobulaOpt.device

        maxLobulaOpt = compute_response(lobulaOpt) 
        nmsLobulaOpt = self.torch_nms(maxLobulaOpt)

        mask_not_nms = (nmsLobulaOpt == 0)
        mushroomBodyOpt = lobulaOpt * mask_not_nms

        maxNumber = torch.max(nmsLobulaOpt)
        if maxNumber <= 0:
            self.trackID = None
            return mushroomBodyOpt

        # --- 获取新检测点 ---
        newID = torch.nonzero(nmsLobulaOpt > 0).float() 
        if len(newID) == 0:
            self.trackID = None
            return mushroomBodyOpt

        curr_y, curr_x = newID[:, -2].long(), newID[:, -1].long()
        # all_new_contrasts 形状 [C, M], M是新目标数
        all_new_contrasts = contrast_tensor[:, 0, curr_y, curr_x]
        
        if self.C is None:
            self.C = contrast_tensor.shape[0]

        M = len(newID)

        # ================= 1. 轨迹匹配 (保持 CPU 高效碰撞处理) =================
        matched_old, matched_new, used_new = [], [], set()
        
        if self.trackID is not None and len(self.trackID) > 0:
            DD = torch.cdist(self.trackID[:, -2:], newID[:, -2:])
            D1, min_idx = torch.min(DD, dim=1)
            
            # 转移到 CPU 做极速冲突判定
            D1_cpu, min_idx_cpu = D1.cpu().numpy(), min_idx.cpu().numpy()
            for i, d1 in enumerate(D1_cpu):
                if d1 <= self.DBSCANDist:
                    j = min_idx_cpu[i]
                    if j not in used_new:
                        used_new.add(j)
                        matched_old.append(i)
                        matched_new.append(j)

        # ================= 2. 环形缓冲区批量状态更新 (0 For循环) =================
        matched_old_ts = torch.tensor(matched_old, dtype=torch.long, device=device)
        matched_new_ts = torch.tensor(matched_new, dtype=torch.long, device=device)
        unmatched_new_ts = torch.tensor([j for j in range(M) if j not in used_new], dtype=torch.long, device=device)

        if self.trackID is not None and len(matched_old) > 0:
            # --- 提取续航的轨迹状态 ---
            next_trackID = newID[matched_new_ts]
            next_trackInfo = self.trackInfo[matched_old_ts]
            next_trackLens = self.trackLens[matched_old_ts]
            next_trackPtr = self.trackPtr[matched_old_ts]
            
            # 批量写入环形缓冲区 (全矩阵操作)
            batch_idx = torch.arange(len(matched_old), device=device)
            next_trackInfo[batch_idx, :, next_trackPtr] = all_new_contrasts[:, matched_new_ts].T
            next_trackLens = torch.clamp(next_trackLens + 1, max=self.lenDBSCAN)
            next_trackPtr = (next_trackPtr + 1) % self.lenDBSCAN
        else:
            # 定义空张量用于拼接
            next_trackID = torch.empty((0, newID.shape[1]), device=device)
            next_trackInfo = torch.empty((0, self.C, self.lenDBSCAN), device=device)
            next_trackLens = torch.empty((0,), dtype=torch.long, device=device)
            next_trackPtr = torch.empty((0,), dtype=torch.long, device=device)

        if len(unmatched_new_ts) > 0:
            # --- 批量初始化新轨迹 ---
            N_add = len(unmatched_new_ts)
            add_trackID = newID[unmatched_new_ts]
            add_trackInfo = torch.zeros((N_add, self.C, self.lenDBSCAN), device=device)
            add_trackInfo[:, :, 0] = all_new_contrasts[:, unmatched_new_ts].T
            add_trackLens = torch.ones(N_add, dtype=torch.long, device=device)
            add_trackPtr = torch.ones(N_add, dtype=torch.long, device=device)
            
            # --- 与续航的轨迹合并 ---
            self.trackID = torch.cat([next_trackID, add_trackID], dim=0)
            self.trackInfo = torch.cat([next_trackInfo, add_trackInfo], dim=0)
            self.trackLens = torch.cat([next_trackLens, add_trackLens], dim=0)
            self.trackPtr = torch.cat([next_trackPtr, add_trackPtr], dim=0)
        else:
            self.trackID, self.trackInfo, self.trackLens, self.trackPtr = next_trackID, next_trackInfo, next_trackLens, next_trackPtr

        # ================= 3. 并行 STD 计算与批量擦除 =================
        if self.trackID is not None and len(self.trackID) > 0:
            valid_mask = self.trackLens > 1
            if valid_mask.any():
                # 只取出长度 > 1 的轨迹
                chk_info = self.trackInfo[valid_mask]    # [K, C, 100]
                chk_lens = self.trackLens[valid_mask]    # [K]
                chk_coords = self.trackID[valid_mask]    # [K, D]
                
                # 创建时间遮罩: 标记环形缓冲区中哪些数据是有效的
                time_idx = torch.arange(self.lenDBSCAN, device=device).view(1, 1, -1) # [1, 1, 100]
                data_mask = time_idx < chk_lens.view(-1, 1, 1) # [K, 1, 100]
                
                # 手动并行计算 Masked STD (利用数学公式：Var = sum((x - mean)^2) / (N - 1))
                # 因为方差是无序的，即便环形缓冲区数据没按时间排序，也绝对不影响最终计算结果！
                sum_val = (chk_info * data_mask).sum(dim=-1) # [K, C]
                mean_val = sum_val / chk_lens.unsqueeze(1).float()
                
                diff_sq = ((chk_info - mean_val.unsqueeze(-1)) * data_mask) ** 2
                var_val = diff_sq.sum(dim=-1) / (chk_lens.unsqueeze(1).float() - 1.0)
                std_val = var_val.sqrt() # [K, C] 
                
                # 找出最大的 STD，并判定
                max_std, _ = std_val.max(dim=1) # [K]
                erase_mask = max_std < self.SDThres # [K]
                
                if erase_mask.any():
                    erase_coords = chk_coords[erase_mask]
                    e_y = erase_coords[:, -2].long()
                    e_x = erase_coords[:, -1].long()
                    # 终极一键批量擦除
                    mushroomBodyOpt[..., e_y, e_x] = 0

        self.Opt = mushroomBodyOpt
        return mushroomBodyOpt

