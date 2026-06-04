import math
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .base_core import BaseCore
from .math_operator import SpatialInhibition
from . import fracstmd_core
from ..util.create_kernel import create_2d_gaussian_kernel


class Lamina(fracstmd_core.Lamina):
    """ Lamina layer of the motion detection system."""        
    def forward(self, LaminaIpt):
        temporal_diff_output = super().forward(LaminaIpt)

        lamina_ON = torch.clamp(temporal_diff_output, min=0) # ON
        lamina_OFF = torch.clamp(-temporal_diff_output, min=0) # OFF

        self.output = (lamina_ON, lamina_OFF)
        return self.output


class cIDP(BaseCore):
    ''' cross-Inhibiton Dynamics Potentials (cIDP) '''

    def __init__(self):
        super().__init__()

        self.g_leak = 0.5 # coefficient of decay
        self.v_rest = 0;  # passive/rest potentials;
        self.v_exci = 1;  # excitatory saturation potentials;
        self.reset_buffer()

    def reset_buffer(self):
        self.post_MP = None

    def forward(self, same_polarity, oppo_polarity):
        if self.post_MP is None:
            self.post_MP = torch.zeros_like(same_polarity)

        # Decay
        decay_term = self.g_leak * (self.v_rest - self.post_MP)

        # Inhibition
        inhi_gain = torch.exp(oppo_polarity)

        # Excitation
        exci_term = same_polarity * (self.v_exci - self.post_MP)
        
        # Euler method for solving ordinary differential equation
        self.post_MP += inhi_gain * decay_term + exci_term
            
        return self.post_MP


class Medulla(BaseCore):
    """
    Medulla layer of the motion detection system.

    Illustration:    

    """

    def __init__(self):
        super().__init__()
        # Initialize components
        self.on_pathway = cIDP()
        self.off_pathway = cIDP()

    def setup(self):
        # Initialize configurations
        self.on_pathway.setup()
        self.off_pathway.setup()

    def forward(self, lamina_ON, lamina_OFF):
        """
        Process the input through the Medulla layer.

        Args:
        - medullaIpt (array-like): Input to the Medulla layer.

        Returns:
        - Von (array-like): Output ON signal from Dual-Dynamic.
        - Voff (array-like): Output OFF signal from Dual-Dynamic.
        """
        
        medulla_ON = self.on_pathway.forward(lamina_ON, lamina_OFF); # ON
        medulla_OFF = self.off_pathway.forward(lamina_OFF, lamina_ON); # OFF

        # Store the output signals
        self.output = (medulla_ON, medulla_OFF)

        return self.output
    

class CDGC(BaseCore):
    """
    Collaborative Directional Encoding-Decoding (CDGC)
    Pure PyTorch Implementation.
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.register_buffer('corr_kernel_cos', torch.empty(0))
        self.register_buffer('corr_kernel_sin', torch.empty(0))

        self.setup()

    def setup(self):
        # 1. 在初始化时预先生成空间方向卷积核
        _cos_kernel, _sin_kernel = self._create_directional_kernels()
        
        # 2. 使用 register_buffer 注册为模型状态
        self.corr_kernel_cos.data = _cos_kernel
        self.corr_kernel_sin.data = _sin_kernel

    def _create_directional_kernels(self):
        """利用 PyTorch 原生算子，向量化生成 Cosine 和 Sine 感受野核"""
        # 生成一维坐标序列，例如核大小为3时，生成 [-1, 0, 1]
        coords = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        
        # 使用 meshgrid 快速生成二维网格 ('ij' 模式确保 y 对应行，x 对应列)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        
        # 计算欧氏距离
        r = torch.sqrt(x**2 + y**2)
        
        # 临时将中心点的 r 设为 1.0 以避免除以 0 导致 NaN (后面会强制将中心值设回 0)
        r[r == 0] = 1.0
        
        # 计算方向权重
        cos_k = x / r
        sin_k = -y / r
        
        # 强制将中心点 (x=0, y=0) 设为 0
        center = self.kernel_size // 2
        cos_k[center, center] = 0.0
        sin_k[center, center] = 0.0
        
        # 调整形状为 (out_channels=1, in_channels=1, H, W)
        cos_k = cos_k.view(1, 1, self.kernel_size, self.kernel_size)
        sin_k = sin_k.view(1, 1, self.kernel_size, self.kernel_size)
        
        return cos_k, sin_k

    def forward(self, medulla_on, medulla_off, lamina_on, lamina_off):
        """
        前向传播
        所有输入均为形状为 (B, C, H, W) 的张量
        """
        C = medulla_on.shape[1]

        # --- 1. 计算协作编码矩阵 (cdedMatrix) ---
        direction_gradient = torch.zeros_like(medulla_on)
        
        # 提取布尔掩码
        mask_on = (lamina_on > 0) & (medulla_on > 0)
        mask_off = (lamina_off > 0) & (medulla_off > 0)
        
        # 通过掩码赋值，避免任何多余的全局除法运算
        direction_gradient[mask_on] = medulla_off[mask_on] / medulla_on[mask_on]
        direction_gradient[mask_off] = medulla_on[mask_off] / medulla_off[mask_off]

        # --- 2. 空间方向卷积 ---
        # 使用 .expand() 将单通道核扩展到匹配输入通道数 C，且不增加显存消耗
        # 注意: 形状调整为 (C, 1, K, K) 以适配 Depthwise Convolution
        weight_cos = self.corr_kernel_cos.expand(C, 1, self.kernel_size, self.kernel_size)
        weight_sin = self.corr_kernel_sin.expand(C, 1, self.kernel_size, self.kernel_size)

        # groups=C 表示进行深度可分离卷积，每个通道独立计算方向
        direction_cos = F.conv2d(direction_gradient, weight_cos, padding='same', groups=C)
        direction_sin = F.conv2d(direction_gradient, weight_sin, padding='same', groups=C)

        # --- 3. 计算角度 ---
        direction = torch.atan2(direction_sin, direction_cos)

        # 将范围调整到 [0, 2*pi]
        self.output = torch.where(direction < 0, direction + 2 * torch.pi, direction)

        self.direction_gradient = direction_gradient

        return self.output
    

class Lobula(BaseCore):
    """
    Lobula layer of the motion detection system.
    """

    def __init__(self):
        super().__init__()
        self.spatial_inhibition = SpatialInhibition(B=3, e=3, sigma1=5, sigma2=10)
        self.cdgc = CDGC()

        self.setup()

    def setup(self):
        """
        Initialization method.
        """
        self.spatial_inhibition.setup()
        self.cdgc.setup()

    def forward(self, medulla_on, medulla_off, lamina_on, lamina_off):
        """
        Processing method.

        Args:
        - medulla_on (np.array):  ON channel signal from medulla layer.
        - medulla_off (np.array):  OFF channel signal from medulla layer.
        - lamina_on (np.array):  ON channel signal from lamina layer.
        - lamina_off (np.array):  OFF channel signal from lamina layer.

        Returns:
        - lobulaoutput (np.array): output for location.
        - direction (np.array): output for direction.
        - correlationOutput (np.array): output without inhibition.
        """

        self.correlation_output = medulla_on * medulla_off
        lobula_output = self.spatial_inhibition.forward(self.correlation_output)

        direction = self.cdgc.forward(medulla_on, medulla_off, lamina_on, lamina_off)

        self.output = (lobula_output, direction)

        return self.output


class Lobula_with_Feedback(BaseCore):
    """Lobula layer of the motion detection system."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.spatial_inhibition = SpatialInhibition()  # SpatialInhibition component
        self.cdgc = CDGC()

        self.beta = 1  # Parameter beta
        self.sigma = 1.5  # Parameters for Gaussian kernel

        self.register_buffer('gaussian_kernel', torch.empty(0))  # Buffer for Gaussian blur kernel

        self.setup()

    def setup(self):
        """Initialization method."""
        # Initializes the Lobula layer component
        super().setup()

        self.gaussian_kernel.data = create_2d_gaussian_kernel(size=3, 
                                                            sigma= self.sigma)
        
        self.spatial_inhibition.setup()
        self.cdgc.setup()

        self.reset_buffer()

    def reset_buffer(self):
        """Reset the buffer for feedback signal."""
        self.feedback_signal = None

    def forward_localization(self, medulla_ON, medulla_OFF):

        self.feedback_signal = torch.zeros_like(medulla_ON)   

        # Formula (8)
        self.v_on = torch.clamp(medulla_ON - self.feedback_signal, min=0) 
        self.v_off = torch.clamp(medulla_OFF - self.feedback_signal, min=0)
        correlationD = self.v_on * self.v_off

        # Formula (10)
        correlationE = F.conv2d(medulla_ON * medulla_OFF, self.gaussian_kernel, padding='same')

        # Only record (correlationD + correlationE) for next delay in Formula (9)
        self.feedback_signal = self.beta * (correlationD + correlationE)

        # Formula (14)
        response = self.spatial_inhibition.forward(correlationD)
        return response

    def forward(self, medulla_ON, medulla_OFF, lamina_ON, lamina_OFF):
        response = self.forward_localization(medulla_ON, medulla_OFF)
        direction = self.cdgc.forward(medulla_ON, medulla_OFF, lamina_ON, lamina_OFF)

        self.output = (response, direction)
        return self.output

        
class FastEuclideanTracker:
    """
    一个基于欧氏距离和匈牙利算法的极速多目标追踪器 (PyTorch 版)。
    """
    def __init__(self, max_distance=5.0, max_unmatched=5, device='cpu'):
        """
        初始化追踪器。

        Args:
            max_distance (float): 匹配的最大欧氏距离阈值。
            max_unmatched (int): 轨迹在被删除前允许的最大未匹配帧数。
            device (str): 运行设备，'cpu' 或 'cuda'。
        """
        self.next_track_id = 0
        self.tracks = {}  # {track_id: {'center': tensor(x, y), 'unmatched_count': int, 'direction': float}}
        self.max_distance = max_distance
        self.max_unmatched = max_unmatched

    def update(self, response):
        """
        更新轨迹。
        Args: 
            response: torch.Tensor
        Returns:
            results: torch.Tensor, 形状为 (N, 3), 包含 (y, x, direction)
        """

        device = response.device 

        track_ids = list(self.tracks.keys())
        
        # === 1. 取出现有轨迹的 centers ===
        if len(track_ids) > 0:
            # stack 将列表中的 1D tensor 堆叠为 2D tensor (T, 2)
            track_centers = torch.stack([self.tracks[tid]['center'] for tid in track_ids])
        else:
            track_centers = torch.empty((0, 2), dtype=torch.float32, device=device)
        # === 若没有轨迹，全部新建 ===
        if len(track_centers) == 0:
            for i in range(len(response)):
                self.tracks[self.next_track_id] = {
                    'center': response[i],
                    'unmatched_count': 0
                }
                self.next_track_id += 1
            return torch.empty((0, 3), device=device)

        # === 2. 向量化构造代价矩阵 ===
        # torch.cdist 高度优化了欧氏距离计算，比手动广播相减再求范数更快
        cost_matrix = torch.cdist(track_centers[:, -2:].float(), response[:, 2:].float(), p=2)  # 形状 (T, D)

        # === 3. 匈牙利匹配 ===
        # PyTorch 没有内置的线性指派求解器，必须在 CPU 上用 scipy 计算
        cost_matrix_np = cost_matrix.cpu().numpy()
        track_idx_arr, det_idx_arr = linear_sum_assignment(cost_matrix_np)

        matched_pairs = []
        for t_i, d_i in zip(track_idx_arr, det_idx_arr):
            if cost_matrix_np[t_i, d_i] <= self.max_distance:
                matched_pairs.append((t_i, d_i))

        # 用于后续集合运算的快速查找
        matched_tracks = {p[0] for p in matched_pairs}
        matched_dets = {p[1] for p in matched_pairs}

        # === 4. 更新匹配成功的轨迹 ===
        if len(matched_pairs) > 0:
            # 提取按配对顺序排列的索引
            t_indices = [p[0] for p in matched_pairs]
            d_indices = [p[1] for p in matched_pairs]

            past_centers = track_centers[t_indices]
            curr_centers = response[d_indices]

            # === 批量计算方向 (Tensor 运算) ===
            dy = curr_centers[:, -2] - past_centers[:, -2]
            dx = curr_centers[:, -1] - past_centers[:, -1]
            
            angles = torch.atan2(-dy, dx)
            angles = torch.remainder(angles, 2 * torch.pi) 

            # 静止目标方向设为 NaN，因为 float Tensor 无法直接存入 None
            static_mask = (dx == 0) & (dy == 0)
            angles[static_mask] = float('nan')

            # === 写回轨迹 ===
            for idx, (t_i, d_i) in enumerate(matched_pairs):
                tid = track_ids[t_i]
                self.tracks[tid]['center'] = curr_centers[idx]
                self.tracks[tid]['unmatched_count'] = 0
                
                ang = angles[idx].item()
                self.tracks[tid]['direction'] = None if math.isnan(ang) else ang

        # === 5. 未匹配的轨迹 -> unmatched_count+=1 或删除 ===
        all_tracks_set = set(range(len(track_centers)))
        for t_i in all_tracks_set - matched_tracks:
            tid = track_ids[t_i]
            if self.tracks[tid]['unmatched_count'] >= self.max_unmatched:
                del self.tracks[tid]
            else:
                self.tracks[tid]['unmatched_count'] += 1

        # === 6. 未匹配检测 -> 新建轨迹 ===
        all_dets_set = set(range(len(response)))
        for d_i in all_dets_set - matched_dets:
            self.tracks[self.next_track_id] = {
                'center': response[d_i],
                'unmatched_count': 0
            }
            self.next_track_id += 1

        # === 7. 输出结果 ===
        # 统一返回一个 PyTorch Tensor, 形状为 (N, 3)，方便下游网络直接使用
        results = []
        for tid, info in self.tracks.items():
            direction = info.get('direction', None)
            if direction is not None:
                c = info['center']
                # 拼接 [y, x, direction]
                res_tensor = torch.tensor([c[-2], c[-1], direction], dtype=torch.float32, device=device)
                results.append(res_tensor)

        if results:
            return torch.stack(results)
        return torch.empty((0, 3), device=device)
    

