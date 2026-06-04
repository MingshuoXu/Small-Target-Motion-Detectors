from collections import deque

import torch
import torch.nn.functional as F

from .base_core import BaseCore
from .estmd_core import Tm1, Mi1
from .math_operator import GammaDelay, SpatialInhibition
from ..util.create_kernel import create_direction_inhi_kernel


class Medulla(BaseCore):
    """Medulla class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Medulla object
        super().__init__()

        # Initialize components
        self.tm3 = Tm3()
        self.mi1_para4 = Mi1(3, 15)

        self.tm2 = Tm2()
        self.tm1_para5 = Tm1(5, 25)
        self.tm1_para6 = Tm1(8, 40)
        
    def setup(self):
        """Initialization method."""
        # Initializes the delay components

        self.mi1_para4.setup()
        self.tm1_para5.setup()
        self.tm1_para6.setup()

    def forward(self, x):
        """Processing method."""
        # Processes input signals and produces output
        
        # Process Tm3 and Tm2 signals
        tm3_output = self.tm3.forward(x) # L ON
        tm2_output = self.tm2.forward(x) # L OFF
        
        # Process signals with delays
        mi1_para4_output = self.mi1_para4.forward(tm3_output)
        tm1_para5_output = self.tm1_para5.forward(tm2_output)
        tm1_para6_output = self.tm1_para6.forward(tm2_output)

        # Output signals
        self.output = (tm3_output, mi1_para4_output, tm1_para5_output, tm1_para6_output)
        return self.output


class Tm2(BaseCore):
    """Tm2 class for motion detection."""

    def forward(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values
        self.output = torch.clamp(-iptMatrix, min=0)

        return self.output


class Tm3(BaseCore):
    """Tm3 class for motion detection."""

    def forward(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values

        self.output = torch.clamp(iptMatrix, min=0)

        return self.output


class Lobula(BaseCore):
    """Lobula class for motion detection."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.alpha1 = 3  # Alpha parameter

        self.register_buffer('theta_list', torch.tensor([(i * torch.pi / 4) for i in range(8)]))
        self.hLateralInhi = SpatialInhibition()  # Lateral inhibition component
        self.hDirectionInhi = DirectionInhibition()  # Directional inhibition component

    def setup(self):
        """Initialization method."""
        # Initializes the lateral and directional inhibition components
        self.hLateralInhi.setup()
        self.hDirectionInhi.setup()

    def forward(self, tm3, mi1_p4, tm1_p5, tm1_p6):
        # tm3, mi1_p4, tm1_p5, tm1_p6 形状均为 [1, 1, H, W]
        _, _, imgH, imgW = tm3.shape
        device = tm3.device

        num_thetas = len(self.theta_list)
        a1 = self.alpha1

        # 2. 计算偏移索引
        shifts_x = torch.round(a1 * torch.cos(self.theta_list)).long() # 形状: [num_thetas]
        shifts_y = torch.round(a1 * torch.sin(self.theta_list)).long() # 形状: [num_thetas]

        # 3. 提取中心 ROI
        y_s, y_e = a1, imgH - a1
        x_s, x_e = a1, imgW - a1
        
        # 提取不变部分的 ROI，保持 4D 形状: [1, 1, h_roi, w_roi]
        tm3_roi = tm3[:, :, y_s:y_e, x_s:x_e]
        tm1_p5_roi = tm1_p5[:, :, y_s:y_e, x_s:x_e]

        # 4. 生成偏移索引网格 (关键点)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(y_s, y_e, device=device),
            torch.arange(x_s, x_e, device=device),
            indexing='ij'
        )

        # 计算所有方向的索引: [num_thetas, h_roi, w_roi]
        src_idx_x = grid_x.unsqueeze(0).unsqueeze(0) - shifts_x .view(1, -1, 1, 1)
        src_idx_y = grid_y.unsqueeze(0).unsqueeze(0) + shifts_y.view(1, -1, 1, 1)

        # 5. 高级索引提取偏移信号
        # mi1_p4[0, 0] 是 [H, W]，通过 src_idx 提取后变成 [num_thetas, h_roi, w_roi]
        # 我们将其扩展回 4D: [1, num_thetas, h_roi, w_roi]
        mi1_p4_shifted = mi1_p4[0, 0, src_idx_y, src_idx_x]
        tm1_p6_shifted = tm1_p6[0, 0, src_idx_y, src_idx_x]

        # 6. 计算相关输出 (利用广播)
        # tm3_roi: [1, 1, h_roi, w_roi]
        # mi1_p4_shifted: [1, num_thetas, h_roi, w_roi]
        # 结果 corre_roi: [1, num_thetas, h_roi, w_roi]
        corre_roi = tm3_roi * (tm1_p5_roi + mi1_p4_shifted) * tm1_p6_shifted

        # 7. 填回全零张量
        correOutput = torch.zeros((1, num_thetas, imgH, imgW), device=device)
        correOutput[:, :, y_s:y_e, x_s:x_e] = corre_roi

        lateralInhioutput = self.hLateralInhi.forward(correOutput)
        self.output = self.hDirectionInhi.forward(lateralInhioutput)

        return self.output


class DirectionInhibition(BaseCore):
    """Directional inhibition in DSTMD."""

    def __init__(self):
        """Constructor method."""
        # Initializes the DirectionInhi object
        super().__init__()
        self.direction = 8  # Number of directions
        self.sigma1 = 1.5  # Sigma for the first Gaussian kernel
        self.sigma2 = 3.0  # Sigma for the second Gaussian kernel

        self.register_buffer("diretional_inhi_kernel", torch.empty(0))  # Placeholder for the directional inhibition kernel

        self.setup()  # Initialize the kernel

    def setup(self):
        """Initialization method."""
        # Initializes the directional inhibition kernel

        _diretional_inhi_kernel = create_direction_inhi_kernel(
                self.direction, self.sigma1, self.sigma2
            )
        # Shape: [1, 1, kernel_size]
        self.diretional_inhi_kernel.data = _diretional_inhi_kernel

    def forward(self, x):
        """
            Input x shape: [B, C, H, W], where C = self.direction
            Output shape: [B, C, H, W]        
        """
        # Performs directional inhibition on the input

        # 1. 准备数据维度
        # 输入是 [B, C, H, W]，卷积需要在 C 维度上滑，所以要把 H, W 暂时视为 Batch
        b, c, h, w = x.shape
        
        # 转换形状: [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        # 这样对于 conv1d 来说，BatchSize = B*H*W, 通道数 = 1, 序列长度 = C
        x = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        
        # 3. 执行循环卷积 (Circular Convolution)
        # padding 设为 kernel_size // 2，且模式设为 'circular'
        pad_size = self.diretional_inhi_kernel.shape[-1]
        center_idx = pad_size // 2
        
        pad_left = center_idx
        pad_right = pad_size - center_idx - 1
        
        # F.pad 在 1D 信号上的填充格式是 (left, right)
        x_padded = F.pad(x, (pad_left, pad_right), mode='circular')

        # F.conv1d 会在 C 维度（方向轴）上滑动
        # 结果形状依然是 [B*H*W, 1, C]
        result = F.conv1d(x_padded, self.diretional_inhi_kernel)
        
        # 4. 激活与恢复形状
        opt = F.relu(result)
        
        # 恢复回 [B, C, H, W]
        return opt.reshape(b, h, w, c).permute(0, 3, 1, 2)


    


