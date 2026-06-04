from collections import deque

import torch
from torch.nn import functional as F

from .base_core import BaseCore
from .math_operator import (compute_temporal_conv_inplace, 
                            SpatialInhibition, GammaDelay)



class Medulla(BaseCore):
    def __init__(self):
        super().__init__()
        self.temporal_kernel_len = 15 # a
        self.spatial_kernel_size = 7 # r
        self.register_buffer('theta_list', torch.tensor([(i * torch.pi / 4) for i in range(8)]))

        self.register_buffer('temporal_ON_kernel', torch.empty(0))
        self.register_buffer('temporal_OFF_kernel', torch.empty(0))

        self.delay_on = GammaDelay(10, 1)
        self.delay_off = GammaDelay(10, 1)

        self.medulla_input_buffer = deque(maxlen=self.temporal_kernel_len)

        self.setup()

    def setup(self):

        # Temporal kernels
        k1 = int(self.temporal_kernel_len / 2)

        self.temporal_ON_kernel = torch.ones((k1, 1))
        self.temporal_OFF_kernel = torch.vstack((torch.zeros((k1, 1)), -torch.ones((k1+1, 1))))

        self.reset_buffer()

    def reset_buffer(self):
        # Allocate memory
        self.medulla_input_buffer.clear()
        self.delay_on.reset_buffer()
        self.delay_off.reset_buffer()

    @staticmethod
    def direction_pooling(x, s=7):
        """
        高效计算 8 个方向的感受野池化 (均值)
        x: 输入特征图 [B, C, H, W]
        s: 池化窗口大小 (建议为奇数，例如 3, 5, 7)
        """
        H, W = x.shape[-2:]
        
        # 1. 确定最大偏移量：窗口大小减 1
        p = s - 1
        
        # 2. 四周补齐 Padding
        # 补齐后尺寸变为 (H + 2p, W + 2p)
        padded_x = F.pad(x, (p, p, p, p))
        
        # 3. 仅做一次全局 AvgPool
        # 步长设为 1，输出尺寸会自动变成 (H + p, W + p)
        pooled = F.avg_pool2d(padded_x, kernel_size=(s, s), stride=1, padding=0)
        pooled_on = torch.clamp(pooled, min=0)  
        pooled_off = torch.clamp(-pooled, min=0)  
        
        # --- 4. 见证奇迹的切片时刻 ---
        # 定义不同方向的起始坐标 (基于偏移量 p)
        offset_min = 0          # 偏向上/左
        offset_mid = p // 2     # 居中对齐
        offset_max = p          # 偏向下/右
        
        spatial_ON_output = torch.cat([
            pooled_on[..., offset_mid : offset_mid+H, offset_min : offset_min+W], # W
            pooled_on[..., offset_max : offset_max+H, offset_min : offset_min+W], # SW
            pooled_on[..., offset_max : offset_max+H, offset_mid : offset_mid+W], # S
            pooled_on[..., offset_max : offset_max+H, offset_max : offset_max+W], # SE
            pooled_on[..., offset_mid : offset_mid+H, offset_max : offset_max+W], # E
            pooled_on[..., offset_min : offset_min+H, offset_max : offset_max+W], # NE
            pooled_on[..., offset_min : offset_min+H, offset_mid : offset_mid+W], # N
            pooled_on[..., offset_min : offset_min+H, offset_min : offset_min+W], # NW
        ], dim=1)

        spatial_OFF_output = torch.cat([
            pooled_off[..., offset_mid : offset_mid+H, offset_max : offset_max+W], # E
            pooled_off[..., offset_min : offset_min+H, offset_max : offset_max+W], # NE
            pooled_off[..., offset_min : offset_min+H, offset_mid : offset_mid+W], # N
            pooled_off[..., offset_min : offset_min+H, offset_min : offset_min+W], # NW
            pooled_off[..., offset_mid : offset_mid+H, offset_min : offset_min+W], # W
            pooled_off[..., offset_max : offset_max+H, offset_min : offset_min+W], # SW
            pooled_off[..., offset_max : offset_max+H, offset_mid : offset_mid+W], # S
            pooled_off[..., offset_max : offset_max+H, offset_max : offset_max+W], # SE
        ], dim=1)
        
        return spatial_ON_output, spatial_OFF_output
        
    def forward(self, medullaIpt):

        ''' Compute temporal part '''
        self.medulla_input_buffer.append(medullaIpt)

        temporal_ON_output = compute_temporal_conv_inplace(self.medulla_input_buffer, self.temporal_ON_kernel)
        temporal_OFF_output = compute_temporal_conv_inplace(self.medulla_input_buffer, self.temporal_OFF_kernel)

        # There's no need for half-wave rectification here
        correlated_temporal_output = temporal_ON_output * temporal_OFF_output

        ''' Compute spacial part '''
        spatial_ON_output, spatial_OFF_output = self.direction_pooling(medullaIpt, self.spatial_kernel_size)

        delayed_spatial_ON_output = self.delay_on.forward(spatial_ON_output)
        delayed_spatial_OFF_output = self.delay_off.forward(spatial_OFF_output)

        correlated_spatial_output = delayed_spatial_ON_output * delayed_spatial_OFF_output

        # Store the output in output property
        self.output = (correlated_spatial_output, correlated_temporal_output)

        return self.output


class Lobula(BaseCore):
    def __init__(self):
        super().__init__()
        self.tau = 1  # a parameter to align the spacialOpt and temporalOpt
        self.spatial_inhibition = SpatialInhibition()
        self.spatial_inhibition.B = 1

    def setup(self):
        self.spatial_inhibition.setup()

    def forward(self, correlated_spatial_output, correlated_temporal_output):
        correlated_spatiotemporal_output = correlated_spatial_output * correlated_temporal_output

        # Apply surround inhibition
        self.output = torch.clamp(self.spatial_inhibition(correlated_spatiotemporal_output), min=0)

        return self.output


