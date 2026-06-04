import math

import torch
from collections import deque

from .base_core import BaseCore
from ..util.create_kernel import create_fracdiff_kernel
from ..core.math_operator import compute_temporal_conv_inplace


class Lamina(BaseCore):
    """
    Lamina layer in ESTMD.
    Pure PyTorch Implementation supporting both IIR (iteration) and FIR (convolution) modes.
    """
    def __init__(self, alpha=0.8, delta=20, mode='iteration'):
        """
        Constructor method.
        
        Parameters:
        - alpha: Fractional differential order (0 < alpha <= 1)
        - delta: Length of the historical buffer for convolution
        - mode: 'iteration' (IIR, fast) or 'conv' (FIR, accurate but slower)
        """
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.mode = mode

        self.register_buffer('frac_kernel', torch.empty(0))  

        self.setup()

    def setup(self):

        _kernel = create_fracdiff_kernel(self.alpha, self.delta)
        self.frac_kernel.data = _kernel

        # 2. 计算迭代模式 (IIR) 的系数
        self.para_cur = _kernel[0].item()
        
        if self.alpha == 1.0:
            self.para_pre = 0.0
        elif 0.0 < self.alpha < 1.0:
            self.para_pre = math.exp(-self.alpha / (1.0 - self.alpha))
        else:
            raise ValueError("Invalid alpha value. Must be in (0, 1].")
        
        self.reset_buffer()

    def reset_buffer(self):
        # 3. 初始化时序状态 (State)
        self.state_ipt = None  
        self.state_opt = None 
        self.buffer = deque(maxlen=self.delta) 

    def forward(self, x):
        """
        Processing method.
        x shape: (B, C, H, W)
        """

        # --- 1. 计算一阶差分 (First order difference) ---
        if self.state_ipt is None:
            diff_x = torch.zeros_like(x)
        else:
            diff_x = x - self.state_ipt
        
        # 使用 .detach() 截断计算图，防止处理长视频时 GPU 显存爆炸
        self.state_ipt = x.detach()

        # --- 2. 选择计算模式 ---
        if self.mode == 'iteration':
            self.output = self._compute_by_iteration(diff_x)
        elif self.mode == 'conv':
            self.output = self._compute_by_conv(diff_x)
        else:
            raise ValueError("Mode must be 'iteration' or 'conv'.")

        return self.output

    def _compute_by_iteration(self, diff_x):
        """IIR (无限脉冲响应) 迭代计算法 - 极速模式"""
        if self.state_opt is None:
            opt = diff_x
        else:
            opt = self.para_cur * diff_x + self.para_pre * self.state_opt
        
        # 同样使用 .detach() 截断历史图
        self.state_opt = opt.detach()
        
        return opt

    def _compute_by_conv(self, diff_x):
        """FIR (有限脉冲响应) 卷积计算法 - 基于历史缓存"""
        self.buffer.append(diff_x)
        
        return compute_temporal_conv_inplace(self.buffer, self.frac_kernel)