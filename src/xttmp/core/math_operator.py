from collections import deque
from typing import Iterable

import torch
import torch.nn.functional as F

from .base_core import BaseCore
from ..util.create_kernel import create_2d_gaussian_kernel, create_gamma_kernel, create_spatial_inhibition_kernel


def compute_temporal_conv_inplace(buffer_refs: Iterable[torch.Tensor], 
                                  time_kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Efficiently computes a 1D temporal convolution over a sequence of spatial feature maps
    using in-place accumulation and automatic sequence truncation.

    This function is designed for streaming video or continuous time-series processing 
    where historical frames are stored in a FIFO queue (like collections.deque). It maps 
    a 1D temporal kernel across the batch and spatial dimensions of the buffered frames.

    Args:
        buffer_refs (Iterable[torch.Tensor]): An iterable (e.g., list or deque) containing 
            historical feature map tensors. 
            - Expected shape of each tensor: [B, C, H, W]
            - Temporal ordering: The rightmost (last) element is assumed to be the 
              newest/most recent frame.
        time_kernel_tensor (torch.Tensor): A 1D tensor representing the temporal convolution 
            weights. 
            - Expected shape: [K], where K is the kernel size.
            - Weight ordering: The leftmost (first) element corresponds to the newest frame.

    Returns:
        torch.Tensor: The result of the temporal convolution, with shape [B, C, H, W]. 
        Returns None if `buffer_refs` is empty.

    Notes:
        - Memory Efficiency: Uses a clone for the initial base tensor, followed by in-place 
          additions (`add_`) for subsequent historical frames to minimize memory allocation.
        - Automatic Truncation: The `zip` function automatically stops at the shortest 
          iterable. If the buffer has fewer frames than the kernel size (e.g., during the 
          initial "warm-up" phase of a stream), it safely computes a partial convolution 
          without out-of-bounds errors or requiring explicit padding.
    """
    temporal_conv_out = None
    
    # reversed(buffer_refs) iterates from the newest frame to the oldest frame.
    # zip automatically aligns the newest frame with the first weight and truncates safely.
    for t_tensor, weight in zip(reversed(buffer_refs), time_kernel_tensor):
        
        weight_val = weight.item()
        
        if temporal_conv_out is None:
            # Initialize the base tensor using the newest frame
            temporal_conv_out = t_tensor.clone().mul_(weight_val)
        else:
            # In-place accumulation of historical frames onto the base tensor
            temporal_conv_out.add_(t_tensor, alpha=weight_val)
            
    return temporal_conv_out


class GaussianBlur(BaseCore):
    """
    Gaussian blur filter: Pure PyTorch implementation.
    """

    def __init__(self, kernel_size=3, sigma=1.0):
        """
        Constructor.
        Initializes the GaussianBlur module.

        Parameters:
        - kernel_size: Size of the filter kernel (int). Should be an odd number.
        - sigma: Standard deviation of the Gaussian distribution (float).
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('blur_kernel', torch.empty(0))

        self.setup()

    def setup(self):
        _kernel = create_2d_gaussian_kernel(self.kernel_size, self.sigma)
        self.blur_kernel.data = _kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        """
        Processing method.
        Applies the Gaussian filter to the input tensor.

        Parameters:
        - x: Input tensor of shape (B, C, H, W)

        Returns:
        - opt: Output after applying the Gaussian filter.
        """

        C = x.shape[1]

        # 动态将单通道高斯核扩展至与输入特征图通道数一致, expand 不占用额外显存
        weight = self.blur_kernel.expand(C, 1, self.kernel_size, self.kernel_size)

        # 使用深度可分离卷积（groups=C），每个通道独立进行高斯模糊
        self.output = F.conv2d(x, weight, padding='same', groups=C)

        return self.output
    

class GammaDelay(BaseCore):
    """
    GammaDelay Class
    
    Implements a gamma filter used in the lamina layer of the ESTMD neural network
    using pure PyTorch and collections.deque for efficient temporal sliding windows.
    """
    def __init__(self, order=1, tau=1.0):
        """
        Constructor method.
        
        Parameters:
            order (int): Order of the gamma filter (n). Default is 1.
            tau (float): Time constant of the filter (\tau).
        """
        super().__init__()
        self.order = max(1, int(order))
        self.tau = tau


        self.setup()

    def setup(self):
        _kernel_len = int(3 * self.tau)
        # 1. 预计算 Gamma 滤波器的时域权重
        kernel = create_gamma_kernel(self.order, self.tau, _kernel_len)
        # 注册为 buffer，随模型自动转移设备 (如 .cuda())
        self.register_buffer('gamma_kernel', kernel)

        # 2. 初始化双端队列作为时序状态缓存区
        self.buffer = deque(maxlen=_kernel_len)

    def reset_buffer(self):
        """
        Resets the internal buffer by clearing all stored frames.
        """
        self.buffer.clear()

    def forward(self, x, in_loop=False):
        """
        Processing method.
        Applies the gamma filter to the input tensor.
        
        Parameters:
        - x: Input tensor of shape (B, C, H, W)
        - in_loop (bool): If True, replaces the last frame instead of appending. 
                          (Equivalent to original isInLoop/cover logic)
        """

        if in_loop and len(self.buffer) > 0:
            # 替换队尾元素 (最新帧)，保持缓存长度不变
            self.buffer[-1] = x
        else:
            self.buffer.append(x)

        self.output = compute_temporal_conv_inplace(self.buffer, self.gamma_kernel)

        return self.output


class GammaBandPassFilter(BaseCore):
    """
    GammaBandPassFilter: Temporal Band-pass filter for ESTMD.
    
    Optimized pure PyTorch implementation. Uses a single deque buffer and 
    mathematically fuses the two Gamma filters into a single convolution kernel 
    to halve memory usage and computation time.
    """

    def __init__(self, 
                 order1=2, tau1=3.0, 
                 order2=6, tau2=9.0):
        """
        Constructor method.
        
        Parameters:
        - order1, tau1: Parameters for the excitatory (positive) Gamma filter.
        - order2, tau2: Parameters for the inhibitory (negative) Gamma filter.
        """
        super().__init__()

        self.order1 = max(1, int(order1))
        self.tau1 = tau1
        self.order2 = max(1, int(order2))
        self.tau2 = tau2
        
        self.in_loop = False  # 默认不覆盖历史帧，直接追加

        self.setup()

    def setup(self):
        _kernel_len = max(int(3 * self.tau1), int(3 * self.tau2))
        # 1. 预计算两个 Gamma 滤波器的权重，并补齐到相同的长度 self.T
        k1 = create_gamma_kernel(self.order1, self.tau1, _kernel_len)
        k2 = create_gamma_kernel(self.order2, self.tau2, _kernel_len)

        # 2. 算子融合 (Operator Fusion)：W_bandpass = W1 - W2
        # 直接将差值注册为模型的 buffer，前向传播只需计算一次
        bandpass_kernel = k1 - k2
        self.register_buffer('bandpass_kernel', bandpass_kernel)

        # 3. 初始化单一的高效时序状态缓存区
        self.buffer = deque(maxlen=_kernel_len)

    def reset_buffer(self):
        """
        Resets the internal buffer by clearing all stored frames.
        """
        self.buffer.clear()

    def forward(self, x):
        """
        Processing method.
        
        Parameters:
        - x: Input tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
        
        Returns:
        - opt_tensor: Processed band-pass output tensor
        """

        # 1. 记录最新一帧
        if self.in_loop and len(self.buffer) > 0:
            # 替换队尾元素 (最新帧)，保持缓存长度不变
            self.buffer[-1] = x
        else:
            self.buffer.append(x)

        self.output = compute_temporal_conv_inplace(self.buffer, self.bandpass_kernel)

        return self.output


class SpatialInhibition(BaseCore):
    """
    Gamma_Filter Gamma filter in lamina layer
    Pure PyTorch implementation for Surround Inhibition.
    """

    def __init__(self, 
                 kernel_size=15, 
                 sigma1=1.5, 
                 sigma2=3.0, 
                 e=1.0, 
                 rho=0.0, 
                 A=1.0, 
                 B=3.0):
        """
        Constructor
        Initializes the SurroundInhibition module.
        
        Parameters:
        - kernel_size: Size of the filter kernel
        - sigma1: Standard deviation for the first Gaussian (Center)
        - sigma1: Standard deviation for the second Gaussian (Surround)
        - e: Exponent for the weighting of the second Gaussian
        - rho: Radius for circular integration / Center offset
        - A: Amplitude of the positive center
        - B: Amplitude of the negative surround
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.e = e
        self.rho = rho
        self.A = A
        self.B = B

        self.register_buffer('kernel', torch.empty(0))

        self.setup()

    def setup(self):
        _spatial_inhibiiton_kernel = create_spatial_inhibition_kernel(self.kernel_size, 
                                                                    self.sigma1,
                                                                    self.sigma2,
                                                                    self.e,
                                                                    self.rho,
                                                                    self.A,
                                                                    self.B)
        self.kernel.data = _spatial_inhibiiton_kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        """
        Processing method
        Applies the surround inhibition filter to the input tensor.
        
        Parameters:
        - x: Input tensor of shape (B, C, H, W)
        """
            
        C = x.shape[1]

        # .expand 不会真的在内存中复制数据，而是通过 stride 机制虚拟映射，极大地节省显存和耗时
        weight = self.kernel.expand(C, 1, self.kernel_size, self.kernel_size)

        # groups=C 表示进行深度可分离卷积（Depthwise Convolution），每个通道独立滤波        
        self.output = F.relu(F.conv2d(x, weight, padding='same', groups=C))

        return self.output