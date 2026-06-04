from collections import deque

import torch
import torch.nn.functional as F

from .base_core import BaseCore
from .math_operator import (compute_temporal_conv_inplace,
                            GaussianBlur, SpatialInhibition, 
                            GammaDelay, GammaBandPassFilter)
from ..util.create_kernel import create_2d_gaussian_kernel


class Retina(GaussianBlur):
    pass


class Lamina(BaseCore):
    """
    LAMINA Lamina layer
    This class implements the Lamina layer of the ESTMD
    
    Author: Mingshuo Xu
    Date: 2024-04-29
    """
    
    def __init__(self):
        """
        Constructor
        Initializes the Lamina object and creates GammaBankPassFilter
        and LaminaLateralInhibition objects
        """
        super().__init__()
        self.gamma_BPF = GammaBandPassFilter()
        self.spatial_inhibition = LaminaLateralInhibition()

    def setup(self):
        self.gamma_BPF.setup()
        self.spatial_inhibition.setup()

    def reset_buffer(self):
        self.gamma_BPF.reset_buffer()

    def forward(self, laminaIpt):
        """
        Processing method
        Applies GammaBankPassFilter and LaminaLateralInhibition to the input matrix
        
        Parameters:
        - laminaIpt: Input matrix
        
        Returns:
        - laminaOpt: Processed output matrix
        """
        signalWithBPF = self.gamma_BPF.forward(laminaIpt)
        self.output = self.spatial_inhibition.forward(signalWithBPF)

        return self.output


class Medulla(BaseCore):
    """
    Medulla Layer of DSTMD
    This class implements the Medulla layer of the ESTMD.
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Medulla object
        """
        super().__init__()
        self.tm1 = Tm1(order=12, tau=25)  # Initialize Tm1 object
        self.tm2 = Tm2()  # Initialize Tm2 object
        self.tm3 = Tm3()  # Initialize Tm3 object
        self.mi1 = Mi1(order=12, tau=25)  # Initialize Tm3 object

    def setup(self):
        """
        Initialization method
        Initializes the Tm1, Tm2, and Tm3 objects
        """
        self.tm1.setup()
        self.tm2.setup()
        self.tm3.setup()
        self.mi1.setup()

    def reset_buffer(self):
        """
        Buffer reset method
        Resets the buffers of Tm1, Tm2, and Tm3 objects
        """
        self.tm1.reset_buffer()
        self.mi1.reset_buffer()

    def forward(self, x):
        """
        Processing method
        Processes the input MedullaIpt through Tm1, Tm2, and Tm3 layers
        
        Parameters:
        - MedullaIpt: Input matrix
        
        Returns:
        - tm3Signal: Output of Tm3 layer
        - tm1Signal: Output of Tm1 layer
        """
        tm2_output = self.tm2.forward(x)  # Process input through Tm2
        tm3_output = self.tm3.forward(x)  # Process input through Tm3

        tm1_output = self.tm1.forward(tm2_output)  # Process Tm2 output through Tm1

        self.output = (tm3_output, tm1_output)  # Update output property with output
        return tm3_output, tm1_output


class Lobula(BaseCore):
    """
    Lobula Layer of DSTMD
    This class implements the Lobula layer of the ESTMD.
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Lobula object
        """
        super().__init__()
        self.a = 0  # Parameter a
        self.b = 0  # Parameter b
        self.c = 1  # Parameter c

    def setup(self):
        """
        Initialization method
        """
        pass

    def forward(self, onSignal, offSignal):
        """
        Processing method
        Processes the input ON and OFF signals
        
        Parameters:
        - varagein: Tuple containing ON and OFF signals
        
        Returns:
        - lobulaOpt: Output of the Lobula layer
        """
        
        # Compute Lobula output using the provided formula
        self.output = self.a*onSignal + self.b*offSignal + self.c*onSignal*offSignal
        
        return self.output


class Mi1(GammaDelay):
    pass


class Tm1(GammaDelay):
    pass


class Tm2(BaseCore):
    """
    Tm2 Medulla Layer Neurons in ESTMD
    """
   
    def __init__(self, device='cpu'):
        """
        Constructor method
        Initializes the Tm2 object
        """
        super().__init__()
        self.spatial_inhibition = SpatialInhibition()  # Initialize SurroundInhibition object

    def setup(self):
        """
        Initialization method
        Initializes the SurroundInhibition object
        """
        self.spatial_inhibition.setup()

    def forward(self, x):
        """
        Processing method
        Applies the Surround Inhibition mechanism to the input matrix iptMatrix
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - tm2Opt: Output of the Tm2 layer
        """
        # Extract the OFF signal from iptMatrix
        L_OFF = torch.clamp(-x, min=0)

        # Process the OFF signal using SurroundInhibition
        self.output = self.spatial_inhibition.forward(L_OFF)  

        return self.output


class Tm3(BaseCore):
    """ Tm3  """
    
    def __init__(self, device='cpu'):
        """ Constructor method

        Initializes the Tm3 object
        """
        super().__init__()
        self.spatial_inhibition = SpatialInhibition()  # Initialize SurroundInhibition object

    def setup(self):
        """ Initialization method

        Initializes the SurroundInhibition object
        """
        self.spatial_inhibition.setup()

    def forward(self, iptMatrix):
        """ Processing method
        
        Description:
            Applies Surround Inhibition to the On-signal matrix iptMatrix
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - tm3Opt: Output of the Tm3 layer
        """

        L_ON = torch.clamp(iptMatrix, min=0)

        self.output = self.spatial_inhibition.forward(L_ON)  # Processes the On-signal using SurroundInhibition
  
        return self.output


class LaminaLateralInhibition(BaseCore):
    """ LAMINALATERALINHIBITION Lateral inhibition in the Lamina layer
    
    This class implements the lateral inhibition mechanism in the Lamina layer
    of the ESTMD using pure PyTorch operations.

    References:
    * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    for the detection of moving targets in visual clutter inspired by
    insect physiology, PLoS ONE 3 (7) (2008) e2784.
    * Wang H, Peng J, Yue S. A directionally selective small target
    motion detecting visual neural network in cluttered backgrounds[J].
    IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    """

    def __init__(self, 
                 sizeW1=[11, 11, 7], 
                 lambda1=3.0, 
                 lambda2=9.0, 
                 sigma1=1.5, 
                 sigma2=None):
        """
        Constructor
        Initializes the LaminaLateralInhibition module
        """
        super().__init__()
        self.sizeW1 = sizeW1
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma1 = sigma1
        self.sigma2 = 2.0 * sigma1 if sigma2 is None else sigma2
        self.T = sizeW1[2]  # Temporal length

        self.register_buffer('spatial_pos_kernel', torch.empty(0))
        self.register_buffer('spatial_neg_kernel', torch.empty(0))
        self.register_buffer('temporal_pos_kernel', torch.empty(0))
        self.register_buffer('temporal_neg_kernel', torch.empty(0))

        self.setup()  # Initialize kernels and buffers

    def _create_spatial_kernels(self):
        """初始化 DoG (Difference of Gaussian) 空间感受野权重"""
        g_sigma2 = create_2d_gaussian_kernel(self.sizeW1[:2], self.sigma1)
        g_sigma3 = create_2d_gaussian_kernel(self.sizeW1[:2], self.sigma2)
        diff_of_gaussian = g_sigma2 - g_sigma3

        # W_{S}^{P} 和 W_{S}^{N}
        pos_kernel = torch.clamp(diff_of_gaussian, min=0)
        neg_kernel = torch.clamp(diff_of_gaussian, max=0)

        # 调整形状为 (out_channels=1, in_channels=1, H, W) 以匹配 F.conv2d 的需求
        pos_kernel = pos_kernel.view(1, 1, *self.sizeW1[:2])
        neg_kernel = neg_kernel.view(1, 1, *self.sizeW1[:2])
        
        return pos_kernel, neg_kernel

    def _create_temporal_kernels(self):
        """初始化时间衰减权重"""
        t = torch.arange(self.T, dtype=torch.float32)
        
        # W_{T}^{P} 和 W_{T}^{N}
        w_t_pos = torch.exp(-t / self.lambda1) / self.lambda1
        w_t_neg = torch.exp(-t / self.lambda2) / self.lambda2
        
        # 调整形状为 (T, 1, 1, 1) 方便后续与 (T, Batch, C, H, W) 张量进行广播乘法
        return w_t_pos.view(-1, 1, 1, 1), w_t_neg.view(-1, 1, 1, 1)

    def setup(self):
        # 1. 预计算空间卷积核 (Spatial Kernels)
        spatial_pos, spatial_neg = self._create_spatial_kernels()
        # 使用 register_buffer，这样模型调用 .cuda() 或 .to(device) 时，核也会自动转移
        self.spatial_pos_kernel.data = spatial_pos
        self.spatial_neg_kernel.data = spatial_neg

        # 2. 预计算时间卷积核 (Temporal Kernels)
        temporal_pos, temporal_neg = self._create_temporal_kernels()
        self.temporal_pos_kernel.data = temporal_pos
        self.temporal_neg_kernel.data = temporal_neg

        # 3. 初始化时序状态缓存区 (采用双端队列 deque 实现高效的滑动窗口)
        self.pos_buffer = deque(maxlen=self.T)
        self.neg_buffer = deque(maxlen=self.T)

    def reset_buffer(self):
        """重置时序状态缓存区"""
        self.pos_buffer.clear()
        self.neg_buffer.clear()

    def forward(self, x):
        """
        输入: 
            x: 形状为 (B, C, H, W) 或 (H, W) 的张量
        """

        # === 1. 空间侧抑制 (Spatial Lateral Inhibition) ===
        on_conv = F.conv2d(x, self.spatial_pos_kernel, padding='same')
        off_conv = F.conv2d(x, self.spatial_neg_kernel, padding='same')

        # 记录当前帧结果到时序缓存区 (新的帧在队列右侧)
        self.pos_buffer.append(on_conv)
        self.neg_buffer.append(off_conv)

        # === 2. 时序卷积 (Temporal Convolution) ===

        pos_out = compute_temporal_conv_inplace(self.pos_buffer, self.temporal_pos_kernel)
        neg_out = compute_temporal_conv_inplace(self.neg_buffer, self.temporal_neg_kernel)

        return pos_out + neg_out

    





