import math

import numpy as np
import torch
import torch.nn.functional as F

from .base_core import BaseCore
from ..util.datarecord import CircularList
from .math_operator import GammaDelay
from .math_operator import SurroundInhibition
from ..util.create_kernel import create_direction_inhi_kernel


class Medulla(BaseCore):
    """Medulla class for motion detection."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Medulla object
        super().__init__(device=device)

        # Initialize components
        self.hTm3 = Tm3(device=device)
        self.hMi1Para4 = Mi1(device=device)

        self.hTm2 = Tm2(device=device)
        self.hTm1Para5 = Tm1(device=device)
        self.hTm1Para6 = Tm1(device=device)
        
        # Set parameters for hDelay6Tm1
        self.hTm1Para6.hGammaDelay.order = 8
        self.hTm1Para6.hGammaDelay.tau = 40

        self.cellTm1Ipt = CircularList()

    def init_config(self):
        """Initialization method."""
        # Initializes the delay components
        
        self.hTm1Para5.hGammaDelay.isRecord = False
        self.hTm1Para6.hGammaDelay.isRecord = False

        self.hMi1Para4.init_config()
        self.hTm1Para5.init_config()
        self.hTm1Para6.init_config()

        if self.cellTm1Ipt.initLen == 0:
            self.cellTm1Ipt.initLen = max(
                self.hTm1Para5.hGammaDelay.lenKernel,
                self.hTm1Para6.hGammaDelay.lenKernel
            )
        self.cellTm1Ipt.reset()

    def process(self, MedullaInput):
        """Processing method."""
        # Processes input signals and produces output
        
        # Process Tm3 and Tm2 signals
        tm3Signal = self.hTm3.process(MedullaInput)
        tm2Signal = self.hTm2.process(MedullaInput)
        
        # Process signals with delays
        mi1Para4Signal = self.hMi1Para4.process(tm3Signal)
        
        self.cellTm1Ipt.record_next(tm2Signal)
        tm1Para5Signal = self.hTm1Para5.process(self.cellTm1Ipt)
        tm1Para6Signal = self.hTm1Para6.process(self.cellTm1Ipt)

        # Output signals
        self.Opt = [tm3Signal, mi1Para4Signal, tm1Para5Signal, tm1Para6Signal]
        return self.Opt


class Mi1(BaseCore):
    """Mi1 class for motion detection."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Mi1 object
        super().__init__(device=device)
        
        # Initialize gamma delay component with default parameters
        self.hGammaDelay = GammaDelay(3, 15, device=device)

    def init_config(self):
        """Initialization method."""
        # Initializes the gamma delay component
        self.hGammaDelay.init_config()

    def process(self, tm3Signal):
        """Processing method."""
        # Processes the input signal using the gamma delay component
        
        # Apply gamma delay to the input signal
        mi1Opt = self.hGammaDelay.process(tm3Signal)

        # Set output
        self.Opt = mi1Opt
        return self.Opt


class Tm1(BaseCore):
    """Tm1 class for motion detection."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Tm1 object
        super().__init__(device=device)
        
        # Initialize gamma delay component with default parameters
        self.hGammaDelay = GammaDelay(5, 25, device=device)

    def init_config(self):
        """Initialization method."""
        # Initializes the gamma delay component
        self.hGammaDelay.init_config()

    def process(self, tm2Signal):
        """Processing method."""
        # Processes the input signal using the gamma delay component
        
        # Apply gamma delay to the input signal
        tm1Opt = self.hGammaDelay.process(tm2Signal)

        # Set output
        self.Opt = tm1Opt
        return self.Opt


class Tm2(BaseCore):
    """Tm2 class for motion detection."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Tm2 object
        super().__init__(device=device)

    def init_config(self):
        """Initialization method."""
        # This method is not used in this class
        pass

    def process(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values
        if self.device != 'cpu':
            tm2Opt = torch.clamp(-iptMatrix, min=0)
        else:
            tm2Opt = np.maximum(-iptMatrix, 0)
        self.Opt = tm2Opt  # Update output
        return self.Opt


class Tm3(BaseCore):
    """Tm3 class for motion detection."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Tm3 object
        super().__init__(device=device)

    def init_config(self):
        """Initialization method."""
        # This method is not used in this class
        pass

    def process(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values
        if self.device != 'cpu':
            tm3Opt = torch.clamp(iptMatrix, min=0)
        else:
            tm3Opt = np.maximum(iptMatrix, 0)
        self.Opt = tm3Opt  # Update output
        return self.Opt


class Lobula(BaseCore):
    """Lobula class for motion detection."""

    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__(device=device)
        self.alpha1 = 3  # Alpha parameter
        self.thetaList = [(i * math.pi / 4) for i in range(8)]  # List of theta values
        self.hLateralInhi = SurroundInhibition(device=device, channel_size=len(self.thetaList))  # Lateral inhibition component
        self.hDirectionInhi = DirectionInhi(device=device)  # Directional inhibition component

    def init_config(self):
        """Initialization method."""
        # Initializes the lateral and directional inhibition components
        self.hLateralInhi.init_config()
        self.hDirectionInhi.init_config()

    def process(self, lobulaIpt):
        
        tm3Signal, mi1Para4Signal, tm1Para5Signal, tm1Para6Signal = lobulaIpt

        if self.device != 'cpu':    
            # tm3, mi1_p4, tm1_p5, tm1_p6 形状均为 [1, 1, H, W]
            tm3, mi1_p4, tm1_p5, tm1_p6 = lobulaIpt
            _, _, imgH, imgW = tm3.shape
            num_thetas = len(self.thetaList)
            a1 = self.alpha1

            # 2. 计算偏移索引
            theta_tensor = torch.tensor(self.thetaList, device=self.device)
            angles = theta_tensor + torch.pi / 2
            shifts_x = torch.round(a1 * torch.cos(angles)).long() # 形状: [num_thetas]
            shifts_y = torch.round(a1 * torch.sin(angles)).long() # 形状: [num_thetas]

            # 3. 提取中心 ROI
            x_s, x_e = a1, imgH - a1
            y_s, y_e = a1, imgW - a1
            
            # 提取不变部分的 ROI，保持 4D 形状: [1, 1, h_roi, w_roi]
            tm3_roi = tm3[:, :, x_s:x_e, y_s:y_e]
            tm1_p5_roi = tm1_p5[:, :, x_s:x_e, y_s:y_e]

            # 4. 生成偏移索引网格 (关键点)
            h_roi, w_roi = x_e - x_s, y_e - y_s
            grid_y, grid_x = torch.meshgrid(
                torch.arange(x_s, x_e, device=self.device),
                torch.arange(y_s, y_e, device=self.device),
                indexing='ij'
            )

            # 计算所有方向的索引: [num_thetas, h_roi, w_roi]
            src_idx_x = grid_x.unsqueeze(0) - shifts_y.view(-1, 1, 1) # 注意：y对应W，x对应H
            src_idx_y = grid_y.unsqueeze(0) - shifts_x.view(-1, 1, 1)

            # 5. 高级索引提取偏移信号
            # mi1_p4[0, 0] 是 [H, W]，通过 src_idx 提取后变成 [num_thetas, h_roi, w_roi]
            # 我们将其扩展回 4D: [1, num_thetas, h_roi, w_roi]
            mi1_p4_shifted = mi1_p4[0, 0, src_idx_y, src_idx_x].unsqueeze(0)
            tm1_p6_shifted = tm1_p6[0, 0, src_idx_y, src_idx_x].unsqueeze(0)

            # 6. 计算相关输出 (利用广播)
            # tm3_roi: [1, 1, h_roi, w_roi]
            # mi1_p4_shifted: [1, num_thetas, h_roi, w_roi]
            # 结果 corre_roi: [1, num_thetas, h_roi, w_roi]
            corre_roi = tm3_roi * (tm1_p5_roi + mi1_p4_shifted) * tm1_p6_shifted

            # 7. 填回全零张量
            correOutput = torch.zeros((1, num_thetas, imgH, imgW), device=self.device)
            correOutput[:, :, x_s:x_e, y_s:y_e] = corre_roi

            lateralInhiOpt = self.hLateralInhi.process(correOutput)

        else:
            imgH, imgW = tm3Signal.shape
            theta_arr = np.array(self.thetaList)
            numDict = len(theta_arr)
            a1 = self.alpha1

            # 1. 预计算偏移量 (向量化计算)
            # 提前加上 pi/2 减少循环内运算
            angles = theta_arr + np.pi / 2
            shiftsX = np.round(a1 * np.cos(angles)).astype(int)
            shiftsY = np.round(a1 * np.sin(angles)).astype(int)

            # 2. 预提取不需要偏移的静态区域 (ROI)
            # 这样在循环内只需要处理偏移部分
            x_start, x_end = a1, imgH - a1
            y_start, y_end = a1, imgW - a1
            
            tm3_roi = tm3Signal[x_start:x_end, y_start:y_end]
            tm1P5_roi = tm1Para5Signal[x_start:x_end, y_start:y_end]

            # 3. 初始化 3D 输出数组 (比 list 效率更高)
            correOutput = np.zeros((numDict, imgH, imgW))

            # 4. 优化后的循环
            for i in range(numDict):
                sx, sy = shiftsX[i], shiftsY[i]
                
                # 计算偏移后的起始和结束索引
                # 原代码：shiftXRange = slice(a1 - sx, imgH - a1 - sx)
                sx_s, sx_e = a1 - sx, imgH - a1 - sx
                sy_s, sy_e = a1 - sy, imgW - a1 - sy

                # 执行核心计算
                # 减少了对 self 属性的访问和重复切片对象的创建
                correOutput[i, x_start:x_end, y_start:y_end] = (
                    tm3_roi 
                    * (tm1P5_roi + mi1Para4Signal[sx_s:sx_e, sy_s:sy_e]) 
                    * tm1Para6Signal[sx_s:sx_e, sy_s:sy_e]
                )

            # 5. 执行抑制处理
            lateralInhiOpt = [self.hLateralInhi.process(correOutput[i]) for i in range(numDict)]

        lobulaOpt = self.hDirectionInhi.process(lateralInhiOpt)

        self.Opt = lobulaOpt
        return lobulaOpt


class DirectionInhi(BaseCore):
    """Directional inhibition in DSTMD."""

    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the DirectionInhi object
        super().__init__(device=device)
        self.direction = 8  # Number of directions
        self.sigma1 = 1.5  # Sigma for the first Gaussian kernel
        self.sigma2 = 3.0  # Sigma for the second Gaussian kernel
        self.diretionalInhiKernel = None  # Directional inhibition kernel

    def init_config(self):
        """Initialization method."""
        # Initializes the directional inhibition kernel
        if self.diretionalInhiKernel is None:
            self.diretionalInhiKernel = create_direction_inhi_kernel(
                self.direction, self.sigma1, self.sigma2
            )
        if self.device != 'cpu':
            self.diretionalInhiKernel = torch.from_numpy(self.diretionalInhiKernel).float().cuda()
        else:
            self.diretionalInhiKernel = self.diretionalInhiKernel.squeeze()

    def process(self, iptCell):
        """Processing method."""
        # Performs directional inhibition on the input
        
        len1 = len(iptCell)
        len2 = len(self.diretionalInhiKernel)
        certer = len2 // 2
        

        if self.device != 'cpu':
            # 1. 准备数据维度
            # 输入是 [1, C, H, W]，卷积需要在 C 维度上滑，所以要把 H, W 暂时视为 Batch
            b, c, h, w = iptCell.shape
            
            # 转换形状: [1, C, H, W] -> [1, C, H*W] -> [H*W, 1, C]
            # 这样对于 conv1d 来说，BatchSize = H*W, 通道数 = 1, 序列长度 = C
            x = iptCell.view(c, -1).permute(1, 0).unsqueeze(1) 
            
            
            # 3. 执行循环卷积 (Circular Convolution)
            # padding 设为 kernel_size // 2，且模式设为 'circular'
            pad_size = self.diretionalInhiKernel.shape[-1]
            center_idx = pad_size // 2
            
            pad_left = center_idx
            pad_right = pad_size - center_idx - 1
            
            # F.pad 在 1D 信号上的填充格式是 (left, right)
            x_padded = F.pad(x, (pad_left, pad_right), mode='circular')

            # F.conv1d 会在 C 维度（方向轴）上滑动
            # 结果形状依然是 [H*W, 1, C]
            result = F.conv1d(x_padded, self.diretionalInhiKernel)
            
            # 4. 激活与恢复形状
            # np.maximum(result, 0) 对应 F.relu
            opt = F.relu(result)
            
            # 恢复回 [1, C, H, W]
            opt = opt.squeeze(1).permute(1, 0).view(b, c, h, w)

        else:
            opt = []
            for idx in range(len1):
                result = np.zeros_like(iptCell[0])
                matrixPoint = idx
                kernelPoint = certer

                for shiftPoint in range(len(self.diretionalInhiKernel)):
                    matrixPoint = idx    - shiftPoint
                    kernelPoint = certer - shiftPoint
                    '''
                    This takes advantage of the fact that the convolution kernel is symmetric, 
                        there is no flip convolution kernel
                    '''
                    result += iptCell[matrixPoint] * self.diretionalInhiKernel[kernelPoint]

                opt.append(np.maximum(result, 0))

        return opt
    


