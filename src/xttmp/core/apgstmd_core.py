from collections import deque

import torch
from torch.nn import functional as F

from .base_core import BaseCore
from ..util.create_kernel import create_attention_kernel, create_prediction_kernel
from ..core.math_operator import compute_temporal_conv_inplace


class AttentionModule(BaseCore):
    """
    AttentionModule class for attention mechanism.
    
    This class implements the attention mechanism module in the ApgSTMD.
    """
    
    def __init__(self):
        """
        Constructor method.
        
        Initializes the AttentionModule object.
        """
        super().__init__()
        self.kernal_size = 17
        self.zeta_list = [2, 2.5, 3, 3.5]
        self.theta_list = torch.tensor([0, torch.pi/4, torch.pi/2, 3*torch.pi/4])
        self.alpha = 1
        self.register_buffer('attention_kernel', torch.empty(0))
        self.setup()
    
    def setup(self):
        """
        Initialization method.
        
        Initializes the attention kernel.
        """

        self.r = len(self.zeta_list)
        self.s = len(self.theta_list)
        _attention_kernel = create_attention_kernel(
            self.kernal_size,
            self.zeta_list,
            self.theta_list
        )
        _stacked_kernel = torch.stack([torch.stack(row) for row in _attention_kernel])
        self.attention_kernel.data = _stacked_kernel.reshape(self.r * self.s, 1, self.kernal_size, self.kernal_size)
    
    def forward(self, retina_opt, prediction_map):
        """
        Processing method (Optimized with F.conv2d).
        
        Processes the retina_opt and prediction_map to generate the
        attention-optimal output.
        """
        if prediction_map is None:
            self.Opt = retina_opt
            return self.Opt

        # 1. 准备输入数据
        map_retina_opt = retina_opt * prediction_map
                    
        B, C, H, W = map_retina_opt.shape

        # 为了对每个 Channel 独立应用这 r*s 个卷积核，
        # 我们把 Batch 和 Channel 维度合并，把输入变形为 (B*C, 1, H, W)
        x = map_retina_opt.reshape(B * C, 1, H, W)

        # 2. 单次并发计算所有的卷积
        # 此时输出形状为 (B*C, r*s, H, W)
        conv_out = F.conv2d(x, self.attention_kernel, padding='same')

        # 3. 执行 Min 和 Max 聚合操作
        # 将输出重塑为 (B*C, r, s, H, W) 以便按维度进行聚合
        conv_out = conv_out.view(B * C, self.r, self.s, H, W)
        min_out = torch.min(conv_out, dim=2)[0]  # shape: (B*C, r, H, W)
        attention_response = torch.max(min_out, dim=1)[0]  # shape: (B*C, H, W)

        # 4. 恢复原始形状并计算最终结果
        attention_response = attention_response.view(B, C, H, W)

        self.Opt = retina_opt + self.alpha * attention_response
        
        return self.Opt


class PredictionModule(BaseCore):
    """
    PredictionModule class for ApgSTMD.
    
    This class implements the prediction module in the ApgSTMD.
    """
    
    def __init__(self):
        """
        Constructor method.
        
        Initializes the PredictionModule object.
        """
        super().__init__()
        self.velocity = None
        self.intDeltaT = 25
        self.sizeFilter = 25
        self.numFilter = 8
        self.zeta = 2
        self.eta = 2.5
        self.kappa = 0.02
        self.mu = 0.75
        self.beta = 1
        self.register_buffer('time_attenuation_kernel', torch.empty(0))
        self.register_buffer('prediction_kernel', torch.empty(0))

        self.setup()
    
    def setup(self):
        """
        initiate config for prediction module.
        """        
        self.intDeltaT = max(int(self.intDeltaT), 1)
        
        if self.velocity is None:
            self.velocity = 25 / 4 / self.intDeltaT
        
        _prediction_kernel = create_prediction_kernel(
            self.velocity,
            self.intDeltaT,
            self.sizeFilter,
            self.numFilter,
            self.zeta,
            self.eta
        )
        self.prediction_kernel.data = torch.stack(_prediction_kernel).unsqueeze(1)

        self.time_attenuation_kernel = torch.exp(self.kappa * torch.arange(-self.intDeltaT, 1))

        self.reset()  # 初始化历史帧缓存

    def reset(self):
        self.prediction_gain_buffer = deque(maxlen=self.intDeltaT)
        self.prediction_map_buffer = deque(maxlen=self.intDeltaT)
    
    def forward(self, lobula_opt):
        """
        Processing method (Highly Optimized with Vectorization).
        
        Processes the input lobula_opt to predict motion and update
        prediction map.
        """
        num_direction = lobula_opt.shape[1] 
        
        if len(self.prediction_gain_buffer) > 0:
            # 计算滤波器输入 (广播机制同时处理所有方向通道)
            filter_input = self.mu * lobula_opt + (1 - self.mu) * self.prediction_gain_buffer[0] 
        else:
            filter_input = lobula_opt
        
        # 分组卷积 (Depthwise Convolution)
        # 一次 F.conv2d 计算出全部 num_direction 个通道的空间卷积，彻底消除 for 循环
        prediction_gain = F.conv2d(filter_input, 
                                   self.prediction_kernel, 
                                   padding='same', 
                                   groups=num_direction)
        
        # 更新最新一帧的历史
        self.prediction_gain_buffer.append(prediction_gain)

        # ==================== 2. Prediction Map =====================
        # 在 num_direction (通道) 维度上求和 -> shape: (1, 1, H, W)
        tobe_prediction_map = torch.sum(prediction_gain, dim=1, keepdim=True)

        # ==================== 3. Facilitated STMD Output ============
        temporal_conv_out = compute_temporal_conv_inplace(
            self.prediction_gain_buffer,
            self.time_attenuation_kernel
        )
        
        # 一步计算出所有特征通道的 facilitated_opt -> shape: (1, num_direction, H, W)
        self.Opt = lobula_opt + self.beta * temporal_conv_out

        # ==================== 4. Memorizer update ===================
        max_tobe_pre_map = torch.max(tobe_prediction_map)
        
        self.prediction_map_buffer.append( (tobe_prediction_map > max_tobe_pre_map * 2e-1).squeeze())

        # prediction_map = self.cell_prediction_map[0]
        return self.Opt, self.prediction_map_buffer[0]


