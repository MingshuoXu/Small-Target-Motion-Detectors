import numpy as np
import cv2

from .base_core import BaseCore
from ..util.create_kernel import create_attention_kernel, create_prediction_kernel
from ..util.compute_module import compute_temporal_conv

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
        self.theta_list = [0, np.pi/4, np.pi/2, np.pi*3/4]
        self.alpha = 1
        self.attention_kernel = None
    
    def init_config(self):
        """
        Initialization method.
        
        Initializes the attention kernel.
        """
        self.attention_kernel = create_attention_kernel(
            self.kernal_size,
            self.zeta_list,
            self.theta_list
        )
    
    def process(self, retina_opt, prediction_map):
        """
        Processing method.
        
        Processes the retina_opt and prediction_map to generate the
        attention-optimal output.
        """
        r = len(self.attention_kernel)
        s = len(self.attention_kernel[0])

        if prediction_map is None:
            attention_opt = retina_opt
        else:
            map_retina_opt = retina_opt * prediction_map
            attention_response = None
            
            for i in range(r):
                for j in range(s):
                    if j == 0:
                        attention_response_with_j = cv2.filter2D(
                            map_retina_opt,
                            -1,
                            self.attention_kernel[i][0],
                        )
                    else:
                        attention_response_with_j = np.minimum(
                            attention_response_with_j,
                            cv2.filter2D(
                                map_retina_opt,
                                -1,
                                self.attention_kernel[i][j],
                            )
                        )
                
                if i == 0:
                    attention_response = attention_response_with_j
                else:
                    attention_response = np.maximum(
                        attention_response,
                        attention_response_with_j
                    )
            
            attention_opt = retina_opt + self.alpha * attention_response
        
        self.Opt = attention_opt
        return attention_opt


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
        self.prediction_kernel = None
        self.cell_prediction_gain = None
        self.cell_prediction_map = None
        self.time_attenuation_kernel = None
    
    def init_config(self):
        """
        initiate config for prediction module.
        """        
        if self.intDeltaT < 0:
            self.intDeltaT = 0
        elif not isinstance(self.intDeltaT, int):
            self.intDeltaT = round(self.intDeltaT)
        
        if self.velocity is None:
            self.velocity = 25 / 4 / self.intDeltaT
        
        self.prediction_kernel = create_prediction_kernel(
            self.velocity,
            self.intDeltaT,
            self.sizeFilter,
            self.numFilter,
            self.zeta,
            self.eta
        )
        self.cell_prediction_gain = [[None]*self.numFilter for _ in range(self.intDeltaT + 1)]
        self.cell_prediction_map = [None]*(self.intDeltaT + 1)
        
        self.time_attenuation_kernel = np.exp(self.kappa * np.arange(-self.intDeltaT, 1))
    
    def process(self, lobula_opt):
        """
        Processing method.
        
        Processes the input lobula_opt to predict motion and update
        prediction map.
        """
        num_dict = len(lobula_opt)
        img_h, img_w = lobula_opt[0].shape
        
        # Prediction Gain
        self.cell_prediction_gain = np.roll(self.cell_prediction_gain, shift=-1, axis=0)
        prediction_gain = []
        for idxD in range(num_dict):
            if self.cell_prediction_gain[0][idxD] is None:
                prediction_gain.append(cv2.filter2D(
                    self.mu * lobula_opt[idxD],
                    -1,
                    self.prediction_kernel[idxD],
                ))
            else:
                prediction_gain.append(cv2.filter2D(
                    self.mu * lobula_opt[idxD] + (1 - self.mu) * self.cell_prediction_gain[0][idxD],
                    -1,
                    self.prediction_kernel[idxD],
                ))
        self.cell_prediction_gain[-1] = prediction_gain
        
        # Prediction Map
        tobe_prediction_map = np.zeros((img_h, img_w))
        for idxD in range(num_dict):
            tobe_prediction_map += prediction_gain[idxD]
        
        # Facilitated STMD Output
        facilitated_opt = [np.copy(lobula_opt[idxD]) for idxD in range(num_dict)]
        for idxD in range(num_dict):
            if self.cell_prediction_gain[-1][idxD] is not None:
                facilitated_opt[idxD] += self.beta * compute_temporal_conv(
                    self.cell_prediction_gain[:, idxD],
                    self.time_attenuation_kernel
                )
        
        # Memorizer update
        max_tobe_pre_map = np.max(tobe_prediction_map)
        self.cell_prediction_map = np.roll(self.cell_prediction_map, shift=-1)
        self.cell_prediction_map[-1] = (tobe_prediction_map > max_tobe_pre_map * 2e-1)
        
        # Output
        prediction_map = self.cell_prediction_map[0]
        self.Opt = facilitated_opt
        return facilitated_opt, prediction_map
