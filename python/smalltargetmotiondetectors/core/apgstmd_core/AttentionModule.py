import numpy as np
from smalltargetmotiondetectors.core.base_core import BaseCore
from smalltargetmotiondetectors.tool.kernel import create_attention_kernel

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
    
    def init(self):
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
        r, s = self.attention_kernel.shape
        
        if prediction_map is None:
            attention_opt = retina_opt
        else:
            map_retina_opt = retina_opt * prediction_map
            attention_response = None
            
            for i in range(r):
                for j in range(s):
                    if j == 0:
                        attention_response_with_j = np.convolve(
                            map_retina_opt,
                            self.attention_kernel[i, 0],
                            mode='same'
                        )
                    else:
                        attention_response_with_j = np.minimum(
                            attention_response_with_j,
                            np.convolve(
                                map_retina_opt,
                                self.attention_kernel[i, j],
                                mode='same'
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
