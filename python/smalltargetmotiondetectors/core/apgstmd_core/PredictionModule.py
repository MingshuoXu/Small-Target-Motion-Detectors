import numpy as np
from smalltargetmotiondetectors.core.base_core import BaseCore
from smalltargetmotiondetectors.util.kernel import create_prediction_kernel
from smalltargetmotiondetectors.util.compute import compute_temporal_conv

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
        from smalltargetmotiondetectors.util.kernel import create_prediction_kernel
        
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
                prediction_gain.append(np.convolve(
                    self.mu * lobula_opt[idxD],
                    self.prediction_kernel[idxD],
                    mode='same'
                ))
            else:
                prediction_gain.append(np.convolve(
                    self.mu * lobula_opt[idxD] + (1 - self.mu) * self.cell_prediction_gain[0][idxD],
                    self.prediction_kernel[idxD],
                    mode='same'
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
