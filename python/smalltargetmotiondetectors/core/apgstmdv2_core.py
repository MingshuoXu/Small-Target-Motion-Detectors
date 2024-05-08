import numpy as np
import cv2

from .base_core import BaseCore
from ..util.create_kernel import create_prediction_kernel


class PredictionModule(BaseCore):
    """PredictionModule class for ApgSTMD."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the PredictionModule object
        super().__init__()
        
        # Parameters
        self.velocity = 0.1  # Velocity: v_{opt} (Default: 0.25)
        self.intDeltaT = 0  # Delta time
        self.sizeFilter = 20  # Size of filter
        self.numFilter = 8  # Number of filters
        self.zeta = 2  # Zeta parameter
        self.eta = 2.5  # Eta parameter
        self.beta = 1  # Beta parameter

        # Hidden properties
        self.predictionKernel = None  # Prediction kernel

    def init_config(self):
        """Initialization method."""
        # Initializes the prediction module

        self.predictionKernel = create_prediction_kernel(
            self.velocity,
            self.intDeltaT,
            self.sizeFilter,
            self.numFilter,
            self.zeta,
            self.eta
        )

    def process(self, lobulaOpt):
        """Processing method."""
        # Processes the input lobulaOpt to predict motion and update prediction map

        numDict = len(lobulaOpt)
        imgH, imgW = lobulaOpt[0].shape

        predictionGain = []
        for idxD in range(numDict):
            predictionGain.append(
                cv2.filter2D(
                    lobulaOpt[idxD],
                    -1,
                    self.predictionKernel[idxD],
                )
            )

        # Prediction Map
        predictionMap = np.zeros((imgH, imgW))
        for idxD in range(numDict):
            predictionMap += predictionGain[idxD]

        # Facilitated STMD Output
        facilitatedOpt = []
        for idxD in range(numDict):
            facilitatedOpt.append(
                lobulaOpt[idxD] + self.beta * predictionGain[idxD]
            )

        # Memorizer update
        maxPreMap = np.max(predictionMap)
            # Logical Matrix
        predictionMap = (predictionMap > maxPreMap * 2e-1)

        # Output
        self.Opt = facilitatedOpt
        
        return facilitatedOpt, predictionMap
