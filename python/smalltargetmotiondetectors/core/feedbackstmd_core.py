from . import BaseCore
from .surround_inhibition import SurroundInhibition
from .math_operator import GammaDelay
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import numpy as np

class Lobula(BaseCore):
    """Lobula layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.hSubInhi = SurroundInhibition()  # SurroundInhibition component
        self.alpha = 1  # Parameter alpha
        self.paraGaussKernel = {'eta': 1.5, 'size': 3}  # Parameters for Gaussian kernel
        self.gammaKernel = None  # Gamma kernel
        self.gaussKernel = None  # Gaussian kernel
        self.hGammaDelay = GammaDelay(10, 25)

    def init_config(self):
        """Initialization method."""
        # Initializes the Lobula layer component
        self.hSubInhi.init_config()
        self.hGammaDelay.init_config()
        self.gaussKernel = gaussian_filter(
            np.zeros((self.paraGaussKernel['size'], self.paraGaussKernel['size'])),
            self.paraGaussKernel['eta']
        )

    def process(self, onSignal, offSignal):
        """Processing method."""
        # Performs temporal convolution, correlation, and surround inhibition

        # Formula (9)
        feedbackSignal = self.alpha * self.hGammaDelay.process(np.zeros_like(onSignal))

        # Formula (8)
        correlationD = np.maximum((onSignal - feedbackSignal), 0) * np.maximum((offSignal - feedbackSignal), 0)

        # Formula (10)
        correlationE = convolve2d(onSignal * offSignal, self.gaussKernel, mode='same')

        # Only record correlationD + correlationE
        self.hGammaDelay.hCellInput.cover(correlationD + correlationE)

        # Formula (14)
        lobulaOpt = self.hSubInhi.process(correlationD)

        # Store the output in Opt property
        self.Opt = lobulaOpt

        return lobulaOpt
