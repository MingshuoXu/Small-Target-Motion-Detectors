import numpy as np
from cv2 import filter2D, BORDER_CONSTANT
import torch
import torch.nn.functional as F

from .base_core import BaseCore
from .math_operator import SurroundInhibition, GammaDelay
from ..util.create_kernel import create_gaussian_kernel


class Lobula(BaseCore):
    """ Lobula layer of the motion detection system."""
    
    def __init__(self, device='cpu'):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__(device=device)
        self.hSubInhi = SurroundInhibition(device=device)  # SurroundInhibition component
        self.alpha = 1  # Parameter alpha
        self.paraGaussKernel = {'eta': 1.5, 'size': 3}  # Parameters for Gaussian kernel
        self.gaussKernel = None  # Gaussian kernel
        self.hGammaDelay = GammaDelay(10, 25)  # GammaDelay component

    def init_config(self):
        """ Initialization method."""
        # Initializes the Lobula layer component
        self.hSubInhi.init_config()
        self.hGammaDelay.init_config()
        self.gaussKernel = create_gaussian_kernel(self.paraGaussKernel['size'], 
                                                  self.paraGaussKernel['eta'])
        if self.device != 'cpu':
            self.gaussKernel = torch.from_numpy(self.gaussKernel).float().to(self.device).unsqueeze(0).unsqueeze(0)

    def process(self, onSignal, offSignal):
        """ Processing method. """
        # Performs temporal convolution, correlation, and surround inhibition

        # Formula (9)
        _temp = np.zeros_like(onSignal) if self.device == 'cpu' else torch.zeros_like(onSignal)
        feedbackSignal = self.alpha * self.hGammaDelay.process(_temp)

        if self.device == 'cpu':
            # Formula (8)
            self.v_on = np.maximum(onSignal - feedbackSignal, 0)
            self.v_off = np.maximum(offSignal - feedbackSignal, 0)
            correlationD = self.v_on * self.v_off

            # Formula (10)
            correlationE = filter2D(onSignal * offSignal, -1, self.gaussKernel, borderType=BORDER_CONSTANT)
        else:
            # Formula (8)
            self.v_on = torch.clamp(onSignal - feedbackSignal, min=0) 
            self.v_off = torch.clamp(offSignal - feedbackSignal, min=0)
            correlationD = self.v_on * self.v_off

            # Formula (10)
            correlationE = F.conv2d(onSignal * offSignal, 
                                    self.gaussKernel, 
                                    padding='same')

        # Only record (correlationD + correlationE) for next delay in Formula (9)
        self.hGammaDelay.listInput.cover(correlationD + correlationE)

        # Formula (14)
        lobulaOpt = self.hSubInhi.process(correlationD)

        # Store the output in Opt property
        self.Opt = lobulaOpt

        return lobulaOpt
