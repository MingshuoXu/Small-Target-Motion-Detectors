import torch
import torch.nn.functional as F

from .base_core import BaseCore
from .math_operator import SpatialInhibition, GammaDelay
from ..util.create_kernel import create_2d_gaussian_kernel


class Lobula(BaseCore):
    """ Lobula layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.spatial_inhibition = SpatialInhibition()  # SpatialInhibition component
        self.alpha = 1  # Parameter alpha
        self.sigma = 1.5  # Parameters for Gaussian kernel

        self.gamma_delay = GammaDelay(10, 25)  # GammaDelay component

        self.register_buffer('gaussian_kernel', torch.empty(0))  # Buffer for Gaussian kernel

        self.setup()

    def setup(self):
        """ Initialization method."""
        # Initializes the Lobula layer component
        self.spatial_inhibition.setup()
        self.gamma_delay.setup()

        self.gaussian_kernel.data = create_2d_gaussian_kernel(size=3, sigma=self.sigma)

        self.reset_buffer()

    def reset_buffer(self):
        """ Resets the buffer of certain components. """
        self.gamma_delay.reset_buffer()

    def forward(self, medulla_ON, medulla_OFF):
        """ Processing method. """
        # Performs temporal convolution, correlation, and surround inhibition

        # Formula (9)
        feedback_output = self.alpha * self.gamma_delay.forward(torch.zeros_like(medulla_ON))

        # Formula (8)
        ON_with_feedback = torch.clamp(medulla_ON - feedback_output, min=0) 
        OFF_with_feedback = torch.clamp(medulla_OFF - feedback_output, min=0)
        correlation_D = ON_with_feedback * OFF_with_feedback

        # Formula (10)
        correlation_E = F.conv2d(medulla_ON * medulla_OFF, self.gaussian_kernel, padding='same')

        # Only record (correlationD + correlationE) for next delay in Formula (9)
        self.gamma_delay.buffer[-1] = correlation_D + correlation_E

        # Formula (14)
        self.output = self.spatial_inhibition(correlation_D)

        return self.output
