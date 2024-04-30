from numpy import maximum
from scipy.ndimage import convolve

from .basecore import BaseCore
from ..util.create_kernel import create_inhi_kernel_W2


class SurroundInhibition(BaseCore):
    """
    Gamma_Filter Gamma filter in lamina layer
    """

    def __init__(self, KernelSize=15, Sigma1=1.5, Sigma2=3, e=1.0, rho=0, A=1, B=3):
        """
        Constructor
        Initializes the SurroundInhibition object with optional parameters
        
        Parameters:
        - KernelSize: Size of the filter kernel
        - Sigma1: Standard deviation for the first Gaussian
        - Sigma2: Standard deviation for the second Gaussian
        - e: Exponent for the weighting of the second Gaussian
        - rho: Radius for circular integration
        - A: Amplitude of the filter
        - B: Offset of the filter
        """
        super().__init__()
        self.KernelSize = KernelSize
        self.Sigma1 = Sigma1
        self.Sigma2 = Sigma2
        self.e = e
        self.rho = rho
        self.A = A
        self.B = B
        self.inhiKernelW2 = None

    def init_config(self):
        """
        Initialization method
        Creates the surround inhibition filter kernel
        """
        self.inhiKernelW2 = create_inhi_kernel_W2(
            self.KernelSize,
            self.Sigma1,
            self.Sigma2,
            self.e,
            self.rho,
            self.A,
            self.B
        )

    def process(self, iptMatrix):
        """
        Processing method
        Applies the surround inhibition filter to the input matrix
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - inhiOpt: Output of the surround inhibition filter
        """
        inhiOpt = convolve(iptMatrix, self.inhiKernelW2, mode='constant')
        inhiOpt = maximum(inhiOpt, 0)
        return inhiOpt
