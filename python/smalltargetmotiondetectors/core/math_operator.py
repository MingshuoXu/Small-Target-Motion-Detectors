import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from .base_core import BaseCore
from ..util.compute_module import compute_temporal_conv, compute_circularlist_conv
from ..util.create_kernel import create_gamma_kernel, create_inhi_kernel_W2
from ..util.datarecord import CircularList


class GaussianBlur(BaseCore):
    """
    Gaussian blur filter.
    This class implements a Gaussian blur filter for image processing.
    """

    def __init__(self):
        """
        Constructor.
        Initializes the GaussianBlur object.
        """
        super().__init__()
        self.size = 3   # Size of the filter kernel
        self.sigma = 1  # Standard deviation of the Gaussian distribution
        self.gaussKernel = None  # Gaussian filter kernel

    def init_config(self):
        """
        Initialization method.
        Creates the Gaussian filter kernel using scipy.ndimage.gaussian_filter.
        """
        # self.gaussKernel = gaussian_filter(size=self.size, sigma=self.sigma)
        pass

    def process(self, iptMatrix):
        """
        Processing method.
        Applies the Gaussian filter to the input matrix.

        Parameters:
        - iptMatrix: Input matrix.

        Returns:
        - optMatrix: Output matrix after applying the Gaussian filter.
        """
        optMatrix = gaussian_filter(iptMatrix, sigma=self.sigma)
        return optMatrix
    

class GammaDelay(BaseCore):
    """
    GammaDelay Class
    
    Implements a gamma filter used in the lamina layer of the ESTMD (Elementary Motion Detection) neural network.
    This filter is based on insect physiology and serves to detect moving targets in visual clutter.
    
    Parameters:
        order (int): Order of the gamma filter. Default is 1.
        tau (float): Time constant of the filter.
        lenKernel (int): Length of the filter kernel. If not provided, it is calculated based on the time constant.
        isRecord (bool): Flag indicating whether to record input history. Default is True.
        isInLoop (bool): Flag indicating whether to cover the point in CircularCell. Default is False.
    
    Methods:
        __init__(order=1, tau=1, len_kernel=None): Constructor method. Initializes a GammaDelay object with the specified parameters.
        init_config(is_record=True): Initialization method. Creates the gamma filter kernel and initializes input history.
        process(inputData): Processing method. Applies the gamma filter to the input data.
        process_matrix(inputMatrix): Process method for matrix input. Applies the gamma filter to the input matrix.
        process_cell(objListIpt): Process method for cell input. Applies the gamma filter to the input cell array.
        process_circularcell(objCircularList): Process method for circular cell input. Applies the gamma filter to the circular cell object.
    """
    def __init__(self, order=1, tau=1, lenKernel=None):
        """
        Constructor method.
        
        Initializes a GammaDelay object with the specified parameters.
        
        Parameters:
            order (int): Order of the gamma filter. Default is 1.
            tau (float): Time constant of the filter.
            len_kernel (int): Length of the filter kernel. If not provided, it is calculated based on the time constant.
        """
        self.order = order
        self.tau = tau
        self.lenKernel = lenKernel
        self.isRecord = True
        self.isInLoop = False
        self.gammaKernel = None
        self.listInput = None

    def init_config(self, isRecord=True):
        self.isRecord = isRecord

        if self.order < 1:
            self.order = 1

        if self.lenKernel is None:
            self.lenKernel = int(np.ceil(3 * self.tau))

        if self.gammaKernel is None:
            self.gammaKernel = create_gamma_kernel(self.order,
                                                   self.tau,
                                                   self.lenKernel)

        if self.isRecord:
            self.listInput = CircularList(self.lenKernel)

    def process(self, inputData):
        if isinstance(inputData, CircularList):
            return self.process_circularlist(inputData)
        elif isinstance(inputData, list):
            return self.process_list(inputData)
        elif isinstance(inputData, np.ndarray):
            return self.process_matrix(inputData)

    def process_matrix(self, inputMatrix):
        if self.isInLoop:
            self.listInput.cover(inputMatrix)
        else:
            self.listInput.circrecord(inputMatrix)

        return self.process_circularlist(self.listInput)

    def process_list(self, objListIpt):
        return compute_temporal_conv(objListIpt, 
                                     self.gammaKernel)

    def process_circularlist(self, objCircularList):
        return compute_circularlist_conv(objCircularList, 
                                         self.gammaKernel)


class GammaBandPassFilter(BaseCore):
    """
    GammaBankPassFilter Gamma bank pass filter
    This class implements a gamma bank pass filter with two gamma delays.
    
    Author: [Your Name]
    Date: [Date]
    """

    def __init__(self):
        """
        Constructor method
        Initializes the GammaBankPassFilter object
        """
        super().__init__()
        
        # Initialize gamma delays with specified parameters
        self.hGammaDelay1 = GammaDelay(2, 3)
        self.hGammaDelay2 = GammaDelay(3, 6)
        self.objListIpt = CircularList()

    def init_config(self):
        """
        Initialization method
        Initializes gamma delays
        """
        # Ensure that the second gamma delay's time constant is greater than the first one
        if self.hGammaDelay1.tau < self.hGammaDelay2.tau:
            self.hGammaDelay2.tau = self.hGammaDelay1.tau + 1

        # Initialize gamma delay 1
        self.hGammaDelay1.init_config(False)
        # Initialize gamma delay 2
        self.hGammaDelay2.init_config(False)

        # Determine the length of the circular cell input
        if self.objListIpt.len is None:
            self.objListIpt.len = max(self.hGammaDelay1.lenKernel, 
                                      self.hGammaDelay2.lenKernel)
        self.objListIpt.init_config()

    def process(self, iptMatrix):
        """
        Processing method
        Processes the input matrix using gamma delays
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - optMatrix: Processed output matrix
        """
        # Record input matrix in circular cell
        self.objListIpt.circrecord(iptMatrix)

        # Compute outputs of gamma delays
        gamma1Output = self.hGammaDelay1.process_circularlist(self.objListIpt)
        gamma2Output = self.hGammaDelay2.process_circularlist(self.objListIpt)
        
        # Compute the difference between the outputs
        
        optMatrix = gamma1Output - gamma2Output
        return optMatrix


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
        inhiOpt = cv2.filter2D(iptMatrix, -1, self.inhiKernelW2)
        inhiOpt = np.maximum(inhiOpt, 0)
        return inhiOpt


if __name__ == "__main__":
    pass