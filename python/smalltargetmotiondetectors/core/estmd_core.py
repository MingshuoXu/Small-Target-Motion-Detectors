import numpy as np
from cv2 import filter2D

from .base_core import BaseCore
from ..util.compute_module import compute_temporal_conv, compute_circularlist_conv
from ..util.create_kernel import create_gaussian_kernel
from .math_operator import *


class Retina(BaseCore):
    """
    Retina filter.
    This class implements Retina Layer.

    Author: Mingshuo Xu
    Date: 2024-04-23
    """

    def __init__(self):
        """
        Constructor.
        Initializes the Retina object and creates a GaussianBlur object.
        """
        super().__init__()
        self.hGaussianBlur = GaussianBlur()

    def init_config(self):
        """
        Initialization method.
        Initializes the GaussianBlur object.
        """
        self.hGaussianBlur.init_config()

    def process(self, retinaIpt):
        """
        Processing method.
        Applies the Gaussian blur filter to the input matrix.

        Parameters:
        - retinaIpt: Input matrix.

        Returns:
        - retinaOpt: Output matrix after applying the Gaussian blur filter.
        """
        retinaOpt = self.hGaussianBlur.process(retinaIpt)
        self.Opt = retinaOpt
        return retinaOpt


class Lamina(BaseCore):
    """
    LAMINA Lamina layer
    This class implements the Lamina layer of the ESTMD
    
    Author: Mingshuo Xu
    Date: 2024-04-29
    """
    
    def __init__(self):
        """
        Constructor
        Initializes the Lamina object and creates GammaBankPassFilter
        and LaminaLateralInhibition objects
        """
        self.hGammaBankPassFilter = GammaBandPassFilter()
        self.hLaminaLateralInhibition = LaminaLateralInhibition()

    def init_config(self):
        """
        Initialization method
        Initializes the GammaBankPassFilter and LaminaLateralInhibition objects
        """
        self.hGammaBankPassFilter.init_config()
        self.hLaminaLateralInhibition.init_config()

    def process(self, laminaIpt):
        """
        Processing method
        Applies GammaBankPassFilter and LaminaLateralInhibition to the input matrix
        
        Parameters:
        - laminaIpt: Input matrix
        
        Returns:
        - laminaOpt: Processed output matrix
        """
        signalWithBPF = self.hGammaBankPassFilter.process(laminaIpt)
        laminaOpt = self.hLaminaLateralInhibition.process(signalWithBPF)
        self.Opt = laminaOpt
        return laminaOpt


class Medulla(BaseCore):
    """
    Medulla Layer of DSTMD
    This class implements the Medulla layer of the ESTMD.
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Medulla object
        """

        self.hTm1 = Tm1()  # Initialize Tm1 object
        self.hTm2 = Tm2()  # Initialize Tm2 object
        self.hTm3 = Tm3()  # Initialize Tm3 object

    def init_config(self):
        """
        Initialization method
        Initializes the Tm1, Tm2, and Tm3 objects
        """
        self.hTm1.init_config()
        self.hTm2.init_config()
        self.hTm3.init_config()

    def process(self, MedullaIpt):
        """
        Processing method
        Processes the input MedullaIpt through Tm1, Tm2, and Tm3 layers
        
        Parameters:
        - MedullaIpt: Input matrix
        
        Returns:
        - tm3Signal: Output of Tm3 layer
        - tm1Signal: Output of Tm1 layer
        """
        tm2Signal = self.hTm2.process(MedullaIpt)  # Process input through Tm2
        tm3Signal = self.hTm3.process(MedullaIpt)  # Process input through Tm3

        tm1Signal = self.hTm1.process(tm2Signal)  # Process Tm2 output through Tm1

        self.Opt = (tm3Signal, tm1Signal)  # Update Opt property with output
        return tm3Signal, tm1Signal


class Lobula(BaseCore):
    """
    Lobula Layer of DSTMD
    This class implements the Lobula layer of the ESTMD.
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Lobula object
        """
        super().__init__()
        self.a = 0  # Parameter a
        self.b = 0  # Parameter b
        self.c = 1  # Parameter c

    def init_config(self):
        """
        Initialization method
        """
        pass

    def process(self, varagein):
        """
        Processing method
        Processes the input ON and OFF signals
        
        Parameters:
        - varagein: Tuple containing ON and OFF signals
        
        Returns:
        - lobulaOpt: Output of the Lobula layer
        """
        onSignal = varagein[0]  # Extract ON signal
        offSignal = varagein[1]  # Extract OFF signal
        
        # Compute Lobula output using the provided formula
        lobulaOpt = self.a*onSignal + self.b*offSignal + self.c*onSignal*offSignal
        
        self.Opt = lobulaOpt  # Update Opt property with output
        return lobulaOpt


class Mi1(BaseCore):
    """
    MI1 
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Mi1 object
        """
        super().__init__()
        self.hGammaDelay = GammaDelay(12, 25)  # Initialize GammaDelay object

    def init_config(self):
        """
        Initialization method
        Initializes the GammaDelay object
        """
        self.hGammaDelay.init_config()

    def process(self, tm3Signal):
        """
        Processing method
        Apply gamma delay to the input signal
        
        Parameters:
        - tm3Signal: Input signal
        
        Returns:
        - mi1Opt: Output of the Mi1 layer
        """
        mi1Opt = self.hGammaDelay.process(tm3Signal)
        self.Opt = mi1Opt
        return mi1Opt


class Tm1(BaseCore):
    """
    Tm1 
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Tm1 object
        """
        self.hGammaDelay = GammaDelay(12, 25)  # Initialize GammaDelay object

    def init_config(self):
        """
        Initialization method
        Initializes the GammaDelay object
        """
        self.hGammaDelay.init_config()

    def process(self, tm2Signal):
        """
        Processing method
        Apply gamma delay to the input signal
        
        Parameters:
        - tm2Signal: Input signal
        
        Returns:
        - tm1Opt: Output of the Tm1 layer
        """
        tm1Opt = self.hGammaDelay.process(tm2Signal)
        self.Opt = tm1Opt
        return tm1Opt


class Tm2(BaseCore):
    """
    Tm2 Medulla Layer Neurons in ESTMD
    """
   
    def __init__(self):
        """
        Constructor method
        Initializes the Tm2 object
        """
        self.hSubInhi = SurroundInhibition()  # Initialize SurroundInhibition object

    def init_config(self):
        """
        Initialization method
        Initializes the SurroundInhibition object
        """
        self.hSubInhi.init_config()

    def process(self, iptMatrix):
        """
        Processing method
        Applies the Surround Inhibition mechanism to the input matrix iptMatrix
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - tm2Opt: Output of the Tm2 layer
        """
        # Extract the OFF signal from iptMatrix
        offSignal = np.maximum(-iptMatrix, 0)  
        # Process the OFF signal using SurroundInhibition
        tm2Opt = self.hSubInhi.process(offSignal)  
        self.Opt = tm2Opt
        return tm2Opt


class Tm3(BaseCore):
    """
    Tm3 
    """
    
    def __init__(self):
        """
        Constructor method
        Initializes the Tm3 object
        """
        self.hSubInhi = SurroundInhibition()  # Initialize SurroundInhibition object

    def init_config(self):
        """
        Initialization method
        Initializes the SurroundInhibition object
        """
        self.hSubInhi.init_config()

    def process(self, iptMatrix):
        """
        Processing method
        Applies Surround Inhibition to the On-signal matrix iptMatrix
        
        Parameters:
        - iptMatrix: Input matrix
        
        Returns:
        - tm3Opt: Output of the Tm3 layer
        """
        onSignal = np.maximum(iptMatrix, 0)  # Extract the On-signal from iptMatrix
        tm3Opt = self.hSubInhi.process(onSignal)  # Processes the On-signal using SurroundInhibition
        self.Opt = tm3Opt
        return tm3Opt


class LaminaLateralInhibition(BaseCore):
    """
    LAMINALATERALINHIBITION Lateral inhibition in the Lamina layer
    This class implements the lateral inhibition mechanism in the Lamina layer
    of the ESTMD.

    References:
    * S. D. Wiederman, P. A. Shoemarker, D. C. O'Carroll, A model
    for the detection of moving targets in visual clutter inspired by
    insect physiology, PLoS ONE 3 (7) (2008) e2784.
    * Wang H, Peng J, Yue S. A directionally selective small target
    motion detecting visual neural network in cluttered backgrounds[J].
    IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.

    Author: Mingshuo Xu
    Date: 2022-04-26
    LastEditTime: 2024-04-26
    """

    def __init__(self, 
                 sizeW1=[11, 11, 7], 
                 lambda1=3, 
                 lambda2=9, 
                 sigma2=1.5, 
                 sigma3=None):
        """
        Constructor
        Initializes the LaminaLateralInhibition object
        """
        self.sizeW1 = sizeW1
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sigma2 = sigma2
        self.sigma3 = sigma3

        # Positive part of the inhibition kernel W1
        self.spatialPositiveKernel = []
        # Negative part of the inhibition kernel W1
        self.spatialNegativeKernel = []
        # Temporal component for the positive part of W1
        self.temporalPositiveKernel = []
        # Temporal component for the negative part of W1
        self.temporalNegativeKernel = []
        # Cell array to store intermediate results for the positive part
        self.cellSpatialPositive = []
        # Cell array to store intermediate results for the negative part
        self.cellSpatialNegative = []

    def init_config(self):
        """
        Initialization method
        Initializes the inhibition kernel W1
        """
        if self.sigma3 is None:
            self.sigma3 = 2 * self.sigma2

        G_sigma2 = create_gaussian_kernel(self.sizeW1[:2], self.sigma2)
        G_sigma3 = create_gaussian_kernel(self.sizeW1[:2], self.sigma3)
        diffOfGaussian = G_sigma2 - G_sigma3
        # W_{S}^{P} in formulate (8) of DSTMD
        self.spatialPositiveKernel = np.maximum(diffOfGaussian, 0) 
        # W_{S}^{N} in formulate (9) of DSTMD
        self.spatialNegativeKernel = np.maximum(-diffOfGaussian, 0) 

        t = np.arange(self.sizeW1[2])
        # W_{T}^{P} in formulate (10) of DSTMD
        self.temporalPositiveKernel = np.exp(-t / self.lambda1) / self.lambda1
        # W_{T}^{N} in formulate (11) of DSTMD
        self.temporalNegativeKernel = np.exp(-t / self.lambda2) / self.lambda2

        self.cellSpatialPositive = CircularList(self.sizeW1[2])
        self.cellSpatialNagetive = CircularList(self.sizeW1[2])

    def process(self, iptMatrix):
        """
        Processing method
        Applies lateral inhibition to the input matrix
        """
        # Lateral inhibition
        self.cellSpatialPositive.circrecord(
            filter2D(iptMatrix, -1, self.spatialPositiveKernel)
            )
        self.cellSpatialPositive.circrecord(
            filter2D(iptMatrix, -1, self.spatialNegativeKernel)
            )

        optMatrix \
            = compute_circularlist_conv(
                self.cellSpatialPositive, 
                self.temporalPositiveKernel) \
            + compute_circularlist_conv(
                self.cellSpatialPositive, 
                self.temporalNegativeKernel)

        if optMatrix.size == 0:
            optMatrix = np.zeros_like(iptMatrix)

        return optMatrix

    





