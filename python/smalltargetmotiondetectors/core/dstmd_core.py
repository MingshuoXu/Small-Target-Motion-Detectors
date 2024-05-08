import numpy as np
import math

from .base_core import BaseCore
from ..util.datarecord import CircularList
from .math_operator import GammaDelay
from .math_operator import SurroundInhibition
from ..util.create_kernel import create_direction_inhi_kernel


class Medulla(BaseCore):
    """Medulla class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Medulla object
        super().__init__()

        # Initialize components
        self.hTm3 = Tm3()
        self.hMi1Para4 = Mi1()

        self.hTm2 = Tm2()
        self.hTm1Para5 = Tm1()
        self.hTm1Para6 = Tm1()
        
        # Set parameters for hDelay6Tm1
        self.hTm1Para6.hGammaDelay.order = 8
        self.hTm1Para6.hGammaDelay.tau = 40

        self.cellTm1Ipt = CircularList()

    def init_config(self):
        """Initialization method."""
        # Initializes the delay components
        
        self.hTm1Para5.hGammaDelay.isRecord = False
        self.hTm1Para6.hGammaDelay.isRecord = False

        self.hMi1Para4.init_config()
        self.hTm1Para5.init_config()
        self.hTm1Para6.init_config()

        if not self.cellTm1Ipt.len:
            self.cellTm1Ipt.len = max(
                self.hTm1Para5.hGammaDelay.lenKernel,
                self.hTm1Para6.hGammaDelay.lenKernel
            )
        self.cellTm1Ipt.init_config()

    def process(self, MedullaInput):
        """Processing method."""
        # Processes input signals and produces output
        
        # Process Tm3 and Tm2 signals
        tm3Signal = self.hTm3.process(MedullaInput)
        tm2Signal = self.hTm2.process(MedullaInput)
        
        # Process signals with delays
        mi1Para4Signal = self.hMi1Para4.process(tm3Signal)
        
        self.cellTm1Ipt.circrecord(tm2Signal)
        tm1Para5Signal = self.hTm1Para5.process(self.cellTm1Ipt)
        tm1Para6Signal = self.hTm1Para6.process(self.cellTm1Ipt)

        # Output signals
        self.Opt = [tm3Signal, mi1Para4Signal, tm1Para5Signal, tm1Para6Signal]
        return self.Opt


class Mi1(BaseCore):
    """Mi1 class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Mi1 object
        super().__init__()
        
        # Initialize gamma delay component with default parameters
        self.hGammaDelay = GammaDelay(3, 15)

    def init_config(self):
        """Initialization method."""
        # Initializes the gamma delay component
        self.hGammaDelay.init_config()

    def process(self, tm3Signal):
        """Processing method."""
        # Processes the input signal using the gamma delay component
        
        # Apply gamma delay to the input signal
        mi1Opt = self.hGammaDelay.process(tm3Signal)

        # Set output
        self.Opt = mi1Opt
        return self.Opt


class Tm1(BaseCore):
    """Tm1 class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Tm1 object
        super().__init__()
        
        # Initialize gamma delay component with default parameters
        self.hGammaDelay = GammaDelay(5, 25)

    def init_config(self):
        """Initialization method."""
        # Initializes the gamma delay component
        self.hGammaDelay.init_config()

    def process(self, tm2Signal):
        """Processing method."""
        # Processes the input signal using the gamma delay component
        
        # Apply gamma delay to the input signal
        tm1Opt = self.hGammaDelay.process(tm2Signal)

        # Set output
        self.Opt = tm1Opt
        return self.Opt


class Tm2(BaseCore):
    """Tm2 class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Tm2 object
        super().__init__()

    def init_config(self):
        """Initialization method."""
        # This method is not used in this class
        pass

    def process(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values
        
        tm2Opt = np.maximum(-iptMatrix, 0)
        self.Opt = tm2Opt  # Update output
        return self.Opt


class Tm3(BaseCore):
    """Tm3 class for motion detection."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Tm3 object
        super().__init__()

    def init_config(self):
        """Initialization method."""
        # This method is not used in this class
        pass

    def process(self, iptMatrix):
        """Processing method."""
        # Processes the input matrix by performing a maximum operation with zero for negative values
        
        tm3Opt = np.maximum(iptMatrix, 0)
        self.Opt = tm3Opt  # Update output
        return self.Opt


class Lobula(BaseCore):
    """Lobula class for motion detection."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.alpha1 = 3  # Alpha parameter
        self.thetaList = [(i * math.pi / 4) for i in range(8)]  # List of theta values
        self.hLateralInhi = SurroundInhibition()  # Lateral inhibition component
        self.hDirectionInhi = DirectionInhi()  # Directional inhibition component

    def init_config(self):
        """Initialization method."""
        # Initializes the lateral and directional inhibition components
        self.hLateralInhi.init_config()
        self.hDirectionInhi.init_config()

    def process(self, lobulaIpt):
        """Processing method."""
        # Performs motion processing on the input

        tm3Signal, mi1Para4Signal, tm1Para5Signal, tm1Para6Signal = lobulaIpt
        imgH, imgW = tm3Signal.shape
        numDict = len(self.thetaList)

        # Correlation range
        xRange = slice(self.alpha1, imgH - self.alpha1)
        yRange = slice(self.alpha1, imgW - self.alpha1)

        # Correlation Output
        correOutput = [np.zeros((imgH, imgW)) for _ in range(numDict)]

        for countTheta, theta in enumerate(self.thetaList):
            # Correlation position
            shiftX = round(self.alpha1 * math.cos(theta + math.pi / 2))
            shiftY = round(self.alpha1 * math.sin(theta + math.pi / 2))
            shiftXRange = slice(self.alpha1 - shiftX, imgH - self.alpha1 - shiftX)
            shiftYRange = slice(self.alpha1 - shiftY, imgW - self.alpha1 - shiftY)

            # Calculate correlation output
            correOutput[countTheta][xRange, yRange] = (
                tm3Signal[xRange, yRange]
                * (tm1Para5Signal[xRange, yRange] + mi1Para4Signal[shiftXRange, shiftYRange])
                * tm1Para6Signal[shiftXRange, shiftYRange]
            )


        # Perform lateral inhibition
        lateralInhiOpt = [self.hLateralInhi.process(output) for output in correOutput]


        # Perform directional inhibition
        lobulaOpt = self.hDirectionInhi.process(lateralInhiOpt)

        self.Opt = lobulaOpt
        return lobulaOpt


class DirectionInhi(BaseCore):
    """Directional inhibition in DSTMD."""

    def __init__(self):
        """Constructor method."""
        # Initializes the DirectionInhi object
        super().__init__()
        self.direction = 8  # Number of directions
        self.sigma1 = 1.5  # Sigma for the first Gaussian kernel
        self.sigma2 = 3.0  # Sigma for the second Gaussian kernel
        self.diretionalInhiKernel = None  # Directional inhibition kernel

    def init_config(self):
        """Initialization method."""
        # Initializes the directional inhibition kernel
        if self.diretionalInhiKernel is None:
            self.diretionalInhiKernel = create_direction_inhi_kernel(
                self.direction, self.sigma1, self.sigma2
            )
        self.diretionalInhiKernel = self.diretionalInhiKernel.squeeze()

    def process(self, iptCell):
        """Processing method."""
        # Performs directional inhibition on the input
        
        len1 = len(iptCell)
        len2 = len(self.diretionalInhiKernel)
        certer = len2 // 2
        opt = []

        for idx in range(len1):
            result = np.zeros_like(iptCell[0])
            matrixPoint = idx
            kernelPoint = certer

            for shiftPoint in range(len(self.diretionalInhiKernel)):
                matrixPoint = idx    - shiftPoint
                kernelPoint = certer - shiftPoint
                '''
                  This takes advantage of the fact that the convolution kernel is symmetric, 
                    there is no flip convolution kernel
                '''
                result += iptCell[matrixPoint] * self.diretionalInhiKernel[kernelPoint]
            opt.append(np.maximum(result, 0))

        return opt


