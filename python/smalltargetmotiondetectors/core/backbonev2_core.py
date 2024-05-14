from cv2 import filter2D
import math
import numpy as np

from .base_core import BaseCore
from .math_operator import SurroundInhibition
from ..util.matrixnms import MatrixNMS
from scipy.spatial.distance import cdist


class MedullaCell(BaseCore):
    # MedullaCell cell in Medulla layer

    def __init__(self):
        super().__init__()

        self.afterhyperpolarizationV = -50
        self.coeffDecay = 0.8
        self.coeffInhi = 0.5
        self.Voltage = None
        self.lastInput = None

    def init_config(self):
        pass

    def process(self, iptMatrix):
        if self.lastInput is None:
            m, n = iptMatrix.shape
            self.Voltage = np.zeros((m, n))
            self.lastInput = np.zeros((m, n), dtype=bool)

        # Decay
        isDecay = self.lastInput
        self.Voltage[isDecay] = self.coeffDecay * self.Voltage[isDecay]

        # Inhi
        isInhi = ~self.lastInput
        self.Voltage[isInhi] = self.coeffInhi * self.Voltage[isInhi]

        # Accumulation
        self.Voltage += iptMatrix

        # record
        self.lastInput = iptMatrix.astype(bool)

        return self.Voltage


class Medulla(BaseCore):
    """
    Medulla layer of the motion detection system.

    Illustration:

    Lamina   LMC1 (L1)            LMC2 (L2)   
                |          *         |          
    ------------|--------------------|--------- 
    Medulla     |  {ON}    *         |  {OFF}   
                |          *         v          
                |    /-------<-<--- Tm2         
                v    |     *         |          
               Mi1 --|->->----\      |          
                |    v     *   v     v          
                |   Mi9    *  TmOn   |          
                |    |     *   |     |          
                v    v     *   v     v          
               Mi4 <-/     *   \->-> Tm9         
                |          *         |          
    ------------|--------------------|--------- 
    Lobula      |          *         |
                v                    v          

    """

    def __init__(self):
        super().__init__()
        # Initialize components
        self.hTm2 = Tm2()
        self.hTm3 = Tm3()

    def init_config(self):
        # Initialize configurations
        self.hTm2.init_config()
        self.hTm3.init_config()

    def process(self, medullaIpt):
        """
        Process the input through the Medulla layer.

        Args:
        - medullaIpt (array-like): Input to the Medulla layer.

        Returns:
        - tm3Signal (array-like): Output signal from Tm3.
        - tm2Signal (array-like): Output signal from Tm2.
        """
        # Process through Tm2 and Tm3
        tm2Signal = self.hTm2.process(medullaIpt)
        tm3Signal = self.hTm3.process(medullaIpt)

        # Store the output signals
        self.Opt = [tm3Signal, tm2Signal]

        return tm3Signal, tm2Signal


class Tm2(MedullaCell):
    """
    Tm2 cell in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, tm2Ipt):
        """
        Processing method.

        Args:
        - tm2Ipt (array-like): Input to the Tm2 cell.

        Returns:
        - tm2Opt (array-like): Output of the Tm2 cell.
        """
        offSignal = np.maximum(-tm2Ipt, 0)
        tm2Opt = super().process(offSignal)
        self.Opt = tm2Opt
        return tm2Opt


class Tm3(MedullaCell):
    """
    Tm3 cell in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, tm3Ipt):
        """
        Processing method.

        Args:
        - tm3Ipt (array-like): Input to the Tm3 cell.

        Returns:
        - tm3Opt (array-like): Output of the Tm3 cell.
        """
        onSignal = np.maximum(tm3Ipt, 0)
        tm3Opt = super().process(onSignal)
        self.Opt = tm3Opt
        return tm3Opt


class Lobula(BaseCore):
    """
    Lobula layer of the motion detection system.
    """

    def __init__(self):
        super().__init__()
        self.hSubInhi = SurroundInhibition()
        self.hDireCell = CustomDirection()

    def init_config(self):
        """
        Initialization method.
        """
        self.hSubInhi.init_config()
        self.hDireCell.init_config()


    def process(self, onSignal, offSignal, laminaOpt):
        """
        Processing method.

        Args:
        - varagein (tuple): Tuple containing the ON and OFF channel signals.

        Returns:
        - varargout (tuple): Tuple containing the processed output(s).
        """

        direction = self.hDireCell.process(laminaOpt, onSignal, offSignal)

        correlationOutput = onSignal * offSignal
        lobulaOpt = self.hSubInhi.process(correlationOutput)

        self.Opt = lobulaOpt, direction, correlationOutput

        return lobulaOpt, direction, correlationOutput


class CustomDirection(BaseCore):
    def __init__(self, kernelSize=3):
        self.kernelSize = kernelSize
        self.kernelCos = np.zeros((self.kernelSize, self.kernelSize))
        self.kernelSin = np.zeros((self.kernelSize, self.kernelSize))

        self.direMatrix = None

    def init_config(self):
        """
        Initialization method.
        """

        half_kernel = self.kernelSize // 2
        for x in range(-half_kernel, half_kernel + 1):
            for y in range(-half_kernel, half_kernel + 1):
                r = math.sqrt(x ** 2 + y ** 2)
                if r == 0:
                    continue
                # 计算cosine值
                self.kernelCos[y + half_kernel, x + half_kernel] = x / r
                self.kernelSin[y + half_kernel, x + half_kernel] = -y / r

    def process(self,
            laminaOpt,
            tm3Opt, # On
            tm2Opt  # Off
            ):
        
        m,n = laminaOpt.shape
        directionCos = np.zeros((m,n))
        directionSin = np.zeros((m,n))
        self.direMatrix = np.zeros((m,n))
         
        isNotZero = (tm3Opt > 0)
        self.direMatrix[isNotZero] = tm2Opt[isNotZero] / tm3Opt[isNotZero]

        isReciprocal = (laminaOpt < 0) & (self.direMatrix > 0)
        self.direMatrix[isReciprocal] = 1 / self.direMatrix[isReciprocal]

        directionCos = filter2D(self.direMatrix, -1, self.kernelCos)
        directionSin = filter2D(self.direMatrix, -1, self.kernelSin)

        direction = np.arctan2(directionSin, directionCos)

        # Adjust directions to be in the range [0, 2*pi]
        direction[direction < 0] += 2 * math.pi
        
        return direction
    



