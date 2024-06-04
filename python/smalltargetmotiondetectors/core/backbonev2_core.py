from cv2 import filter2D
import math
import numpy as np

from .base_core import BaseCore
from .math_operator import SurroundInhibition


class MECumulativeCell(BaseCore):
    # CumulativeCell cell in Medulla layer

    def __init__(self):
        super().__init__()

        self.coeffDecay = 0.8
        self.coeffInhi = 0.5
        self.postMembranePotential = None

    def init_config(self):
        pass

    def process(self, iptMatrix):
        if self.postMembranePotential is None:
            self.postMembranePotential = np.zeros_like(iptMatrix)

        # Decay
        postMP = self.coeffDecay * self.postMembranePotential

        # Accumulation
        postMP = postMP + iptMatrix
        
        # Inhi
        isInhi = (iptMatrix == 0)
        postMP[isInhi] *= self.coeffInhi

        self.postMembranePotential = postMP

        return postMP


class Mi4(MECumulativeCell):
    """
    Mi4 cell of ON pathway in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, iptMatrix: np.array):
        """
        Processing method.

        Args:
        - iptMatrix (array-like): Input

        Returns:
        - self.Opt (array-like): Output
        """
        onSignal = np.maximum(iptMatrix, 0)
        self.Opt = super().process(onSignal)
        return self.Opt


class Tm9(MECumulativeCell):
    """
    Tm9 cell of OFF pathway in the Medulla layer.
    """

    def __init__(self):
        super().__init__()

    def init_config(self):
        """
        Initialization method.
        """
        pass

    def process(self, iptMatrix: np.array):
        """
        Processing method.

        Args:
        - iptMatrix (array-like): Input

        Returns:
        - self.Opt (array-like): Output
        """
        offSignal = np.maximum(-iptMatrix, 0)
        self.Opt = super().process(offSignal)
        return self.Opt
    

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
        self.hMi4 = Mi4()
        self.hTm9 = Tm9()

    def init_config(self):
        # Initialize configurations
        self.hMi4.init_config()
        self.hTm9.init_config()

    def process(self, medullaIpt):
        """
        Process the input through the Medulla layer.

        Args:
        - medullaIpt (array-like): Input to the Medulla layer.

        Returns:
        - onSignal (array-like): Output signal from Mi4.
        - offSignal (array-like): Output signal from Tm9.
        """
        # Process through hMi4 and hTm9
        onSignal = self.hMi4.process(medullaIpt)
        offSignal = self.hTm9.process(medullaIpt)

        # Store the output signals
        self.Opt = [onSignal, offSignal]

        return onSignal, offSignal
    

class LPTangentialCell(BaseCore):
    def __init__(self, kernelSize=3):
        self.kernelSize = kernelSize
        self.lptcMatrix = None

    def init_config(self):
        """
        Initialization method.
        """
        self.kernelCos = np.zeros((self.kernelSize, self.kernelSize))
        self.kernelSin = np.zeros((self.kernelSize, self.kernelSize))

        halfKernel = self.kernelSize // 2
        for x in range(-halfKernel, halfKernel + 1):
            for y in range(-halfKernel, halfKernel + 1):
                r = math.sqrt(x ** 2 + y ** 2)
                if r == 0:
                    continue
                # 计算cosine值
                self.kernelCos[y + halfKernel, x + halfKernel] = x / r
                self.kernelSin[y + halfKernel, x + halfKernel] = -y / r

    def process(self,
            laminaOpt: np.array,
            onSignal: np.array, # On
            offSignal: np.array  # Off
            ):
        
        # get indexes
        onDire = (onSignal>0) & (laminaOpt>0)
        offDire = (offSignal>0) & (laminaOpt<0)
        
        direMatrix = np.zeros_like(laminaOpt)
        # On direction
        direMatrix[onDire] = offSignal[onDire] / onSignal[onDire]
        # Off direction
        direMatrix[offDire] = onSignal[offDire] / offSignal[offDire]

        directionCos = filter2D(direMatrix, -1, self.kernelCos)
        directionSin = filter2D(direMatrix, -1, self.kernelSin)
        direction = np.arctan2(directionSin, directionCos)
        # Adjust directions to be in the range [0, 2*pi]
        direction[direction < 0] += 2 * math.pi
        
        self.lptcMatrix = direMatrix
        self.Opt = {'direction': direction, 
                    'lptcMatric': direMatrix}
        return direction
    

class Lobula(BaseCore):
    """
    Lobula layer of the motion detection system.
    """

    def __init__(self):
        super().__init__()
        self.hSubInhi = SurroundInhibition()
        self.hLPTC = LPTangentialCell()

    def init_config(self):
        """
        Initialization method.
        """
        self.hSubInhi.init_config()
        self.hLPTC.init_config()

    def process(self, onSignal, offSignal, laminaOpt):
        """
        Processing method.

        Args:
        - onSignal (np.array):  ON channel signal from medulla layer.
        - offSignal (np.array):  OFF channel signal from medulla layer.
        - laminaOpt (np.array):  output signal from medulla layer.

        Returns:
        - lobulaOpt (np.array): output for location.
        - direction (np.array): output for direction.
        - correlationOutput (np.array): output without inhibition.
        """

        direction = self.hLPTC.process(laminaOpt, onSignal, offSignal)

        correlationOutput = onSignal * offSignal
        lobulaOpt = self.hSubInhi.process(correlationOutput)

        self.Opt = lobulaOpt, direction, correlationOutput

        return lobulaOpt, direction, correlationOutput



