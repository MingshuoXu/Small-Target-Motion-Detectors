from cv2 import filter2D
import math
import numpy as np

from .base_core import BaseCore
from .math_operator import SurroundInhibition


class MECumulativeCell(BaseCore):
    # CumulativeCell cell in Medulla layer

    V_REST = 0;  # passive/rest potentials;
    V_EXCI = 1;  # excitatory saturation potentials;

    def __init__(self):
        super().__init__()

        self.coeffDecay = 0.5 # coefficient of decay
        self.postMP = None

    def init_config(self):
        pass

    def process(self, samePolarity, oppoPolarity):
        if self.postMP is None:
            self.postMP = np.zeros_like(samePolarity)

        # Decay
        decayTerm = self.coeffDecay * (self.V_REST - self.postMP)
        # Inhibition
        # inhiGain = np.exp(oppoPolarity)
        # To reduce the computational load, a first-order Taylor expansion was used.
        inhiGain = 1 + oppoPolarity + oppoPolarity**2
        # Excitation
        exciTerm = samePolarity * (self.V_EXCI - self.postMP)
        
        # Euler method for solving ordinary differential equation
        self.postMP = self.postMP + inhiGain * decayTerm + exciTerm
            
        return self.postMP


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
        self.hMi4 = MECumulativeCell()
        self.hTm9 = MECumulativeCell()

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
        onSignal = np.maximum(medullaIpt, 0); # ON
        offSignal = np.maximum(-medullaIpt, 0); # OFF
        
        mi4Opt = self.hMi4.process(onSignal, offSignal); # ON
        tm9Opt = self.hTm9.process(offSignal, onSignal); # OFF

        # Store the output signals
        self.Opt = (mi4Opt, tm9Opt)

        return mi4Opt, tm9Opt
    

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



