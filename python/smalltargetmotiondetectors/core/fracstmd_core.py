import numpy as np
import math

from .base_core import BaseCore
from ..util.create_kernel import create_fracdiff_kernel
from ..util.datarecord import CircularCell
from ..util.compute_module import compute_circularcell_conv

class Lamina(BaseCore):
    """Lamina class for the lamina layer."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lamina object
        super().__init__()
        self.alpha = 0.8
        self.fps = 240
        self.delta = 20
        self.fracKernel = None
        self.paraCur = None
        self.paraPre = None
        self.preLaminaIpt = None
        self.preLaminaOpt = None
        self.cellRetinaOutput = None
    
    def init_config(self):
        """Initialization method."""
        # Initializes the fractional differential kernel        
        self.fracKernel = create_fracdiff_kernel(self.alpha, self.delta)
        self.paraCur = self.fracKernel[0]
        self.paraPre = math.exp(-self.alpha / (1 - self.alpha))
        self.cellRetinaOutput = CircularCell(self.delta)
    
    def process(self, LaminaIpt):
        """Processing method."""
        # Processes the LaminaIpt to generate the lamina output
        if self.preLaminaIpt is None:
            diffLaminaIpt = np.zeros_like(LaminaIpt)
        else:
            # First order difference
            diffLaminaIpt = LaminaIpt - self.preLaminaIpt
        
        laminaopt = self.compute_by_iteration(diffLaminaIpt)
        
        self.Opt = laminaopt
        self.preLaminaIpt = LaminaIpt
        return laminaopt
    
    def compute_by_conv(self, diffLaminaIpt):
        """Computes the lamina output by convolution."""
       
        self.cellRetinaOutput.circrecord(diffLaminaIpt)
        laminaopt = compute_circularcell_conv(self.cellRetinaOutput, self.fracKernel)
        return laminaopt
    
    def compute_by_iteration(self, diffLaminaIpt):
        """Computes the lamina output by iteration."""
        if self.preLaminaOpt is None:
            laminaopt = self.paraCur * diffLaminaIpt
        else:
            laminaopt = self.paraCur * diffLaminaIpt + self.paraPre * self.preLaminaOpt
        
        self.preLaminaOpt = laminaopt
        return laminaopt
