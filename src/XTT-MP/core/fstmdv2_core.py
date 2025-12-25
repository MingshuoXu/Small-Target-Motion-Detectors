import numpy as np

from . import fracstmd_core

class Lamina(fracstmd_core.Lamina):
    """Lamina class for the lamina layer."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lamina object
        super().__init__()
        self.loopLaminaOpt = None
        self.isInLoop = False

    def process(self, LaminaIpt):
        """Processing method."""
        # Processes the LaminaIpt to generate the lamina output
        if self.preLaminaIpt is None:
            diffLaminaIpt = np.zeros_like(LaminaIpt)
        else:
            # First order difference
            diffLaminaIpt = LaminaIpt - self.preLaminaIpt
        
        laminaOpt = self.compute_by_iteration(diffLaminaIpt)
        self.Opt = laminaOpt
        
        if not self.isInLoop:
            self.preLaminaIpt = LaminaIpt

        return laminaOpt

    def compute_by_iteration(self, diffLaminaIpt):
        """Compute lamina output by iteration."""
        if self.preLaminaOpt is None:
            laminaopt = self.paraCur * diffLaminaIpt
        else:
            if not self.isInLoop:
                self.preLaminaIpt = self.loopLaminaOpt
            laminaopt = self.paraCur * diffLaminaIpt + self.paraPre * self.preLaminaOpt
        
        self.loopLaminaOpt = laminaopt
        return laminaopt
