import numpy as np

from .base_core import BaseCore
from .math_operator import GammaBandPassFilter, SurroundInhibition
from . import estmd_core

class Lamina(BaseCore):
    """Lamina layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lamina object
        super().__init__()
        # Initialize the GammaBankPassFilter component
        self.hGammaBandPassFilter = GammaBandPassFilter()

    def init_config(self):
        """Initialization method."""
        # This method initializes the Lamina layer component
        self.hGammaBandPassFilter.init_config()

    def process(self, laminaIpt):
        """Processing method."""
        # Applies gamma bank pass filtering to the input
        # Process the input using GammaBankPassFilter
        laminaOpt = self.hGammaBandPassFilter.process(laminaIpt)
        # Store the output in Opt property
        self.Opt = laminaOpt
        return laminaOpt


class Medulla(BaseCore):
    """Medulla layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Medulla object
        super().__init__()
        # Initialize components
        self.hTm1 = estmd_core.Tm1()
        self.hMi1 = estmd_core.Mi1()
        self.hTm2 = Tm2()
        self.hTm3 = Tm3()

    def init_config(self):
        """Initialization method."""
        # This method initializes the Medulla layer components
        self.hTm1.init_config()
        self.hTm2.init_config()
        self.hTm3.init_config()

    def process(self, MedullaIpt):
        """Processing method."""
        # Applies processing to the input and returns the output
        
        # Process Tm2 and Tm3 components
        tm2Signal = self.hTm2.process(MedullaIpt)
        tm3Signal = self.hTm3.process(MedullaIpt)

        # Process Tm1 component using output of Tm2
        tm1Signal = self.hTm1.process(tm2Signal)
        
        # Store the output signals in Opt property
        varageout = [tm3Signal, tm1Signal]
        self.Opt = varageout
        return varageout


class Lobula(BaseCore):
    """Lobula layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        # Initialize the SurroundInhibition component
        self.hSubInhi = SurroundInhibition()
        # Parameters related to the recombination of ON and OFF channels
        self.a = 0
        self.b = 0
        self.c = 1

    def init_config(self):
        """Initialization method."""
        # This method initializes the Lobula layer component
        self.hSubInhi.init_config()

    def process(self, varagein):
        """Processing method."""
        # Performs a correlation operation on the ON and OFF channels
        # and then applies surround inhibition
        
        # Extract ON and OFF channel signals from the input
        onSignal = varagein[0]
        offSignal = varagein[1]
        
        # Perform the correlation operation
        correlationOutput = (
            self.a * onSignal +
            self.b * offSignal +
            self.c * onSignal * offSignal
        )

        # Apply surround inhibition
        lobulaOpt = self.hSubInhi.process(correlationOutput)
        
        # Store the output in Opt property
        self.Opt = lobulaOpt
        return lobulaOpt, correlationOutput


class Tm2(BaseCore):
    """Tm2"""

    def __init__(self):
        """Constructor method."""
        # Initializes the Tm2 object
        super().__init__()

    def init_config(self):
        """Initialization method."""
        # This method initializes Tm2 (no initialization required)
        pass

    def process(self, tm2Ipt):
        """Processing method."""
        # Applies surround inhibition to the input to generate the output
        
        tm2Opt = np.maximum(-tm2Ipt, 0)  # Apply surround inhibition
        self.Opt = tm2Opt  # Store the output in Opt property
        return tm2Opt
    

class Tm3(BaseCore):
    """Tm3 is On signal with a Surround Inhibition."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Tm3 object
        super().__init__()

    def init_config(self):
        """Initialization method."""
        # This method initializes Tm3 (no initialization required)
        return

    def process(self, tm3OptIpt):
        """Processing method."""
        # Applies a surround inhibition to the input to generate the output
        
        tm3Opt = np.maximum(tm3OptIpt, 0)  # Apply surround inhibition
        self.Opt = tm3Opt  # Store the output in Opt property
        return tm3Opt






