import torch

from .base_core import BaseCore
from .math_operator import GammaBandPassFilter, SpatialInhibition
from . import estmd_core


class Lamina(GammaBandPassFilter):
    pass


class Medulla(BaseCore):
    """Medulla layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Medulla object
        super().__init__()
        # Initialize components
        self.tm1 = estmd_core.Tm1()
        self.mi1 = estmd_core.Mi1()
        self.tm2 = Tm2()
        self.tm3 = Tm3()

    def setup(self):
        """Initialization method."""
        # This method initializes the Medulla layer components
        self.tm1.setup()
        self.mi1.setup()
        self.tm2.setup()
        self.tm3.setup()

    def forward(self, MedullaIpt):
        """Processing method."""
        # Applies processing to the input and returns the output
        
        # Process Tm2 and Tm3 components
        tm2_output = self.tm2.forward(MedullaIpt)
        tm3_output = self.tm3.forward(MedullaIpt)

        # Process Tm1 component using output of Tm2
        tm1_output = self.tm1.forward(tm2_output)

        # Store the output signals in output property
        self.output = [tm3_output, tm1_output]

        return self.output


class Lobula(BaseCore):
    """Lobula layer of the motion detection system."""
    
    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        # Initialize the SpatialInhibition component
        self.spatial_inhibition = SpatialInhibition()
        # Parameters related to the recombination of ON and OFF channels
        self.a = 0
        self.b = 0
        self.c = 1

    def setup(self):
        """Initialization method."""
        # This method initializes the Lobula layer component
        self.spatial_inhibition.setup()

    def forward(self, on_output, off_output):
        """Processing method."""
        # Performs a correlation operation on the ON and OFF channels
        # and then applies surround inhibition
        
        # Perform the correlation operation
        correlationOutput = (
            self.a * on_output +
            self.b * off_output +
            self.c * on_output * off_output
        )

        # Apply surround inhibition
        lobulaoutput = self.spatial_inhibition.forward(correlationOutput)
        
        # Store the output in output property
        self.output = lobulaoutput
        return lobulaoutput, correlationOutput


class Tm2(BaseCore):
    """Tm2 without spatial inhibition."""
    def forward(self, x):
        """Processing method."""
        # Applies surround inhibition to the input to generate the output
        self.output = torch.clamp(-x, min=0)  # Apply surround inhibition
        return self.output
    

class Tm3(BaseCore):
    """Tm3 without spatial inhibition."""
    def forward(self, tm3outputIpt):
        """Processing method."""
        # Applies a surround inhibition to the input to generate the output
        self.output = torch.clamp(tm3outputIpt, min=0)  # Apply surround inhibition
        return self.output






