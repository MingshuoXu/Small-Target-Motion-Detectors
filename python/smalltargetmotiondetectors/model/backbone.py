from abc import ABC, abstractmethod

from ..core import estmd_core


class BaseModel(ABC):
    """
    Base class for Small Target Motion Detector models.
    """

    def __init__(self):
        """
        Constructor method.
        """
        
        self.hRetina = None # Handle for the retina layer
        self.hLamina = None # Handle for the lamina layer
        self.hMedulla = None # Handle for the medulla layer
        self.hLobula = None # Handle for the lobula layer

        self.inputFps = None

        self.retinaOpt = None # Retina layer output
        self.laminaOpt = None # Lamina layer output
        self.medullaOpt = None # Medulla layer output
        self.lobulaOpt = None # Lobula layer output

        # Model output structure
        self.odelOpt = {'response': [], 'direction': []}

    @abstractmethod
    def init_config(self):
        """
        Abstract method for initializing model components.
        """
        pass

    @abstractmethod
    def model_structure(self, modelIpt):
        """
        Abstract method for defining model structure.
        
        Parameters:
            modelIpt: Input for model processing.
        """
        pass

    def process(self, modelIpt):
        """
        Processes the input and returns the model output.

        Parameters:
            modelIpt: Input for model processing.

        Returns:
            modelOpt: Model output structure.
        """
        # Call the model structure method
        self.model_structure(modelIpt)
        # Return the model output
        return self.modelOpt
    

class ESTMD(BaseModel):
    def __init__(self):
        # Call the superclass constructor
        super().__init__()
        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = estmd_core.Lamina()
        self.hMedulla = estmd_core.Medulla()
        self.hLobula = estmd_core.Lobula()

    def init_config(self):
        # Initialize ESTMD components
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()

    def model_structure(self, iptMatrix):
        # Define the structure of the ESTMD model
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)
        # direction not set in the  ESTMD model
        self.modelOpt = {
            'response': self.lobulaOpt,
            'direction': None
            } 










