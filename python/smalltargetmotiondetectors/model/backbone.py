from abc import ABC, abstractmethod

from ..core import estmd_core, estmd_backbone, fracstmd_core, dstmd_core, backbonev2_core
from ..util.compute_module import compute_response, compute_direction

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
        self.modelOpt = {'response': [], 'direction': []}

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
        self.modelOpt['response'] = self.lobulaOpt

class ESTMDBackbone(BaseModel):
    """
    ESTMDBackbone - Backbone model based on ESTMD
    """

    def __init__(self):
        """
        ESTMDBackbone Constructor method
        Initializes an instance of the ESTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = estmd_backbone.Lamina()
        self.hMedulla = estmd_backbone.Medulla()
        self.hLobula = estmd_backbone.Lobula()

    def init_config(self):
        """
        INIT Method
        Initializes the ESTMDBackbone components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        """
        MODEL_STRUCTURE Method
        Defines the structure of the ESTMDBackbone model.
        """
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt, _ = self.hLobula.process(self.medullaOpt)

        # Set model response
        self.modelOpt['response'] = self.lobulaOpt


class FracSTMD(ESTMDBackbone):
    """
    FracSTMD - Fractional-order Small Target Motion Detector
    """

    def __init__(self):
        """
        FracSTMD Constructor method
        Initializes an instance of the FracSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Customize Lamina and Lobula components
        self.hLamina = fracstmd_core.Lamina()
        self.hMedulla.hTm1.hGammaDelay.order = 100
        self.hLobula.hSubInhi.e = 1.8


class DSTMD(BaseModel):
    """
    DSTMD - Directional-Small Target Motion Detector
    """
    
    def __init__(self):
        """
        DSTMD Constructor method
        Initializes an instance of the DSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = estmd_core.Lamina()
        self.hMedulla = dstmd_core.Medulla()
        self.hLobula = dstmd_core.Lobula()

    def init_config(self):
        """
        INIT Method
        Initializes the DSTMD components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        """
        MODEL_STRUCTURE Method
        Defines the structure of the DSTMD model.
        """        
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.lobulaOpt)
        self.modelOpt['direction'] = compute_direction(self.lobulaOpt)


class DSTMDBackbone(BaseModel):
    """
    DSTMDBackbone - Backbone model based on DSTMD
    """
    
    def __init__(self):
        """
        DSTMDBackbone Constructor method
        Initializes an instance of the DSTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = estmd_backbone.Lamina()
        self.hMedulla = dstmd_core.Medulla()
        self.hLobula = dstmd_core.Lobula()

    def init_config(self):
        """
        INIT Method
        Initializes the DSTMDBackbone components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        """
        MODEL_STRUCTURE Method
        Defines the structure of the DSTMDBackbone model.
        """
        
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.lobulaOpt)
        self.modelOpt['direction'] = compute_direction(self.lobulaOpt)


class Backbonev2(BaseModel):
    """
    Backbonev2 - SNN in STMD based model
    """

    def __init__(self):
        """
        Constructor function
        """
        super().__init__()

        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = fracstmd_core.Lamina()
        self.hMedulla = backbonev2_core.Medulla()
        self.hLobula = backbonev2_core.Lobula()

        self.hLamina.alpha = 0.3

    def init_config(self):
        """
        Initialize configurations for the components
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, modelIpt):
        """
        Define the structure of the model
        """
        self.retinaOpt = self.hRetina.process(modelIpt)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt, self.modelOpt['direction'], _ = self.hLobula.process(self.medullaOpt)
        self.modelOpt['response'] = self.lobulaOpt




