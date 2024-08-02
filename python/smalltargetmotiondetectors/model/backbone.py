from abc import ABC, abstractmethod
import warnings

from ..core import estmd_core, estmd_backbone, fracstmd_core, dstmd_core, backbonev2_core
from ..util.compute_module import compute_response, compute_direction

class BaseModel(ABC):
    """
    Base class for Small Target Motion Detector models.
    """

    def __init__(self):
        """ Constructor method.
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

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1': 'self.hRetina.hGaussianBlur.sigma',
            'sigma2': 'self.lobulaOpt.hGaussianBlur.sigma',
        }
        
    @abstractmethod
    def init_config(self, *args, **kwargs):
        """
        Abstract method for initializing model components.
        """
        pass

    @abstractmethod
    def model_structure(self, modelIpt, *args, **kwargs):
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
    
    def print_parameter(self):
        print(f'The parameter list of {self.__class__.__name__} includes: \n')
        paraList = eval(f'self._{self.__class__.__name__}__parameterList')
        if not paraList:
            print('\tNULL')
            return
        for name, value in paraList.items():
            if isinstance(value, tuple):
                for court, item in enumerate(value):
                    if court == 0:
                        print(name, '\t:', item, '\t = ', eval(item))
                    else:
                        print(' '*len(name), '\t ', item, '\t = ', eval(item))
            else:
                print(name, '\t:', value, '\t = ', eval(value))

    def set_parameter(self, **kwargs):
        """
        Sets parameters for the class instance based on provided keyword arguments.
        
        This method updates instance attributes using keyword arguments passed to it. 
        The attributes to be updated are determined by a private attribute that 
        maps parameter names to their respective instance attribute names or tuples of attribute names.

        Parameters:
        - **kwargs: Keyword arguments where each key-value pair represents a parameter name and its new value.
        
        Behavior:
        - The method iterates over each key-value pair in `kwargs`.
        - It retrieves the dictionary of parameter mappings for the current class instance by accessing a private attribute.
        - If the parameter name (`key`) exists in the dictionary:
            - If the corresponding value is a tuple, it assigns the new value to each attribute in the tuple using `setattr`.
            - If the corresponding value is not a tuple, it assigns the new value to the single attribute specified using `setattr`.
        - If the parameter name does not exist in the dictionary, a warning is issued.
        
        Raises:
        - None directly, but issues a warning if the parameter does not exist.
        """
        paraList = getattr(self, f'_{self.__class__.__name__}__parameterList', {})
        
        for key, value in kwargs.items():
            if key in paraList.keys():
                if isinstance(paraList[key], tuple):
                    for mapKey in paraList[key]:
                        exec(mapKey + ' = value') 
                else:
                    exec(paraList[key] + ' = value') 
            else:
                warnings.warn(f"Private variable '{key}' does not exist.", UserWarning)

    
class ESTMD(BaseModel):
    def __init__(self):
        # Call the superclass constructor
        super().__init__()
        # Initialize components
        self.hRetina = estmd_core.Retina()
        self.hLamina = estmd_core.Lamina()
        self.hMedulla = estmd_core.Medulla()
        self.hLobula = estmd_core.Lobula()

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
            'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
            'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
            'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
            'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
            'lambda1'   : 'self.hLamina.hLaminaLateralInhibition.lambda1',
            'lambda2'   : 'self.hLamina.hLaminaLateralInhibition.lambda2',
            'sigma2'    : 'self.hLamina.hLaminaLateralInhibition.sigma2',
            'sigma3'    : 'self.hLamina.hLaminaLateralInhibition.sigma3',
            'A'         : ('self.hMedulla.hTm2.hSubInhi.A', 'self.hMedulla.hTm3.hSubInhi.A'),
            'B'         : ('self.hMedulla.hTm2.hSubInhi.B', 'self.hMedulla.hTm3.hSubInhi.B'),
            'e'         : ('self.hMedulla.hTm2.hSubInhi.e', 'self.hMedulla.hTm3.hSubInhi.e'),
            'rho'       : ('self.hMedulla.hTm2.hSubInhi.rho', 'self.hMedulla.hTm3.hSubInhi.rho'),
            'sigma4'    : ('self.hMedulla.hTm2.hSubInhi.Sigma1', 'self.hMedulla.hTm3.hSubInhi.Sigma1'),
            'sigma5'    : ('self.hMedulla.hTm2.hSubInhi.Sigma2', 'self.hMedulla.hTm3.hSubInhi.Sigma2'),
            'n3'        : ('self.hMedulla.hTm1.hGammaDelay.order', 'self.hMedulla.hMi1.hGammaDelay.order'),
            'tau3'      : ('self.hMedulla.hTm1.hGammaDelay.tau', 'self.hMedulla.hMi1.hGammaDelay.tau')
        } 

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

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
            'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
            'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
            'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
            'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
            'A'         : 'self.hLobula.hSubInhi.A',
            'B'         : 'self.hLobula.hSubInhi.B',
            'e'         : 'self.hLobula.hSubInhi.e',
            'rho'       : 'self.hLobula.hSubInhi.rho',
            'sigma4'    : 'self.hLobula.hSubInhi.Sigma1',
            'sigma5'    : 'self.hLobula.hSubInhi.Sigma2',
            'order3'    : ('self.hMedulla.hTm1.hGammaDelay.order', 'self.hMedulla.hMi1.hGammaDelay.order'),
            'tau3'      : ('self.hMedulla.hTm1.hGammaDelay.tau', 'self.hMedulla.hMi1.hGammaDelay.tau'),
        }
        
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

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 


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

        # Bind model parameters and their corresponding parameter pointers.
        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
            'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
            'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
            'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
            'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
            'lambda1'   : 'self.hLamina.hLaminaLateralInhibition.lambda1',
            'lambda2'   : 'self.hLamina.hLaminaLateralInhibition.lambda2',
            'sigma2'    : 'self.hLamina.hLaminaLateralInhibition.sigma2',
            'sigma3'    : 'self.hLamina.hLaminaLateralInhibition.sigma3',
            'n4'        : 'self.hMedulla.hMi1Para4.hGammaDelay.order',
            'tau4'      : 'self.hMedulla.hMi1Para4.hGammaDelay.tau',
            'n5'        : 'self.hMedulla.hTm1Para5.hGammaDelay.order',
            'tau5'      : 'self.hMedulla.hTm1Para5.hGammaDelay.tau',
            'n6'        : 'self.hMedulla.hTm1Para6.hGammaDelay.order',
            'tau6'      : 'self.hMedulla.hTm1Para6.hGammaDelay.tau',
            'alpha1'    : 'self.hLobula.alpha1', 
            'A'         : 'self.hLobula.hLateralInhi.A',
            'B'         : 'self.hLobula.hLateralInhi.B', 
            'e'         : 'self.hLobula.hLateralInhi.e', 
            'rho'       : 'self.hLobula.hLateralInhi.rho', 
            'sigma4'    : 'self.hLobula.hLateralInhi.Sigma1', 
            'sigma5'    : 'self.hLobula.hLateralInhi.Sigma2', 
        } 

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

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 

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

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1': 'self.hRetina.hGaussianBlur.sigma',
            'alpha' : 'self.hLamina.alpha',
            'delta' : 'self.hLamina.delta',
            'gLeak' : ('self.hMedulla.hMi4.gLeak', 'self.hMedulla.hTm9.gLeak'),
            'vRest' : ('self.hMedulla.hMi4.vRest', 'self.hMedulla.hTm9.vRest'),
            'vEx'   : ('self.hMedulla.hMi4.vExci', 'self.hMedulla.hTm9.vExci'),
            'A'     : 'self.hLobula.hSubInhi.A', 
            'B'     : 'self.hLobula.hSubInhi.B', 
            'e'     : 'self.hLobula.hSubInhi.e', 
            'rho'   : 'self.hLobula.hSubInhi.rho', 
            'sigma2': 'self.hLobula.hSubInhi.Sigma1', 
            'sigma3': 'self.hLobula.hSubInhi.Sigma2', 
        } 
        
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
        self.lobulaOpt, self.modelOpt['direction'], _ \
            = self.hLobula.process(self.medullaOpt[0], 
                                   self.medullaOpt[1], 
                                   self.laminaOpt )
        self.modelOpt['response'] = self.lobulaOpt




