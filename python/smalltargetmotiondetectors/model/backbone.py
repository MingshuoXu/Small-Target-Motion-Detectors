from abc import ABC, abstractmethod
import warnings
import logging
import time

import torch

from ..core import estmd_core, estmd_backbone, fracstmd_core, dstmd_core
from ..util.compute_module import compute_response, compute_direction


class BaseModel(ABC):
    """ Base class for Small Target Motion Detector models. """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # here is just an example
            'sigma1': 'self.hRetina.hGaussianBlur.sigma',
            'sigma2': 'self.lobulaOpt.hGaussianBlur.sigma',
        }

    def __init__(self, device = 'cpu'):
        """ Constructor method.
        """
        self.device = device
        
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

    def init_config(self, *args, **kwargs):
        """
        Abstract method for initializing model components.
        """
        pass

    def _initialize(self, *args, **kwargs):
        return self.init_config(*args, **kwargs)

    @abstractmethod
    def model_structure(self, modelIpt, *args, **kwargs):
        """
        Abstract method for defining model structure.
        
        Parameters:
            modelIpt: Input for model processing.
        """
        pass

    def process(self, modelIpt):
        """ Processes the input and returns the model output.

        Parameters:
            modelIpt: Input for model processing.

        Returns:
            modelOpt: Model output structure.
            time_end: Time taken for processing.
        """
        
        time_start = time.time()
        # Call the model structure method
        self.model_structure(modelIpt)
        if self.device != 'cpu':
            torch.cuda.synchronize()
        time_end = time.time() - time_start
        # Return the model output
        return self.modelOpt, time_end
    
    def print_para(self):
        logger = logging.getLogger(__name__)

        paraList = eval(f'self._{self.__class__.__name__}__paraMappingList')

        if not paraList:
            logger.info(f'The parameters of <{self.__class__.__name__}> is empty.')
            return
        
        msg = f'The parameters of <{self.__class__.__name__}> are:\n'
        for name, value in paraList.items():
            msg += f'  {name:6}'
            if isinstance(value, tuple):
                for i, item in enumerate(value):
                    if i == 0:
                        msg += ' -->'
                    elif i == len(value) - 1:
                        msg += f'{' '*len(name):6} \\--->'
                    else:
                        msg += f'{' '*len(name):6} |--->'
                    msg += f' {item} = {eval(item)}\n'
            else:
                msg += f' --> {value} = {eval(value)}\n'
                
        logger.info(msg)
        
    def set_para(self, **kwargs):
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
        _paraList = getattr(self, f'_{self.__class__.__name__}__paraMappingList', {})
        
        for key, value in kwargs.items():
            if key in _paraList.keys():
                if isinstance(_paraList[key], tuple):
                    for mapKey in _paraList[key]:
                        exec(mapKey + ' = value') 
                else:
                    exec(_paraList[key] + ' = value') 
            else:
                warnings.warn(f"Private variable '{key}' does not exist.", UserWarning)

    # The following code has been retained for compatibility with older versions
    def print_parameter(self):
        """Compatibility method for older versions. See `print_para` for the new version."""
        self.print_para()

    def set_parameter(self, **kwargs):
        """Compatibility method for older versions. See `set_para` for the new version."""
        self.set_para(**kwargs)
    

class ESTMD(BaseModel):
    """ ESTMD: Elementary small target motion detector

    Ref: 
        * Wiederman S D, Shoemaker P A, O'Carroll D C. A model for the detection of moving targets in visual clutter inspired by insect physiology[J]. PloS one, 2008, 3(7): e2784.
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.

    Remark:
        The implementation and parameters in this code follow Ref [2].

    Parameters:
        Retina:
            - sigma1: Standard deviation for Gaussian blur in the retina, representing visual preprocessing. (Eq. 1)
        Lamina:
            - n1, tau1: Order and time constant for the first gamma bandpass filter delay in the lamina. (Eq. 4)
            - n2, tau2: Order and time constant for the second gamma bandpass filter delay in the lamina. (Eq. 4)
            - sigma2, sigma3: Standard deviations for lateral inhibition in the lamina. (Eq. 8-9)
            - lambda1, lambda2: Parameters controlling lateral inhibition intensity. (Eq. 10-11)
        Medulla:
            - A, B: Parameters for second-inhibition mechanisms in the medulla. (Eq. 20)
            - sigma4, sigma5: Standard deviations for second-inhibition spatial spread in medulla. (Eq. 21)
            - e, rho: Non-linear interaction parameters in second-inhibition. (Eq. 21)
            - n3, tau3: Order and time constant for gamma delay in neurons Tm1 and Mi1 in the medulla. (Eq. 24)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (1)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order', # Eq. (4)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        'sigma2'    : 'self.hLamina.hLaminaLateralInhibition.sigma2', # Eq. (8)(9)
        'sigma3'    : 'self.hLamina.hLaminaLateralInhibition.sigma3',
        'lambda1'   : 'self.hLamina.hLaminaLateralInhibition.lambda1', # Eq. (10)(11)
        'lambda2'   : 'self.hLamina.hLaminaLateralInhibition.lambda2',
        # medulla
        'A'         : ('self.hMedulla.hTm2.hSubInhi.A', 'self.hMedulla.hTm3.hSubInhi.A'), # Eq. (20)
        'B'         : ('self.hMedulla.hTm2.hSubInhi.B', 'self.hMedulla.hTm3.hSubInhi.B'),
        'sigma4'    : ('self.hMedulla.hTm2.hSubInhi.Sigma1', 'self.hMedulla.hTm3.hSubInhi.Sigma1'), # Eq. (21)
        'sigma5'    : ('self.hMedulla.hTm2.hSubInhi.Sigma2', 'self.hMedulla.hTm3.hSubInhi.Sigma2'),
        'e'         : ('self.hMedulla.hTm2.hSubInhi.e', 'self.hMedulla.hTm3.hSubInhi.e'),
        'rho'       : ('self.hMedulla.hTm2.hSubInhi.rho', 'self.hMedulla.hTm3.hSubInhi.rho'),
        'n3'        : ('self.hMedulla.hTm1.hGammaDelay.order', 'self.hMedulla.hMi1.hGammaDelay.order'), # Eq. (24)
        'tau3'      : ('self.hMedulla.hTm1.hGammaDelay.tau', 'self.hMedulla.hMi1.hGammaDelay.tau')
        } 
     
    def __init__(self, device = 'cpu'):
        # Call the superclass constructor
        super().__init__(device=device)
        # Initialize components
        self.hRetina = estmd_core.Retina(device=device)
        self.hLamina = estmd_core.Lamina(device=device)
        self.hMedulla = estmd_core.Medulla(device=device)
        self.hLobula = estmd_core.Lobula(device=device)

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
    """ ESTMDBackbone: A backbone based on ESTMD 

    Ref: 
        * Wiederman S D, Shoemaker P A, O'Carroll D C. A model for the detection of moving targets in visual clutter inspired by insect physiology[J]. PloS one, 2008, 3(7): e2784.
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
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
    
    def __init__(self, device = 'cpu'):
        """ ESTMDBackbone Constructor method

        Initializes an instance of the ESTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__(device=device)

        # Initialize components
        self.hRetina = estmd_core.Retina(device=device)
        self.hLamina = estmd_backbone.Lamina()
        self.hMedulla = estmd_backbone.Medulla(device=device)
        self.hLobula = estmd_backbone.Lobula(device=device)
        
    def init_config(self):
        """ INIT Method

        Initializes the ESTMDBackbone components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

        
    def model_structure(self, iptMatrix):
        """ MODEL_STRUCTURE Method

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
    """ FracSTMD: Fractional-order Small Target Motion Detector

    Ref: 
        * Xu M, Wang H, Chen H, et al. A fractional-order visual neural model for small target motion detection[J]. Neurocomputing, 2023, 550: 126459.

    Description:
        The FracSTMD model leverages a fractional-order approach to enhance the precision of small target motion detection for low-sampling-frequency. 
        It captures instantaneous luminance change and integrates it with memory information, where the instantaneous information dominates the integrated signal. Due to the rapid response of instantaneous information and the supplement of memory information, the proposed model locates the small moving targets accurately and robustly in low-sampling-frequencies.

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur applied in the retina layer to reduce noise and high-frequency artifacts, enhancing visual clarity for subsequent processing. (Eq. 2)

        Lamina:
            - alpha: Order of Fractional-differnece operator in the lamina. (Eq. 5)
            - delta: Time constant in fractional order operators

        Medulla:
            - n1: Order of the gamma delay function in the Tm1 pathway, which contributes to the temporal filtering in the medulla. (Eq. 10)
            - tau1: Time constant for the gamma delay in Tm1, determining the speed at which temporal integration occurs, optimizing the medulla’s responsiveness to small target motion.

        Lobula:
            - A, B: Amplitude parameter for lateral inhibition in the lobula. (Eq. 14)
            - e, rho, sigma2, sigma3: Parameter controlling the strength of inhibition for lateral interactions. (Eq. 15)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (2)
        # lamina
        'alpha'     : 'self.hLamina.alpha', # Eq. (5)
        'delta'     : 'self.hLamina.delta',
        # medulla
        'n1'    : 'self.hMedulla.hTm1.hGammaDelay.order', # Eq. (10)
        'tau1'      : 'self.hMedulla.hTm1.hGammaDelay.tau',  
        # lobula
        'A'         : 'self.hLobula.hSubInhi.A', # Eq. (14)
        'B'         : 'self.hLobula.hSubInhi.B',
        'e'         : 'self.hLobula.hSubInhi.e', # Eq. (15)
        'rho'       : 'self.hLobula.hSubInhi.rho',
        'sigma2'    : 'self.hLobula.hSubInhi.Sigma1',
        'sigma3'    : 'self.hLobula.hSubInhi.Sigma2',
        }

    def __init__(self, device = 'cpu'):
        """
        FracSTMD Constructor method
        Initializes an instance of the FracSTMD class.
        """
        # Call superclass constructor
        super().__init__(device=device)

        # Customize Lamina and Lobula components
        self.hLamina = fracstmd_core.Lamina(device=device)
        self.hMedulla.hTm1.hGammaDelay.order = 100
        self.hLobula.hSubInhi.e = 1.8

    def init_config(self):
        """ INIT Method

        Initializes the ESTMDBackbone components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()


class DSTMD(BaseModel):
    """ 
    DSTMD: Directional-Small Target Motion Detector 
    
    Ref: 
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.

    Parameters:
        Retina:
            - sigma1: Standard deviation for the Gaussian blur in the retina layer, serving as a pre-filter for noise reduction. (Eq. 1)
        
        Lamina:
            - n1, tau1: Order and time constant of the first gamma bandpass filter in the lamina. (Eq. 4)
            - n2, tau2: Order and time constant of the second gamma bandpass filter. (Eq. 4)
            - sigma2, sigma3: Standard deviations for lateral inhibition in the lamina, helping to suppress non-target background motion. (Eq. 8-9)
            - lambda1, lambda2: Parameters controlling lateral inhibition strength. (Eq. 10-11)
        
        Medulla:
            - n4, tau4: Order and time constant of gamma delay in the Mi1 neuron pathway, enhancing motion sensitivity. (Eq. 25)
            - n5, tau5: Order and time constant of gamma delay in the Tm1 pathway. (Eq. 25)
            - n6, tau6: Order and time constant of gamma delay in another Tm1 pathway variant, allowing selective tuning for target size and direction. (Eq. 25)
        
        Lobula:
            - alpha1: Parameter controlling signal intensity in the lobula, modulating directional selectivity. (Eq. 26)
            - A, B: Lateral inhibition parameters. (Eq. 20)
            - e, rho: Nonlinear parameters for lateral inhibition, affecting signal strength and inhibition shape. (Eq. 21)
            - sigma4, sigma5: Standard deviations for lateral inhibition spatial spread, setting the inhibition area.  (Eq. 21)
            - sigma6, sigma7: Parameters for directionally selective inhibition. (Eq. 29)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (1)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order', # Eq. (4)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        'sigma2'    : 'self.hLamina.hLaminaLateralInhibition.sigma2', # Eq. (8)(9)
        'sigma3'    : 'self.hLamina.hLaminaLateralInhibition.sigma3',
        'lambda1'   : 'self.hLamina.hLaminaLateralInhibition.lambda1', # Eq. (10)(11)
        'lambda2'   : 'self.hLamina.hLaminaLateralInhibition.lambda2',
        # medulla
        'n4'        : 'self.hMedulla.hMi1Para4.hGammaDelay.order', # Eq. (25)
        'tau4'      : 'self.hMedulla.hMi1Para4.hGammaDelay.tau',
        'n5'        : 'self.hMedulla.hTm1Para5.hGammaDelay.order',
        'tau5'      : 'self.hMedulla.hTm1Para5.hGammaDelay.tau',
        'n6'        : 'self.hMedulla.hTm1Para6.hGammaDelay.order',
        'tau6'      : 'self.hMedulla.hTm1Para6.hGammaDelay.tau',
        # lobula
        'alpha1'    : 'self.hLobula.alpha1', # Eq. (26)
        'A'         : 'self.hLobula.hLateralInhi.A', # Eq. (20)
        'B'         : 'self.hLobula.hLateralInhi.B', 
        'e'         : 'self.hLobula.hLateralInhi.e',  # Eq. (21)
        'rho'       : 'self.hLobula.hLateralInhi.rho', 
        'sigma4'    : 'self.hLobula.hLateralInhi.Sigma1', 
        'sigma5'    : 'self.hLobula.hLateralInhi.Sigma2', 
        'sigma6'    : 'self.hLobula.hDirectionInhi.sigma1', # Eq. (29)
        'sigma7'    : 'self.hLobula.hDirectionInhi.sigma2',
        } 
    
    def __init__(self, device = 'cpu'):
        """ DSTMD Constructor method

        Initializes an instance of the DSTMD class.
        """
        # Call superclass constructor
        super().__init__(device=device)

        # Initialize components
        self.hRetina = estmd_core.Retina(device=device)
        self.hLamina = estmd_core.Lamina(device=device)
        self.hMedulla = dstmd_core.Medulla(device=device)
        self.hLobula = dstmd_core.Lobula(device=device)

    def init_config(self):
        """ INIT Method

        Initializes the DSTMD components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        """ MODEL_STRUCTURE Method

        Defines the structure of the DSTMD model.
        """        
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.lobulaOpt, device=self.device)
        self.modelOpt['direction'] = compute_direction(self.lobulaOpt, device=self.device)


class DSTMDBackbone(BaseModel):
    """ DSTMDBackbone: A directional backbone based on DSTMD 
    
    Ref: 
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    """
    
    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        # medulla
        'n4'        : 'self.hMedulla.hMi1Para4.hGammaDelay.order',
        'tau4'      : 'self.hMedulla.hMi1Para4.hGammaDelay.tau',
        'n5'        : 'self.hMedulla.hTm1Para5.hGammaDelay.order',
        'tau5'      : 'self.hMedulla.hTm1Para5.hGammaDelay.tau',
        'n6'        : 'self.hMedulla.hTm1Para6.hGammaDelay.order',
        'tau6'      : 'self.hMedulla.hTm1Para6.hGammaDelay.tau',
        # lobula
        'alpha1'    : 'self.hLobula.alpha1', 
        'A'         : 'self.hLobula.hLateralInhi.A',
        'B'         : 'self.hLobula.hLateralInhi.B', 
        'e'         : 'self.hLobula.hLateralInhi.e', 
        'rho'       : 'self.hLobula.hLateralInhi.rho', 
        'sigma4'    : 'self.hLobula.hLateralInhi.Sigma1', 
        'sigma5'    : 'self.hLobula.hLateralInhi.Sigma2', 
        } 
    
    def __init__(self, device = 'cpu'):
        """ DSTMDBackbone Constructor method

        Initializes an instance of the DSTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__(device=device)

        # Initialize components
        self.hRetina = estmd_core.Retina(device=device)
        self.hLamina = estmd_backbone.Lamina(device=device)
        self.hMedulla = dstmd_core.Medulla(device=device)
        self.hLobula = dstmd_core.Lobula(device=device)

    def init_config(self):
        """ INIT Method

        Initializes the DSTMDBackbone components.
        """
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()


    def model_structure(self, iptMatrix):
        """ MODEL_STRUCTURE Method

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







