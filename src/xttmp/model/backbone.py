from abc import ABC, abstractmethod
import warnings
import logging
import time

import torch

from ..core import estmd_core, estmd_backbone, fracstmd_core, dstmd_core
from ..util.compute_module import compute_response, compute_direction


class BaseModel(ABC, torch.nn.Module):
    """ Base class for Small Target Motion Detector models. """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # here is just an example
            'sigma1': 'retina.gaussian_blur.sigma',
            'sigma2': 'lobula.gaussian_blur.sigma',
        }

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # 检查子类的字典中是否定义了同名方法
        if 'reset' in cls.__dict__:
            raise TypeError(f"禁止重写: 子类 {cls.__name__} 不能覆盖 'reset_buffer' 方法。")
        if 'setup' in cls.__dict__:
            raise TypeError(f"禁止重写: 子类 {cls.__name__} 不能覆盖 'setup' 方法。")
        if 'print_para' in cls.__dict__:
            raise TypeError(f"禁止重写: 子类 {cls.__name__} 不能覆盖 'print_para' 方法。")
        if 'set_para' in cls.__dict__:
            raise TypeError(f"禁止重写: 子类 {cls.__name__} 不能覆盖 'set_para' 方法。")
        
    def __init__(self):
        """ Constructor method.
        """
        super().__init__()
        
        self.retina = None # Handle for the retina layer
        self.lamina = None # Handle for the lamina layer
        self.medulla = None # Handle for the medulla layer
        self.lobula = None # Handle for the lobula layer

        self.input_fps = None

        self.register_buffer('_dummy_device', torch.empty(0)) # Buffer for the correlation output, used for direction computation

        # Model output structure
        self.model_output = {'response': None, 'direction': None}

    def setup(self):
        """
        递归地遍历模型中的所有子模块，
        如果该子模块有 setup 方法，就调用它。
        """
        for module in self.children():
            if hasattr(module, 'setup'):
                module.setup()

    def reset_buffer(self):
        """
        递归地遍历模型中的所有子模块，
        如果该子模块有 reset_buffer 方法，就调用它。
        """
        for module in self.children():
            if hasattr(module, 'reset_buffer'):
                module.reset_buffer()

    @abstractmethod
    def forward(self, img_tensor: torch.Tensor):
        """
        Abstract method for forwarding input through the model.

        Parameters:
            modelIpt: torch.tensor([B, 1, H, W]) Input for model forwarding.
        Returns:
            model_output: Model output structure.
        """
        pass
    
    def process(self, img_tensor: torch.Tensor):
        """ (Old API) Process method for the model.

        This method serves as a wrapper around the forward method

        Parameters:
            img_tensor: torch.Tensor Input tensor for processing.
        Returns:
            model_output: The output from the forward method, potentially after additional processing.
            time_cost: The time taken to process the input, useful for performance evaluation.
        """
        device = self._dummy_device.device
        if device == torch.device('cuda'):
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before starting the timer
        start_time = time.perf_counter()
        model_output = self.forward(img_tensor)
        if device == torch.device('cuda'):
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete before starting the timer
        end_time = time.perf_counter()

        return model_output, end_time - start_time

    def print_para(self) -> None:
        logger = logging.getLogger(__name__)

        para_list = getattr(self, f'_{self.__class__.__name__}__paraMappingList', {})

        if not para_list:
            logger.info(f'The parameters of <{self.__class__.__name__}> is empty.')
            return
        
        msg = f'The parameters of <{self.__class__.__name__}> are:\n'
        for name, paths in para_list.items():
            msg += f'  {name:6}'
            if isinstance(paths, tuple):
                for i, path in enumerate(paths):
                    val = self._get_nested_attr(path)
                    if i == 0:
                        msg += f' --> {path} = {val}\n'
                    elif i == len(paths) - 1:
                        msg += f'{" "*len(name):6} \\---> {path} = {val}\n'
                    else:
                        msg += f'{" "*len(name):6} |---> {path} = {val}\n'
            else:
                val = self._get_nested_attr(paths)
                msg += f' --> {paths} = {val}\n'
                
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
        para_list = getattr(self, f'_{self.__class__.__name__}__paraMappingList', {})
        
        for key, value in kwargs.items():
            if key in para_list.keys():
                paths = para_list[key]
                if isinstance(paths, tuple):
                    for mapped_key in paths:
                        self._set_nested_attr(mapped_key, value)
                else:
                    self._set_nested_attr(paths, value)
            else:
                warnings.warn(f"Private variable '{key}' does not exist.", UserWarning)

    def _get_nested_attr(self, attr_str):
        """ 安全地获取嵌套属性，如 'retina.gaussian_blur.sigma' """
        obj = self
        for attr in attr_str.split('.'):
            obj = getattr(obj, attr)
        return obj

    def _set_nested_attr(self, attr_str, value):
        """ 安全地设置嵌套属性，如将 'retina.gaussian_blur.sigma' 设为 value """
        obj = self
        attrs = attr_str.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
 

class ESTMD(BaseModel):
    """ ESTMD: Elementary small target motion detector

    Ref: 
        * Wiederman S D, Shoemaker P A, O'Carroll D C. A model for the detection of moving targets in visual clutter inspired by insect physiology[J]. PloS one, 2008, 3(7): e2784.
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.

    Remark:
        The implementation and parameters in this code follow Ref [2].

    Parameters:
        Retina:
            - sigma1: Standard deviation for Gaussian blur in the retina, representing visual preforwarding. (Eq. 1)
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
        'sigma1'    : 'retina.sigma', # Eq. (1)
        # lamina
        'n1'        : 'lamina.gamma_BPF.order1', # Eq. (4)
        'tau1'      : 'lamina.gamma_BPF.tau1',
        'n2'        : 'lamina.gamma_BPF.order2',
        'tau2'      : 'lamina.gamma_BPF.tau2',
        'sigma2'    : 'lamina.spatial_inhibition.sigma1', # Eq. (8)(9)
        'sigma3'    : 'lamina.spatial_inhibition.sigma2',
        'lambda1'   : 'lamina.spatial_inhibition.lambda1', # Eq. (10)(11)
        'lambda2'   : 'lamina.spatial_inhibition.lambda2',
        # medulla
        'A'         : ('medulla.tm2.spatial_inhibition.A', 'medulla.tm3.spatial_inhibition.A'), # Eq. (20)
        'B'         : ('medulla.tm2.spatial_inhibition.B', 'medulla.tm3.spatial_inhibition.B'),
        'sigma4'    : ('medulla.tm2.spatial_inhibition.sigma1', 'medulla.tm3.spatial_inhibition.sigma1'), # Eq. (21)
        'sigma5'    : ('medulla.tm2.spatial_inhibition.sigma2', 'medulla.tm3.spatial_inhibition.sigma2'),
        'e'         : ('medulla.tm2.spatial_inhibition.e', 'medulla.tm3.spatial_inhibition.e'),
        'rho'       : ('medulla.tm2.spatial_inhibition.rho', 'medulla.tm3.spatial_inhibition.rho'),
        'n3'        : ('medulla.tm1.order', 'medulla.mi1.order'), # Eq. (24)
        'tau3'      : ('medulla.tm1.tau', 'medulla.mi1.tau')
        } 
     
    def __init__(self):
        # Call the superclass constructor
        super().__init__()
        # Initialize components
        self.retina = estmd_core.Retina()
        self.lamina = estmd_core.Lamina()
        self.medulla = estmd_core.Medulla()
        self.lobula = estmd_core.Lobula()

    def forward(self, x):
        # Define the structure of the ESTMD model
        # forward input matrix through model components
        retina_output = self.retina.forward(x)
        lamina_output = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_output)
        lobula_output = self.lobula.forward(medulla_ON, medulla_OFF)
        # direction not set in the  ESTMD model
        self.model_output['response'] = lobula_output

        return self.model_output


class ESTMDBackbone(BaseModel):
    """ ESTMDBackbone: A backbone based on ESTMD 

    Ref: 
        * Wiederman S D, Shoemaker P A, O'Carroll D C. A model for the detection of moving targets in visual clutter inspired by insect physiology[J]. PloS one, 2008, 3(7): e2784.
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        'sigma1'    : 'retina.sigma',
        'n1'        : 'lamina.order1',
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        'A'         : 'lobula.spatial_inhibition.A',
        'B'         : 'lobula.spatial_inhibition.B',
        'e'         : 'lobula.spatial_inhibition.e',
        'rho'       : 'lobula.spatial_inhibition.rho',
        'sigma4'    : 'lobula.spatial_inhibition.sigma1',
        'sigma5'    : 'lobula.spatial_inhibition.sigma2',
        'order3'    : ('medulla.tm1.order', 'medulla.mi1.order'),
        'tau3'      : ('medulla.tm1.tau', 'medulla.mi1.tau'),
        }
    
    def __init__(self):
        """ ESTMDBackbone Constructor method

        Initializes an instance of the ESTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.retina = estmd_core.Retina()
        self.lamina = estmd_backbone.Lamina()
        self.medulla = estmd_backbone.Medulla()
        self.lobula = estmd_backbone.Lobula()

    def forward(self, img_tensor):
        """ forward Method

        Defines the structure of the ESTMDBackbone model.
        """
        # forward input matrix through model components
        retina_output = self.retina.forward(img_tensor)
        lamina_output = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_output)
        self.lobula_output, _ = self.lobula.forward(medulla_ON, medulla_OFF)

        # Set model response
        self.model_output['response'] = self.lobula_output

        return self.model_output


class FracSTMD(ESTMDBackbone):
    """ FracSTMD: Fractional-order Small Target Motion Detector

    Ref: 
        * Xu M, Wang H, Chen H, et al. A fractional-order visual neural model for small target motion detection[J]. Neurocomputing, 2023, 550: 126459.

    Description:
        The FracSTMD model leverages a fractional-order approach to enhance the precision of small target motion detection for low-sampling-frequency. 
        It captures instantaneous luminance change and integrates it with memory information, where the instantaneous information dominates the integrated signal. Due to the rapid response of instantaneous information and the supplement of memory information, the proposed model locates the small moving targets accurately and robustly in low-sampling-frequencies.

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur applied in the retina layer to reduce noise and high-frequency artifacts, enhancing visual clarity for subsequent forwarding. (Eq. 2)

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
        'sigma1'    : 'retina.sigma', # Eq. (2)
        # lamina
        'alpha'     : 'lamina.alpha', # Eq. (5)
        'delta'     : 'lamina.delta',
        # medulla
        'n1'        : 'medulla.tm1.order', # Eq. (10)
        'tau1'      : 'medulla.tm1.tau',  
        # lobula
        'A'         : 'lobula.spatial_inhibition.A', # Eq. (14)
        'B'         : 'lobula.spatial_inhibition.B',
        'e'         : 'lobula.spatial_inhibition.e', # Eq. (15)
        'rho'       : 'lobula.spatial_inhibition.rho',
        'sigma2'    : 'lobula.spatial_inhibition.sigma1',
        'sigma3'    : 'lobula.spatial_inhibition.sigma2',
        }

    def __init__(self):
        """
        FracSTMD Constructor method
        Initializes an instance of the FracSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Customize Lamina and Lobula components
        self.lamina = fracstmd_core.Lamina()
        self.medulla.tm1.order = 100
        self.lobula.spatial_inhibition.e = 1.8


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
        'sigma1'    : 'retina.sigma', # Eq. (1)
        # lamina
        'n1'        : 'lamina.gamma_BPF.order1', # Eq. (4)
        'tau1'      : 'lamina.gamma_BPF.tau1',
        'n2'        : 'lamina.gamma_BPF.order2',
        'tau2'      : 'lamina.gamma_BPF.tau2',
        'sigma2'    : 'lamina.spatial_inhibition.sigma1', # Eq. (8)(9)
        'sigma3'    : 'lamina.spatial_inhibition.sigma2',
        'lambda1'   : 'lamina.spatial_inhibition.lambda1', # Eq. (10)(11)
        'lambda2'   : 'lamina.spatial_inhibition.lambda2',
        # medulla
        'n4'        : 'medulla.mi1_para4.order', # Eq. (25)
        'tau4'      : 'medulla.mi1_para4.tau',
        'n5'        : 'medulla.tm1_para5.order',
        'tau5'      : 'medulla.tm1_para5.tau',
        'n6'        : 'medulla.tm1_para6.order',
        'tau6'      : 'medulla.tm1_para6.tau',
        # lobula
        'alpha1'    : 'lobula.alpha1', # Eq. (26)
        'A'         : 'lobula.hLateralInhi.A', # Eq. (20)
        'B'         : 'lobula.hLateralInhi.B', 
        'e'         : 'lobula.hLateralInhi.e',  # Eq. (21)
        'rho'       : 'lobula.hLateralInhi.rho', 
        'sigma4'    : 'lobula.hLateralInhi.sigma1', 
        'sigma5'    : 'lobula.hLateralInhi.sigma2', 
        'sigma6'    : 'lobula.hDirectionInhi.sigma1', # Eq. (29)
        'sigma7'    : 'lobula.hDirectionInhi.sigma2',
        } 
    
    def __init__(self):
        """ DSTMD Constructor method

        Initializes an instance of the DSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.retina = estmd_core.Retina()
        self.lamina = estmd_core.Lamina()
        self.medulla = dstmd_core.Medulla()
        self.lobula = dstmd_core.Lobula()


    def forward(self, x):
        """ forward Method

        Defines the structure of the DSTMD model.
        """        
        # forward input matrix through model components
        retina_output = self.retina.forward(x)
        lamina_output = self.lamina.forward(retina_output)
        medulla_tm3_output, medulla_mi1_p4_output, medulla_tm1_p5_output, medulla_tm1_p6_output = \
            self.medulla.forward(lamina_output)
        lobula_output = self.lobula.forward(medulla_tm3_output, medulla_mi1_p4_output, 
                                            medulla_tm1_p5_output, medulla_tm1_p6_output)

        # Compute response and direction
        self.model_output['response'] = compute_response(lobula_output)
        self.model_output['direction'] = compute_direction(lobula_output)

        return self.model_output


class DSTMDBackbone(DSTMD):
    """ DSTMDBackbone: A directional backbone based on DSTMD 
    
    Ref: 
        * Wang H, Peng J, Yue S. A directionally selective small target motion detecting visual neural network in cluttered backgrounds[J]. IEEE transactions on cybernetics, 2018, 50(4): 1541-1555.
    """
    
    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'retina.sigma',
        # lamina
        'n1'        : 'lamina.order1',
        'tau1'      : 'lamina.tau1',
        'n2'        : 'lamina.order2',
        'tau2'      : 'lamina.tau2',
        # medulla
        'n4'        : 'medulla.mi1_para4.order',
        'tau4'      : 'medulla.mi1_para4.tau',
        'n5'        : 'medulla.tm1_para5.order',
        'tau5'      : 'medulla.tm1_para5.tau',
        'n6'        : 'medulla.tm1_para6.order',
        'tau6'      : 'medulla.tm1_para6.tau',
        # lobula
        'alpha1'    : 'lobula.alpha1', 
        'A'         : 'lobula.hLateralInhi.A',
        'B'         : 'lobula.hLateralInhi.B', 
        'e'         : 'lobula.hLateralInhi.e', 
        'rho'       : 'lobula.hLateralInhi.rho', 
        'sigma4'    : 'lobula.hLateralInhi.sigma1', 
        'sigma5'    : 'lobula.hLateralInhi.sigma2', 
        } 
    
    def __init__(self):
        """ DSTMDBackbone Constructor method

        Initializes an instance of the DSTMDBackbone class.
        """
        # Call superclass constructor
        super().__init__()

        # Initialize components
        self.lamina = estmd_backbone.Lamina()







