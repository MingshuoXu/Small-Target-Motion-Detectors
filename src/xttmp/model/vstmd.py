import numpy as np
import torch
import torch.nn.functional as F
import scipy.ndimage

from ..core import estmd_core, vstmd_core
from ..util.compute_module import AreaNMS
from .backbone import BaseModel
from copy import deepcopy




class vSTMD(BaseModel):
    """ STMDNet: A Lightweight Directional Framework for Motion Pattern Recognition of Tiny Targets.
    
    Ref:
        * Xu M, Luan H, Hao Z D, et al. STMDNet: A Lightweight Directional Framework for Motion Pattern Recognition of Tiny Targets[J]. arXiv preprint arXiv:2501.13054, 2025.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1': 'retina.sigma',
        # lamina
        'alpha' : 'lamina.alpha',
        'delta' : 'lamina.delta',
        # medulla
        'g_leak' : ('medulla.on_pathway.g_leak', 'medulla.off_pathway.g_leak'),
        'v_rest' : ('medulla.on_pathway.v_rest', 'medulla.off_pathway.v_rest'),
        'vEx'   : ('medulla.on_pathway.v_exci', 'medulla.off_pathway.v_exci'),
        # lobula
        'A'     : 'lobula.spatial_inhibition.A', 
        'B'     : 'lobula.spatial_inhibition.B', 
        'e'     : 'lobula.spatial_inhibition.e', 
        'rho'   : 'lobula.spatial_inhibition.rho', 
        'sigma2': 'lobula.spatial_inhibition.sigma1', 
        'sigma3': 'lobula.spatial_inhibition.sigma2', 
        } 
    
    def __init__(self): 
        """ Constructor function """
        super().__init__()

        # Initialize components
        self.retina = estmd_core.Retina()
        self.lamina = vstmd_core.Lamina()
        self.medulla = vstmd_core.Medulla()
        self.lobula = vstmd_core.Lobula()

        self.lamina.alpha = 0.25
        self.medulla.on_pathway.g_leak = 0.35
        self.medulla.off_pathway.g_leak = 0.35

    def forward(self, modelIpt):
        """ Define the structure of the model """
        retina_output = self.retina.forward(modelIpt)
        lamina_ON, lamina_OFF = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)
        self.model_output['response'], self.model_output['direction'] \
            = self.lobula.forward(medulla_ON, medulla_OFF, lamina_ON, lamina_OFF)    

        return self.model_output
    
        
class vSTMD_F(vSTMD):
    """ vSTMD_F: vSTMD with Feedback Mechanism.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = deepcopy(vSTMD._vSTMD__paraMappingList)
    __paraMappingList.update(
        {'beta'     : 'lobula.beta',
         'sigma_4'  : 'lobula.sigma', })
    
    def __init__(self):
        """
        STMDNetF Constructor method
        Initializes an instance of the FeedbackSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Lobula with feedback mechanism
        self.lobula = vstmd_core.Lobula_with_Feedback()


class vSTMD_L(vSTMD):
    """ vSTMD_L: vSTMD Location Module.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = deepcopy(vSTMD._vSTMD__paraMappingList)
    

    def forward(self, x):
        """ DaC: Dymanics and Correlate """
        # denoise by Gaussian filter
        retina_output = self.retina.forward(x)
        # temporal difference and signal separation
        lamina_ON, lamina_OFF = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)

        # location only
        self.correlation_output = medulla_ON * medulla_OFF
        self.model_output['response'] = self.lobula.spatial_inhibition.forward(self.correlation_output)

        return self.model_output


class vSTMD_F_L(vSTMD_F):
    """ vSTMD_L: vSTMD Location Module.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = deepcopy(vSTMD_F._vSTMD_F__paraMappingList)
    
    def forward(self, x):
        """ DaC: Dymanics and Correlate """
        # denoise by Gaussian filter
        retina_output = self.retina.forward(x)
        # temporal difference and signal separation
        lamina_ON, lamina_OFF = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)

        # location only
        self.correlation_output = medulla_ON * medulla_OFF
        self.model_output['response'] = self.lobula.spatial_inhibition.forward(self.correlation_output)

        return self.model_output


class vSTMD_M(vSTMD_L):
    __paraMappingList = deepcopy(vSTMD_L._vSTMD_L__paraMappingList)

    def __init__(self): 
        """ Constructor function """
        super().__init__()
        self.direction_computer = vstmd_core.FastEuclideanTracker()
        self.torch_nms = AreaNMS(radio=8)

    def get_direction_by_matching(self, model_response):
        device = model_response.device

        direction_output = torch.full_like(model_response, float('nan'), device=device)            

        responses = torch.argwhere(model_response > 0)
        if len(responses) == 0:
            return direction_output
        
        tracks = self.direction_computer.update(responses)
            
        if len(tracks) > 0:
            tracks_t = torch.as_tensor(tracks, device=device)
            dim_0 = tracks_t[:, 0].long()
            dim_1 = tracks_t[:, 1].long()
            directions = tracks_t[:, 2].float()
            
            # 3. 向量化赋值 (高级索引)
            direction_output[0, 0, dim_0, dim_1] = directions

        return direction_output
    
    def forward(self, modelIpt):
        """ forward Method: Defines the structure of the vSTMD_F model. """
        super().forward(modelIpt)

        response = self.torch_nms(self.model_output['response'])
        self.model_output['direction'] = self.get_direction_by_matching(response)

        return self.model_output


class vSTMD_F_M(vSTMD_F_L):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = deepcopy(vSTMD_L._vSTMD_L__paraMappingList)
 

    def __init__(self): 
        """ Constructor function """
        super().__init__()
        self.direction_computer = vstmd_core.FastEuclideanTracker()

        self.torch_nms = AreaNMS(radio=8)

    def get_direction_by_matching(self, model_response):
        device = model_response.device

        direction_output = torch.full_like(model_response, float('nan'), device=device)            

        responses = torch.argwhere(model_response > 0)
        if len(responses) == 0:
            return direction_output
        
        tracks = self.direction_computer.update(responses)
            
        if len(tracks) > 0:
            tracks_t = torch.as_tensor(tracks, device=device)
            dim_0 = tracks_t[:, 0].long()
            dim_1 = tracks_t[:, 1].long()
            directions = tracks_t[:, 2].float()
            
            # 3. 向量化赋值 (高级索引)
            direction_output[0, 0, dim_0, dim_1] = directions

        return direction_output

    def forward(self, modelIpt):
        """ forward Method: Defines the structure of the vSTMD_F model. """
        super().forward(modelIpt)

        response = self.torch_nms(self.model_output['response'])
        self.model_output['direction'] = self.get_direction_by_matching(response)

        return self.model_output



 

# ablation
class vSTMD_without_GF(vSTMD):
    
    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1': 'retina.sigma',
        # lamina
        'alpha' : 'lamina.alpha',
        'delta' : 'lamina.delta',
        # medulla
        'g_leak' : ('medulla.on_pathway.g_leak', 'medulla.off_pathway.g_leak'),
        'v_rest' : ('medulla.on_pathway.v_rest', 'medulla.off_pathway.v_rest'),
        'vEx'   : ('medulla.on_pathway.v_exci', 'medulla.off_pathway.v_exci'),
        # lobula
        'A'     : 'lobula.spatial_inhibition.A', 
        'B'     : 'lobula.spatial_inhibition.B', 
        'e'     : 'lobula.spatial_inhibition.e', 
        'rho'   : 'lobula.spatial_inhibition.rho', 
        'sigma2': 'lobula.spatial_inhibition.sigma1', 
        'sigma3': 'lobula.spatial_inhibition.sigma2', 
        } 
    
    def forward(self, x):
        """ Define the structure of the model """
        lamina_ON, lamina_OFF = self.lamina.forward(x)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)
        self.model_output['response'], self.model_output['direction'] \
            = self.lobula.forward(medulla_ON, medulla_OFF, lamina_ON, lamina_OFF)
        
        return self.model_output


class vSTMD_without_cIDP(vSTMD):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1': 'retina.sigma',
        # lamina
        'alpha' : 'lamina.alpha',
        'delta' : 'lamina.delta',
        # medulla
        'g_leak' : ('medulla.on_pathway.g_leak', 'medulla.off_pathway.g_leak'),
        'v_rest' : ('medulla.on_pathway.v_rest', 'medulla.off_pathway.v_rest'),
        'vEx'   : ('medulla.on_pathway.v_exci', 'medulla.off_pathway.v_exci'),
        # lobula
        'A'     : 'lobula.spatial_inhibition.A', 
        'B'     : 'lobula.spatial_inhibition.B', 
        'e'     : 'lobula.spatial_inhibition.e', 
        'rho'   : 'lobula.spatial_inhibition.rho', 
        'sigma2': 'lobula.spatial_inhibition.sigma1', 
        'sigma3': 'lobula.spatial_inhibition.sigma2', 
        } 
    
    def __init__(self): 
        from ..core.math_operator import GammaDelay
        """ Constructor function """
        super().__init__()

        # Initialize components
        self.gamma_delay = GammaDelay(12, 25) 

    def forward(self, x):
        """ Define the structure of the model """
        retina_output = self.retina.forward(x)
        lamina_ON, lamina_OFF = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)
        delayed_medulla_OFF = self.gamma_delay.forward(medulla_OFF)
        self.model_output['response'], self.model_output['direction'] \
            = self.lobula.forward(medulla_ON, delayed_medulla_OFF, lamina_ON, lamina_OFF)
        
        return self.model_output


class vSTMD_without_CDGC(BaseModel):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 
    
    def __init__(self): 
        from ..model.backbone import DSTMD
        
        super().__init__()
        self.location_part = vSTMD()
        self.direction_part = DSTMD()

    def forward(self, modelIpt):
        """ Define the structure of the model """
        self.location_part.forward(modelIpt)
        self.model_output['response'] = self.location_part.model_output['response']

        self.direction_part.forward(modelIpt)
        _direction = self.direction_part.model_output['direction']
        self.model_output['direction'] = self.match_direction(self.model_output['response'], _direction)

        return self.model_output

    @staticmethod
    def match_direction(response, direction):
        """
        Fill NaN direction values for positive response locations with the nearest non-NaN direction value.
        Uses matrix operations for acceleration.
        """
        mask_nan = torch.isnan(direction)
        mask_pos = response > 0

        # Only fill where response > 0 and direction is nan
        fill_mask = mask_nan & mask_pos

        # Create a mask of valid direction locations
        valid_mask = ~mask_nan

        # Replace NaNs with 0 for distance calculation
        direction_filled = torch.where(valid_mask, direction, 0)


        # 1. 提取布尔掩码，放回 CPU 并转为 NumPy 数组
        # (~valid_mask) 等价于 valid_mask == 0，即 NaN 的地方为 True (需要被填充的区域)
        target_mask_np = (~valid_mask).cpu().numpy()

        # 2. 调用 SciPy 计算距离变换和索引
        distance_np, indices_np = scipy.ndimage.distance_transform_edt(
            target_mask_np, return_indices=True
        )

        # 3. 将 NumPy 格式的索引转回 PyTorch，并放到原来的设备 (GPU) 上
        indices = torch.from_numpy(indices_np).to(device=direction.device, dtype=torch.long)

        # Get nearest valid direction for each pixel
        nearest_direction = direction_filled[tuple(indices)]

        # Prepare output
        directionOpt = direction.clone()

        # Fill only where fill_mask is True
        directionOpt[fill_mask] = nearest_direction[fill_mask]

        return directionOpt
        
        
class vSTMD_F_without_GF(vSTMD_F):
    
    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1': 'retina.sigma',
        # lamina
        'alpha' : 'lamina.alpha',
        'delta' : 'lamina.delta',
        # medulla
        'g_leak' : ('medulla.on_pathway.g_leak', 'medulla.off_pathway.g_leak'),
        'v_rest' : ('medulla.on_pathway.v_rest', 'medulla.off_pathway.v_rest'),
        'vEx'   : ('medulla.on_pathway.v_exci', 'medulla.off_pathway.v_exci'),
        # lobula
        'A'     : 'lobula.spatial_inhibition.A', 
        'B'     : 'lobula.spatial_inhibition.B', 
        'e'     : 'lobula.spatial_inhibition.e', 
        'rho'   : 'lobula.spatial_inhibition.rho', 
        'sigma2': 'lobula.spatial_inhibition.sigma1', 
        'sigma3': 'lobula.spatial_inhibition.sigma2', 
        } 
    
    def forward(self, x):
        """ Define the structure of the model """
        lamina_ON, lamina_OFF = self.lamina.forward(x)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)
        self.model_output['response'], self.model_output['direction'] \
            = self.lobula.forward(medulla_ON, medulla_OFF, lamina_ON, lamina_OFF)
        
        return self.model_output


class vSTMD_F_without_cIDP(vSTMD_F):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = { 
        # retina
        'sigma1': 'retina.sigma',
        # lamina
        'alpha' : 'lamina.alpha',
        'delta' : 'lamina.delta',
        # medulla
        'g_leak' : ('medulla.on_pathway.g_leak', 'medulla.off_pathway.g_leak'),
        'v_rest' : ('medulla.on_pathway.v_rest', 'medulla.off_pathway.v_rest'),
        'vEx'   : ('medulla.on_pathway.v_exci', 'medulla.off_pathway.v_exci'),
        # lobula
        'A'     : 'lobula.spatial_inhibition.A', 
        'B'     : 'lobula.spatial_inhibition.B', 
        'e'     : 'lobula.spatial_inhibition.e', 
        'rho'   : 'lobula.spatial_inhibition.rho', 
        'sigma2': 'lobula.spatial_inhibition.sigma1', 
        'sigma3': 'lobula.spatial_inhibition.sigma2', 
        } 
    
    def __init__(self): 
        from ..core.math_operator import GammaDelay
        """ Constructor function """
        super().__init__()

        self.gamma_delay = GammaDelay(12, 25) 

    def forward(self, x):
        """ Define the structure of the model """
        retina_output = self.retina.forward(x)
        lamina_ON, lamina_OFF = self.lamina.forward(retina_output)
        medulla_ON, medulla_OFF = self.medulla.forward(lamina_ON, lamina_OFF)
        delayed_medulla_OFF = self.gamma_delay.forward(medulla_OFF)
        self.model_output['response'], self.model_output['direction'] \
            = self.lobula.forward(medulla_ON, delayed_medulla_OFF, lamina_ON, lamina_OFF)
        
        return self.model_output


class vSTMD_F_without_CDGC(BaseModel):

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 
    
    def __init__(self): 
        from ..model.backbone import DSTMD
        
        super().__init__()
        self.location_part = vSTMD_F()
        self.direction_part = DSTMD()

    def forward(self, modelIpt):
        """ Define the structure of the model """
        self.location_part.forward(modelIpt)
        self.model_output['response'] = self.location_part.model_output['response']

        self.direction_part.forward(modelIpt)
        _direction = self.direction_part.model_output['direction']
        self.model_output['direction'] = vSTMD_without_CDGC.match_direction(self.model_output['response'], _direction)

        return self.model_output
