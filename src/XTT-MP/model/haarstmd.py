import warnings

from ..core import haarstmd_core
from .backbone import ESTMDBackbone

class HaarSTMD(ESTMDBackbone):
    ''' HaarSTMD: Advancing small target motion detection in dim light
    
    Ref: 
        [1] Chen H, Sun X, Hu C, et al. Unveiling the power of Haar frequency domain: Advancing small target motion detection in dim light[J]. Applied Soft Computing, 2024, 167: 112281.
    '''

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
            'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
            'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
            'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
            'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
            'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
            'sigma2'    : 'self.hLobula.hSubInhi.Sigma1',
            'sigma3'    : 'self.hLobula.hSubInhi.Sigma2',
            'TAU'       : 'self.hLobula.tau',
            }
        
    def __init__(self, device = 'cpu'):
        ''' Constructor method '''
        super().__init__(device=device)

        self.hMedulla = haarstmd_core.Medulla()
        self.hLobula = haarstmd_core.Lobula()

        # init parameter
        self.hRetina.hGaussianBlur.sigma = 1
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.order = 10
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau = 3
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.order = 10
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau = 9
        self.hLobula.hSubInhi.Sigma1 = 1.5
        self.hLobula.hSubInhi.Sigma2 = 3
        self.hLobula.tau = 1      
        # a       = 34,
        # theta   = math.pi,
        # r       = 4,

    def init_config(self):                                  
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

        if self.device != 'cpu':
            self.device = 'cpu'
            warnings.warn('Currently, only CPU is supported. The device parameter will be ignored.', UserWarning)

    def model_structure(self, iptMatrix):
        ''' MODEL_STRUCTURE: Defines the structure of the HaarSTMD model. 

        Input:
            iptMatrix - Input matrix for processing

        '''
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        cellSpatialOpt, temporalOpt = self.hMedulla.process(self.laminaOpt)
        self.lobulaOpt = self.hLobula.process(cellSpatialOpt, temporalOpt)

        self.modelOpt['response'] = self.lobulaOpt