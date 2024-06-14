import math

from ..core import haarstmd_core
from .backbone import ESTMDBackbone


class HaarSTMD(ESTMDBackbone):
    def __init__(self):
        super().__init__()

        self.hMedulla = haarstmd_core.Medulla()
        self.hLobula = haarstmd_core.Lobula()

        self._map_and_init_parameter(sigma1 = 10,
                                    n1      = 10,
                                    tau1    = 3,
                                    n2      = 10,
                                    tau2    = 9,
                                    sigma2  = 1.5,
                                    sigma3  = 3,
                                    TAU     = 1
                                    )
        # a       = 34,
        # theta   = math.pi,
        # r       = 4,

    def _map_and_init_parameter(self, **kwargs):
        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {
            'sigma1'    : 'self.hRetina.hGaussianBlur.sigma',
            'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',
            'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
            'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
            'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
            'sigma2'    : 'self.hLobula.hSubInhi.Sigma1',
            'sigma3'    : 'self.hLobula.hSubInhi.Sigma2',
            'TAU'       : 'self.hLobula.tau',
            }
        
        # init parameters
        self.set_parameter(**kwargs)
        

    def init_config(self):                                  
        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        ''' MODEL_STRUCTURE:
            Defines the structure of the HaarSTMD model. Processes the input matrix 
            through the HaarSTMD model components (retina, lamina, medulla, and lobula)
            and generates the model's response.

        Input:
            iptMatrix - Input matrix for processing

        '''
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        cellSpatialOpt, temporalOpt = self.hMedulla.process(self.laminaOpt)
        self.lobulaOpt = self.hLobula.process(cellSpatialOpt, temporalOpt)

        self.modelOpt['response'] = self.lobulaOpt