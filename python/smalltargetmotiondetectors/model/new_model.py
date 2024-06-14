import math

from ..core import haarstmd_core
from .backbone import ESTMDBackbone


class HaarSTMD(ESTMDBackbone):
    def __init__(self):
        super().__init__()
        self.hMedulla = haarstmd_core.Medulla()
        self.hLobula = haarstmd_core.Lobula()
        
        self.__sigma1 = 1
        self.__n1 = 10
        self.__tau1 = 3
        self.__n2 = 10
        self.__tau2 = 9
        self.__a = 34
        self.__theta = math.pi
        self.__r = 4
        self.__sigma2 = 1.5
        self.__sigma3 = 3
        self.__TAU = 1

        self.__sigma1 = {'pointer': self.hRetina.hGaussianBlur.sigma}


    def init_config(self):                                  
        self.hRetina.hGaussianBlur.sigma = self.__sigma1
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.order = self.__n1
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau = self.__tau1
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.order = self.__n2
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau = self.__tau2

        self.hLobula.hSubInhi.Sigma1 = self.__sigma2
        self.hLobula.hSubInhi.Sigma2 = self.__sigma3
        self.hLobula.tau = self.__TAU

        self.hRetina.init_config()
        self.hLamina.init_config()
        self.hMedulla.init_config()
        self.hLobula.init_config()

    def model_structure(self, iptMatrix):
        ''' 
        MODEL_STRUCTURE Method:
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