from .backbone import DSTMDBackbone
from ..core import stmdplus_core, apgstmd_core
from ..util.compute_module import compute_response, compute_direction

class STMDPlus(DSTMDBackbone):
    """ STMDPlus: A facilitated model based on DSTMD with an additional contrast pathway.

    Ref:
        * Wang H, Peng J, Zheng X, et al. "A robust visual system for small target motion detection against cluttered moving backgrounds." IEEE Transactions on Neural Networks and Learning Systems, 2019, 31(3): 839-853.

    Description:
        The STMDPlus model builds upon the DSTMD architecture, enhancing target detection accuracy in cluttered moving backgrounds by introducing a contrast pathway. This pathway provides a complementary processing mechanism to improve robustness against dynamic noise and varying background contrasts. The model parameters align with those specified in the reference.

    Parameters:
        Retina:
            - sigma1: Controls the standard deviation for Gaussian blur applied in the retina layer, reducing noise and emphasizing potential targets. (Eq. 1)
        
        Lamina:
            - n1, tau1: Order and time constant of the first gamma bandpass filter in the lamina. (Eq. 3)
            - n2, tau2: Order and time constant of the second gamma bandpass filter.
        
        Medulla:
            - n3, tau3: Order and time constant of gamma delay in the Mi1 pathway. (Eq. 11)
            - n4, tau4: Order and time constant of gamma delay in the Tm1 pathway. (Eq. 11)
            - n5, tau5: Order and time constant for another gamma delay variant in the Tm1 pathway. (Eq. 11)
        
        Lobula:
            - alpha1: Parameter modulating signal strength for the lobula, contributing to directional selectivity. (Eq. 10)
            - A, B: Lateral inhibition coefficients. (Eq. 13)
            - e, rho, sigma4, sigma5: Lateral inhibition coefficients. (Eq. 14)

        Contrast Pathway:
            - eta: Parameter adjusting the contrast pathway’s influence. (Eq. 15)
            - alpha2: Signal modulation factor within the contrast pathway. (Eq. 17)
    """


    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (1)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',  # Eq. (3)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        # medulla
        'n3'        : 'self.hMedulla.hMi1Para4.hGammaDelay.order',  # Eq. (11)
        'tau3'      : 'self.hMedulla.hMi1Para4.hGammaDelay.tau',
        'n4'        : 'self.hMedulla.hTm1Para5.hGammaDelay.order',
        'tau4'      : 'self.hMedulla.hTm1Para5.hGammaDelay.tau',
        'n5'        : 'self.hMedulla.hTm1Para6.hGammaDelay.order',
        'tau5'      : 'self.hMedulla.hTm1Para6.hGammaDelay.tau',
        # lobula
        'alpha1'    : 'self.hLobula.alpha1',  # Eq. (10)
        'A'         : 'self.hLobula.hLateralInhi.A',  # Eq. (13)
        'B'         : 'self.hLobula.hLateralInhi.B', 
        'e'         : 'self.hLobula.hLateralInhi.e',  # Eq. (14)
        'rho'       : 'self.hLobula.hLateralInhi.rho', 
        'sigma4'    : 'self.hLobula.hLateralInhi.Sigma1', 
        'sigma5'    : 'self.hLobula.hLateralInhi.Sigma2', 
        # Contrast pathway
        'eta'       : 'self.hContrastPathway.eta', # Eq. (15)
        'alpha2'    : 'self.hContrastPathway.alpha2', # Eq. (17)
        } 
    
    def __init__(self):
        """ Constructor method """
        super().__init__()

        # Initialize contrast pathway and mushroom body components
        self.hContrastPathway = stmdplus_core.ContrastPathway()
        self.hMushroomBody = stmdplus_core.MushroomBody()
        
    def init_config(self):
        """
        Initializes the STMDPlus components.
        """
        super().init_config()

        # Initialize contrast pathway and mushroom body
        self.hContrastPathway.init_config()
        self.hMushroomBody.init_config()

    def model_structure(self, iptMatrix):
        """ Defines the structure of the STMDPlus model. """      

        # A. Ommatidia (Retina)
        self.retinaOpt = self.hRetina.process(iptMatrix)

        # B. Motion Pathway (Lamina, Medulla, Lobula)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)

        # C. Contrast Pathway
        self.direContrastOpt = self.hContrastPathway.process(self.retinaOpt)

        # D. Mushroom Body
        self.mushroomBodyOpt = self.hMushroomBody.process(
            self.lobulaOpt, self.direContrastOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.mushroomBodyOpt)
        self.modelOpt['direction'] = compute_direction(self.mushroomBodyOpt)


class ApgSTMD(STMDPlus):
    """ ApgSTMD: Attention-Prediction-guided Small Target Motion Detector

    Ref: 
        * Wang H, Zhao J, Wang H, et al. Attention and prediction-guided motion detection for low-contrast small moving targets[J]. IEEE Transactions on Cybernetics, 2022, 53(10): 6340-6352.

    Description:
        The ApgSTMD model extends the STMDPlus model by introducing attention and prediction pathways to improve target detection.

    Parameters:            
        Retina:
            - sigma1: Standard deviation of the Gaussian blur applied in the retina layer to pre-process input images by smoothing, reducing background noise. (Eq. 2)
        
        Lamina:
            - n1, tau1: Order and time constant of the first gamma bandpass filter in the lamina. (Eq. 6)
            - n2, tau2: Order and time constant of the second gamma bandpass filter. (Eq. 6)
        
        Medulla:
            - n3, tau3: Order and time constant in the Mi1 pathway. (Eq. 14)
            - n4, tau4: Order and time constant in the Tm1 pathway. (Eq. 14)
            - n5, tau5: Additional delay component in Tm1. (Eq. 14)
        
        Lobula:
            - gamma: Signal modulation parameter in the lobula, aiding in selective attention towards targets by enhancing certain spatial patterns. (Eq. 13)
            - A, B: Coefficients for lateral inhibition, reducing background clutter through inhibition of surrounding non-target signals. (Eq. 15)
            - e, rho, sigma4, sigma5: Parameters controlling non-linear inhibition effects. (Eq. 16)
        
        Attention Pathway:
            - eta_list: A list of attentional weights adjusting model responsiveness to spatial regions with potential target information. (Eq. 3)
            - theta_list: (Eq. 3)
        
        Prediction Pathway:
            - zeta, eta: Parameters for prediction kernels. (Eq. 20)
            - kappa: Parameter adjusting the weighting for predicted versus observed signals, balancing real-time data with anticipated target trajectories. (Eq. 23)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (2)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order',  # Eq. (6)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        # medulla
        'n3'        : 'self.hMedulla.hMi1Para4.hGammaDelay.order',  # Eq. (14)
        'tau3'      : 'self.hMedulla.hMi1Para4.hGammaDelay.tau',
        'n4'        : 'self.hMedulla.hTm1Para5.hGammaDelay.order',
        'tau4'      : 'self.hMedulla.hTm1Para5.hGammaDelay.tau',
        'n5'        : 'self.hMedulla.hTm1Para6.hGammaDelay.order',
        'tau5'      : 'self.hMedulla.hTm1Para6.hGammaDelay.tau',
        # lobula
        'gamma'    : 'self.hLobula.alpha1',  # Eq. (13)
        'A'         : 'self.hLobula.hLateralInhi.A',  # Eq. (15)
        'B'         : 'self.hLobula.hLateralInhi.B', 
        'e'         : 'self.hLobula.hLateralInhi.e',  # Eq. (16)
        'rho'       : 'self.hLobula.hLateralInhi.rho', 
        'sigma4'    : 'self.hLobula.hLateralInhi.Sigma1', 
        'sigma5'    : 'self.hLobula.hLateralInhi.Sigma2', 
        # Attention Pathway
        'zeta_list'  : 'self.hAttentionPathway.zeta_list', # Eq. (3)
        'theta_list': 'self.hAttentionPathway.theta_list',
        # Prediction Pathway
        'zeta'      : 'self.hPredictionPathway.zeta', # Eq. (20)
        'eta'       : 'self.hPredictionPathway.eta', # Eq. (20)
        'kappa'     : 'self.hPredictionPathway.kappa', # Eq. (23)
        } 

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()

        # Initialize attention pathway and prediction pathway components
        self.hAttentionPathway = apgstmd_core.AttentionModule()
        self.hPredictionPathway = apgstmd_core.PredictionModule()

        # Set properties of Lobula's LateralInhibition module
        self.hLobula.hLateralInhi.B = 3.5
        self.hLobula.hLateralInhi.Sigma1 = 1.25
        self.hLobula.hLateralInhi.Sigma2 = 2.5
        self.hLobula.hLateralInhi.e = 1.2

        self.predictionMap = None

    def init_config(self):
        """
        Initializes the ApgSTMD components.
        """
        super().init_config()

        # Initialize attention pathway and prediction pathway components
        self.hAttentionPathway.init_config()
        self.hPredictionPathway.init_config()

    def model_structure(self, iptMatrix):
        """ Defines the structure of the ApgSTMD model. """

        # Preprocessing Module
        self.retinaOpt = self.hRetina.process(iptMatrix)

        # Attention Module
        self.attentionOpt = self.hAttentionPathway.process(
            self.retinaOpt, self.predictionMap)

        # STMD-based Neural Network
        self.laminaOpt = self.hLamina.process(self.attentionOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt)

        # STMDPlus
        self.direContrastOpt = self.hContrastPathway.process(self.retinaOpt)
        self.mushroomBodyOpt = self.hMushroomBody.process(
            self.lobulaOpt, self.direContrastOpt)

        # Prediction Module
        #   self.predictionOpt is the facilitated STMD output Q(x; y; t; theta) in Eq. (23)
        self.predictionOpt, self.predictionMap = \
            self.hPredictionPathway.process(self.mushroomBodyOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.predictionOpt)
        self.modelOpt['direction'] = compute_direction(self.predictionOpt)








