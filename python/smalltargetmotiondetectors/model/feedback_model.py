import numpy as np

from .backbone import ESTMDBackbone
from ..core import feedbackstmd_core, fstmd_core, stfeedbackstmd_core

class FeedbackSTMD(ESTMDBackbone):
    """ FeedbackSTMD: Small Target Motion Detector with feedback pathway in lobula

    Ref: 
        * Wang H, Wang H, Zhao J, et al. A time-delay feedback neural network for discriminating small, fast-moving targets in complex dynamic environments[J]. IEEE Transactions on Neural Networks and Learning Systems, 2021, 34(1): 316-330.

    Description:
        The FeedbackSTMD model is an enhancement of the ESTMD backbone with a feedback pathway integrated in the lobula. 

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur in the retina. (Eq. 2)

        Lamina:
            - n1, tau1: Order and time constant for the first gamma bandpass filter in the lamina. (Eq. 3)
            - n2, tau2: Order and time constant for the second gamma filter.

        Medulla:
            - n3, tau3: Tuple containing orders and time constants for Tm1 and Mi1 pathways, both crucial in delaying signals to provide temporal integration for motion detection. (Eq. 7)
        
        Lobula:
            - alpha: Feedback constant in the lobula. (Eq. 9)
            - n4, tau4: Order and time constant in the lobula’s gamma delay, supporting time-delay feedback that strengthens temporal coherence in target tracking.
            - eta: Parameter in the Gaussian kernel. (Eq. 10)
            - A, B: Lateral inhibition coefficients. (Eq. 15)
            - e, rho, sigma2, sigma3: Parameters that modulate nonlinear inhibition. (Eq. 16)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (2)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order', # Eq. (3)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        # medulla
        'n3'        : ('self.hMedulla.hTm1.hGammaDelay.order', 'self.hMedulla.hMi1.hGammaDelay.order'), # Eq. (7)
        'tau3'      : ('self.hMedulla.hTm1.hGammaDelay.tau', 'self.hMedulla.hMi1.hGammaDelay.tau'), 
        # lobula
        'alpha'     : 'self.hLobula.alpha', # Eq. (9)
        'n4'        : 'self.hLobula.hGammaDelay.order',
        'tau4'      : 'self.hLobula.hGammaDelay.tau', 
        'eta'       : 'self.hLobula.paraGaussKernel[\'eta\']', # Eq. (10)
        'A'         : 'self.hLobula.hSubInhi.A', # Eq. (15)
        'B'         : 'self.hLobula.hSubInhi.B',
        'e'         : 'self.hLobula.hSubInhi.e', # Eq. (16)
        'rho'       : 'self.hLobula.hSubInhi.rho',
        'sigma2'    : 'self.hLobula.hSubInhi.Sigma1',
        'sigma3'    : 'self.hLobula.hSubInhi.Sigma2',
        }
    
    def __init__(self):
        """
        FeedbackSTMD Constructor method
        Initializes an instance of the FeedbackSTMD class.
        """
        # Call superclass constructor
        super().__init__()

        # Customize Lobula component
        self.hLobula = feedbackstmd_core.Lobula()
        
        # Customize Lamina's GammaBankPassFilter properties
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.order = 4
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau = 8
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.order = 16
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau = 32

        # Customize Medulla's Tm1 component properties
        self.hMedulla.hTm1.hGammaDelay.order = 9
        self.hMedulla.hTm1.hGammaDelay.tau = 45

    def init_config(self):
        """ INIT Initializes the FeedbackSTMD components. """
        super().init_config()

    def model_structure(self, iptMatrix):
        """ MODEL_STRUCTURE Method

        Defines the structure of the FeedbackSTMD model.
        """
        # Process input matrix through model components
        self.retinaOpt = self.hRetina.process(iptMatrix)
        self.laminaOpt = self.hLamina.process(self.retinaOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt
        self.lobulaOpt = self.hLobula.process(self.medullaOpt[0], self.medullaOpt[1])

        # Set model response
        self.modelOpt['response'] = self.lobulaOpt


class FSTMD(ESTMDBackbone):
    """ FSTMD: Small Target Motion Detector with feedback loop between lobula and lamina

    Ref: 
        * Ling J, Wang H, Xu M, et al. Mathematical study of neural feedback roles in small target motion detection[J]. Frontiers in Neurorobotics, 2022, 16: 984430.

    Description:
        The FSTMD model introduces a feedback loop between the lobula and lamina layers.

    Parameters:
        Retina:
            - sigma1: Standard deviation of the Gaussian blur in the retina, pre-processing the visual input by smoothing noise and high-frequency signals. (Eq. 1)

        Lamina:
            - n1, tau1: Order and time constant for the first gamma bandpass filter in the lamina. (Eq. 6)
            - n2, tau2: Order and time constant for the second gamma filter.

        Medulla:
            - n3, tau3: Tuple containing the order and time constants for the gamma delay functions in Tm1 and Mi1 pathways, which provide temporal integration critical for target motion analysis. (Eq. 9)

        Lobula:
            - e, rho, sigma2, sigma3: Parameter controlling the strength of lateral inhibition in the lobula. (Eq. 13)

        Feedback Pathway:
            - n4, tau4: Order and time constant for the gamma delay in the feedback pathway, adjusting temporal coherence to align with the dynamics of target motion. (Eq. 4)
            - a: Feedback constant, regulating the strength of the feedback signal from the lobula to the lamina, balancing sensitivity and stability in target detection by dynamically adjusting the lamina’s response to target motion. (Eq. 4)
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', # Eq. (1)
        # lamina
        'n1'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.order', # Eq. (6)
        'tau1'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay1.tau',
        'n2'        : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.order',
        'tau2'      : 'self.hLamina.hGammaBandPassFilter.hGammaDelay2.tau',
        # medulla
        'n3'        : ('self.hMedulla.hTm1.hGammaDelay.order', 'self.hMedulla.hMi1.hGammaDelay.order'), # Eq. (9)
        'tau3'      : ('self.hMedulla.hTm1.hGammaDelay.tau', 'self.hMedulla.hMi1.hGammaDelay.tau'), 
        # lobula
        'e'         : 'self.hLobula.hSubInhi.e', # Eq. (13)
        'rho'       : 'self.hLobula.hSubInhi.rho',
        'sigma2'    : 'self.hLobula.hSubInhi.Sigma1',
        'sigma3'    : 'self.hLobula.hSubInhi.Sigma2',
        # feedback pathway
        'n4'        : 'self.hFeedbackPathway.hGammaDelay.order', # Eq. (4)
        'tau4'      : 'self.hFeedbackPathway.hGammaDelay.tau', 
        'a'         : 'self.hFeedbackPathway.feedbackConstant', # Eq. (4)
        }
    
    def __init__(self):
        """ FSTMD Constructor method

        Initializes an instance of the FSTMD class.
        """
        # Call superclass constructor
        super().__init__()
        
        # Initialize feedback pathway component
        self.hFeedbackPathway = fstmd_core.FeedbackPathway()

        self.maxIterationNum = 10
        self.iterationThreshold = 1e-3

        # Customize Medulla's Tm1 component properties
        self.hMedulla.hTm1.hGammaDelay.order = 5

    def init_config(self):
        """ INIT Method

        Initializes the FSTMD components.
        """
        # Call superclass init method
        super().init_config()

        # Initialize feedback pathway
        self.hFeedbackPathway.init_config()

    def model_structure(self, iptMatrix):
        """ MODEL_STRUCTURE Method

        Defines the structure of the FSTMD model.
        """
        m, n = iptMatrix.shape
        lastFeedbackSignal = np.ones((m, n))
        self.feedbackSignal = np.zeros((m, n))

        # Retina layer
        self.retinaOpt = self.hRetina.process(iptMatrix)

        # Feedback loop
        iterationCount = 1
        self.set_loop_state(False)
        while iterationCount < self.maxIterationNum and np.max(
                np.abs(self.feedbackSignal - lastFeedbackSignal)) > self.iterationThreshold:
            lastFeedbackSignal = self.feedbackSignal.copy()

            # Execute feedback loop
            self.laminaOpt = self.hLamina.process(self.retinaOpt + self.feedbackSignal)
            self.hMedulla.process(self.laminaOpt)
            self.medullaOpt = self.hMedulla.Opt
            self.lobulaOpt, correlationOpt = self.hLobula.process(self.medullaOpt)
            self.feedbackSignal = self.hFeedbackPathway.process(correlationOpt)

            iterationCount += 1
            self.set_loop_state(True)

        # Set model response
        self.modelOpt['response'] = self.lobulaOpt

    def set_loop_state(self, state):
        """ Sets the loop state of certain components. """
        # Disable circshift for certain components
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.isInLoop = state
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.isInLoop = state
        self.hMedulla.hTm1.hGammaDelay.isInLoop = state
        self.hFeedbackPathway.hGammaDelay.isInLoop = state


class STFeedbackSTMD(ESTMDBackbone):
    """ STFeedbackSTMD - Small Target Motion Detection With Spatio-Temporal Feedback

    # !!! Notice: This model has not been reproduced yet
    
    Ref: 
        [1] Wang H, Zhong Z, Lei F, et al. Bio-Inspired Small Target Motion Detection With Spatio-Temporal Feedback in Natural Scenes[J]. IEEE Transactions on Image Processing, 2023.

    
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()       

        # Customize Medulla and Lobula component
        self.hMedulla = stfeedbackstmd_core.Medulla()
        self.hLobula = stfeedbackstmd_core.Lobula()

    def init_config(self):
        """
        Initializes the STFeedbackSTMD components.
        """
        super().init_config()










