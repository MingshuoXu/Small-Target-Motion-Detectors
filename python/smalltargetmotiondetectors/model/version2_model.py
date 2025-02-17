import numpy as np

from .stmdnet import STMDNet
from ..core import (fstmd_core, fstmdv2_core,
                     stmdplus_core, stmdplusv2_core, apgstmd_core, apgstmdv2_core)
from ..util.compute_module import compute_response


class FSTMDv2(STMDNet):
    """ FSTMDv2: A Feedback Extended FSTMD by inheriting from the Backbonev2.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {
        # retina
        'sigma1'    : 'self.hRetina.hGaussianBlur.sigma', 
        # lamina
        # medulla
        # lobula
        'n'        : 'self.hFeedbackPathway.hGammaDelay.order', # Eq. (4)
        'tau'      : 'self.hFeedbackPathway.hGammaDelay.tau', 
        }

    def __init__(self):
        """ Constructor method """
        super().__init__()
        
        # Initialize components
        self.hLamina = fstmdv2_core.Lamina()
        self.hFeedbackPathway = fstmd_core.FeedbackPathway()
        
        self.maxIteraNum = 10
        self.iterationThres = 1e-3

        # Initialize feedback pathway component
        self.hFeedbackPathway.hGammaDelay.order = 20
        self.hFeedbackPathway.hGammaDelay.tau = 1

    def init_config(self):
        """ Initializes the FSTMD components. """
        super().init_config()
        # Initialize feedback pathway
        self.hFeedbackPathway.init_config()

    def model_structure(self, iptMatrix):
        """
        Defines the structure of the FSTMD model.
        """
        m, n = iptMatrix.shape
        lastFeedback = np.ones((m, n))
        self.feedbackSignal = np.zeros((m, n))

        # Retina layer
        self.retinaOpt = self.hRetina.process(iptMatrix)

        # Feedback loop
        iterationCount = 1
        self.set_loop_state(False)
        while iterationCount < self.maxIteraNum \
            and np.max(np.abs(self.feedbackSignal - lastFeedback)) > self.iterationThres:

            lastFeedback = self.feedbackSignal

            self.laminaOpt = self.hLamina.process(self.retinaOpt + self.feedbackSignal)
            self.hMedulla.process(self.laminaOpt)
            self.medullaOpt = self.hMedulla.Opt

            self.lobulaOpt, direction, correlationOpt\
                 = self.hLobula.process(self.medullaOpt[0],
                                        self.medullaOpt[1],
                                        self.laminaOpt )
            self.feedbackSignal = self.hFeedbackPathway.process(correlationOpt)

            iterationCount += 1
            self.set_loop_state(True)

        # Set model response
        self.modelOpt['response'] = self.lobulaOpt
        self.modelOpt['direction'] = direction

    def set_loop_state(self, state):
        """ Disable circshift for certain components. """
        self.hLamina.isInLoop = state
        self.hFeedbackPathway.hGammaDelay.isInLoop = state


class STMDPlusv2(STMDNet):
    """ STMDPlusv2: A Feedback Extended STMDPlus by inheriting from the Backbonev2.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 

    def __init__(self):
        """ Constructor method """
        super().__init__()

        # Initialize contrast pathway and mushroom body components
        self.hContrastPathway = stmdplus_core.ContrastPathway()
        self.hMushroomBody = stmdplusv2_core.MushroomBody()

    def init_config(self):
        """ Initializes the STMDPlus components. """
        super().init_config()

        # Initialize contrast pathway and mushroom body
        self.hContrastPathway.init_config()
        self.hMushroomBody.init_config()

    def model_structure(self, iptMatrix):
        """ Defines the structure of the STMDPlusv2 model. """
        super().model_structure(iptMatrix)

        # C. Contrast Pathway
        self.direContrastOpt = self.hContrastPathway.process(self.retinaOpt)

        # D. Mushroom Body
        self.mushroomBodyOpt = self.hMushroomBody.process(
            self.lobulaOpt,
            self.direContrastOpt
        )

        # Compute response and direction
        self.modelOpt['response'] = self.mushroomBodyOpt


class ApgSTMDv2(STMDPlusv2):
    """ ApgSTMDv2: A Feedback Extended ApgSTMD by inheriting from the Backbonev2.
    """

    # Bind model parameters and their corresponding parameter pointers.
    __paraMappingList = {} 

    def __init__(self):
        """ Constructor method """
        super().__init__()

        # Initialize attention pathway and prediction pathway components
        self.hAttentionPathway = apgstmd_core.AttentionModule()
        self.hPredictionPathway = apgstmdv2_core.PredictionModule()

        # Set properties of Lobula's SubInhibition module
        self.hLobula.hSubInhi.B = 3.5
        self.hLobula.hSubInhi.Sigma1 = 1.25
        self.hLobula.hSubInhi.Sigma2 = 2.5
        self.hLobula.hSubInhi.e = 1.2
        self.predictionMap = None

    def init_config(self):
        """ Initializes the ApgSTMDv2 components. """
        super().init_config()

        # Initialize attention pathway and prediction pathway components
        self.hAttentionPathway.init_config()
        self.hPredictionPathway.init_config()

    def model_structure(self, iptMatrix):
        """ Defines the structure of the ApgSTMDv2 model. """

        # Preprocessing Module
        self.retinaOpt = self.hRetina.process(iptMatrix)

        # Attention Module
        self.attentionOpt = self.hAttentionPathway.process(
            self.retinaOpt,
            self.predictionMap
        )

        # STMD-based Neural Network
        self.laminaOpt = self.hLamina.process(self.attentionOpt)
        self.hMedulla.process(self.laminaOpt)
        self.medullaOpt = self.hMedulla.Opt

        self.lobulaOpt, self.modelOpt['direction'], _ \
            = self.hLobula.process(self.medullaOpt[0],
                                   self.medullaOpt[1],
                                   self.laminaOpt)

        # STMDPlus
        self.direContrastOpt = self.hContrastPathway.process(self.retinaOpt)
        self.mushroomBodyOpt = self.hMushroomBody.process(
            self.lobulaOpt,
            self.direContrastOpt
        )

        # Prediction Module
        multiDirectoinOpt = self.get_multi_direction_opt(
            self.mushroomBodyOpt,
            self.hLobula.hLPTC.lptcMatrix,
        )
        # self.predictionOpt is the facilitated STMD output Q(x; y; t; theta)
        self.predictionOpt, self.predictionMap = self.hPredictionPathway.process(multiDirectoinOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.predictionOpt)

    @classmethod
    def get_multi_direction_opt(cls, mushroomBodyOpt, direMatrix):
        multiDireOpt = []
        m, n = mushroomBodyOpt.shape

        # Right
        direCoeff = np.zeros((m, n))  
        direCoeff[:, :-1] = direMatrix[:, 1:] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)
        
        # UpRight
        direCoeff = np.zeros((m, n))  
        direCoeff[1:, :-1] = direMatrix[:-1, 1:] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # Up
        direCoeff = np.zeros((m, n))  
        direCoeff[1:, :] = direMatrix[:-1, :] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # UpLeft
        direCoeff = np.zeros((m, n))  
        direCoeff[1:, 1:] = direMatrix[:-1, :-1] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # Left
        direCoeff = np.zeros((m, n))  
        direCoeff[:, 1:] = direMatrix[:, :-1] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # BottomLeft
        direCoeff = np.zeros((m, n))  
        direCoeff[:-1, 1:] = direMatrix[1:, :-1] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # Bottom
        direCoeff = np.zeros((m, n))  
        direCoeff[:-1, :] = direMatrix[1:, :] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        # BottomRight
        direCoeff = np.zeros((m, n))  
        direCoeff[:-1, :-1] = direMatrix[1:, 1:] 
        multiDireOpt.append(mushroomBodyOpt * direCoeff)

        return multiDireOpt











