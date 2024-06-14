import numpy as np

from .backbone import ESTMDBackbone
from ..core import feedbackstmd_core, fstmd_core, stfeedbackstmd_core

class FeedbackSTMD(ESTMDBackbone):
    """
    FeedbackSTMD - Feedback Small Target Motion Detector
    This class implements a Feedback Extended Small Target Motion Detector
    by inheriting from the ESTMDBackbone class.
    """

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

        self._map_and_init_parameter()

    def _map_and_init_parameter(self, **kwargs):
        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 
        
        # init parameters
        self.set_parameter(**kwargs)

    def init_config(self):
        """
        INIT Initializes the FeedbackSTMD components.
        """
        super().init_config()

    def model_structure(self, iptMatrix):
        """
        MODEL_STRUCTURE Method
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
    """
    FSTMD - Feedback Small Target Motion Detector
    This class implements a Feedback Small Target Motion Detector by
    inheriting from the ESTMDBackbone class.
    """

    def __init__(self):
        """
        FSTMD Constructor method
        Initializes an instance of the FSTMD class.
        """
        # Call superclass constructor
        super().__init__()
        self.maxIterationNum = 10
        self.iterationThreshold = 1e-3

        # Initialize feedback pathway component
        self.hFeedbackPathway = fstmd_core.FeedbackPathway()

        # Customize Medulla's Tm1 component properties
        self.hMedulla.hTm1.hGammaDelay.order = 5

        self._map_and_init_parameter()

    def _map_and_init_parameter(self, **kwargs):
        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 
        
        # init parameters
        self.set_parameter(**kwargs)

    def init_config(self):
        """
        INIT Method
        Initializes the FSTMD components.
        """
        # Call superclass init method
        super().init_config()

        # Initialize feedback pathway
        self.hFeedbackPathway.init_config()

    def model_structure(self, iptMatrix):
        """
        MODEL_STRUCTURE Method
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
        """
        Sets the loop state of certain components.
        """
        # Disable circshift for certain components
        self.hLamina.hGammaBandPassFilter.hGammaDelay1.isInLoop = state
        self.hLamina.hGammaBandPassFilter.hGammaDelay2.isInLoop = state
        self.hMedulla.hTm1.hGammaDelay.isInLoop = state
        self.hFeedbackPathway.hGammaDelay.isInLoop = state


class STFeedbackSTMD(ESTMDBackbone):
    """
    FeedbackSTMD - Feedback Small Target Motion Detector
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()       

        # Customize Medulla and Lobula component
        self.hMedulla = stfeedbackstmd_core.Medulla()
        self.hLobula = stfeedbackstmd_core.Lobula()

        self._map_and_init_parameter()

    def _map_and_init_parameter(self, **kwargs):
        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 
        
        # init parameters
        self.set_parameter(**kwargs)

    def init_config(self):
        """
        Initializes the STFeedbackSTMD components.
        """
        super().init_config()










