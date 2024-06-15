from .backbone import DSTMDBackbone
from ..core import stmdplus_core, apgstmd_core
from ..util.compute_module import compute_response, compute_direction

class STMDPlus(DSTMDBackbone):
    """
    STMDPlus - This class implements an extended version of the Directional-Small Target Motion Detector (DSTMD).
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()

        # Initialize contrast pathway and mushroom body components
        self.hContrastPathway = stmdplus_core.ContrastPathway()
        self.hMushroomBody = stmdplus_core.MushroomBody()

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 

    def init_config(self):
        """
        Initializes the STMDPlus components.
        """
        super().init_config()

        # Initialize contrast pathway and mushroom body
        self.hContrastPathway.init_config()
        self.hMushroomBody.init_config()

    def model_structure(self, iptMatrix):
        """
        Defines the structure of the STMDPlus model.
        """
        # Import necessary packages
        

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
    """
    ApgSTMD - Attention-Prediction-based Small Target Motion Detector
    """

    def __init__(self):
        """
        Constructor method
        """
        super().__init__()

        # Initialize attention pathway and prediction pathway components
        self.hAttentionPathway = apgstmd_core.AttentionModule()
        self.hPredictionPathway = apgstmd_core.PredictionModule()

        # Bind model parameters and their corresponding parameter pointers.
        self.__parameterList = {} 

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
        """
        Defines the structure of the ApgSTMD model.
        """

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
        # self.predictionOpt is the facilitated STMD output Q(x; y; t; theta) 
        #   in formula (23)
        self.predictionOpt, self.predictionMap = \
            self.hPredictionPathway.process(self.mushroomBodyOpt)

        # Compute response and direction
        self.modelOpt['response'] = compute_response(self.predictionOpt)
        self.modelOpt['direction'] = compute_direction(self.predictionOpt)








