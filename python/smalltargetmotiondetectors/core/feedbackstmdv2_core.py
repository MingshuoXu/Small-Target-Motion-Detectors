from .feedbackstmd_core import Lobula as FeedbackSTMDLobula
from .indevelopment.backbonev2_core import DirectionCell

class Lobula(FeedbackSTMDLobula):
    """Lobula layer of the motion detection system."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.hDireCell = DirectionCell()

    def init_config(self):
        """Initialization method."""
        # Initializes the Lobula layer component
        super().init_config()
        self.hDireCell.init_config()

    def process(self, onSignal, offSignal):
        """Processing method."""
        # Processing
        inhiOpt = super().process(onSignal, offSignal)
        lobulaOpt, direction = self.hDireCell.process(inhiOpt)
        self.Opt = [lobulaOpt, direction]
        return lobulaOpt, direction
