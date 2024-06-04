from . import feedbackstmd_core
from . import backbonev2_core

class Lobula(feedbackstmd_core.Lobula):
    """Lobula layer of the motion detection system."""

    def __init__(self):
        """Constructor method."""
        # Initializes the Lobula object
        super().__init__()
        self.hDireCell = backbonev2_core.LPTangentialCell()

    def init_config(self):
        """Initialization method."""
        # Initializes the Lobula layer component
        super().init_config()
        self.hDireCell.init_config()

    def process(self, onSignal, offSignal, laminaOpt):
        """Processing method."""
        # Processing
        lobulaOpt = super().process(onSignal, offSignal)
        direction = self.hDireCell.process(laminaOpt, onSignal, offSignal)
        self.Opt = [lobulaOpt, direction]
        return lobulaOpt, direction
