from base_core import BaseCore
from .math_operator import GammaDelay

class FeedbackPathway(BaseCore):
    """FeedbackPathway class for the feedback pathway."""

    def __init__(self):
        """Constructor method."""
        # Initializes the FeedbackPathway object
        super().__init__()
        
        
        self.hGammaDelay = GammaDelay(5, 10)
        self.feedbackConstant = 0.22

    def init_config(self):
        """Initialization method."""
        # Initializes the GammaDelay object
        self.hGammaDelay.init_config()

    def process(self, feedbackIpt):
        """Processing method."""
        # Processes the feedbackIpt to generate the feedback output
        feedbackOpt = self.feedbackConstant * self.hGammaDelay.process(feedbackIpt)
        self.Opt = feedbackOpt
        return feedbackOpt
