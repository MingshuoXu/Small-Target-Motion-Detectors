from .math_operator import GammaDelay


class FeedbackPathway(GammaDelay):
    """FeedbackPathway class for the feedback pathway."""

    def __init__(self):
        """Constructor method."""
        # Initializes the FeedbackPathway object
        super().__init__(5, 10)
        
        self.feedback_coefficient = 0.22

    def forward(self, x):
        return self.feedback_coefficient * super().forward(x)
