from abc import ABC, abstractmethod

import torch


class BaseCore(ABC, torch.nn.Module):
    """
    Abstract base class for core processing components.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()

        self.output = None

    def setup(self, *args, **kwargs):
        """
        Abstract method for initialization.
        """
        pass

    def reset(self):
        """
        Abstract method for resetting the state.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Abstract method for processing.
        """
        pass
