from abc import ABC, abstractmethod

class BaseCore(ABC):
    """
    Abstract base class for core processing components.
    """

    def __init__(self, device ='cpu'):
        """
        Constructor.
        """
        self.Opt = None
        self.device = device

    @abstractmethod
    def init_config(self, *args, **kwargs):
        """
        Abstract method for initialization.
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        """
        Abstract method for processing.
        """
        pass
