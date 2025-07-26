from abc import ABC, abstractmethod
import uuid

import logging
logger = logging.getLogger("neural_web")

class Organelle(ABC):
    @abstractmethod
    def __init__(self):
        self.id = uuid.uuid4()
        pass

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def distance(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass
