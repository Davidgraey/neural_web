from abc import ABC, abstractmethod

class Organelle(ABC):
    @abstractmethod
    def __init__(self, positional):
        self.positional = positional

    @abstractmethod
    def __lt__(self, other):
        pass

    @abstractmethod
    def distance(self, other):
        pass

    @abstractmethod
    def __str__(self):
        pass
