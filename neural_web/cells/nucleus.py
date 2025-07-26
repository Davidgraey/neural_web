'''
the nucleus holds the POSITIONAL information of the neuron (neural web /
Self-Organizing Map).  Connections move neurons closer together.
'''
from numpy.typing import NDArray
import numpy as np

from organelle import Organelle
from typing import Callable, protocol
from cell_distances import DistanceMetric

import logging
logger = logging.getLogger("neural_web")



class Nucleus(Organelle):
    def __init__(self, position_xyz: NDArray, distance_metric: Callable):
        self.position: NDArray = position_xyz

    def __sub__(self, other: Organelle):
        return Nucleus(self.position - other.position)

    def
