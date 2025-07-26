'''
The Dendrite is the OUTGOING weighting mechanism - it also holds the
connections to other neurons
each dendrtic connection has a spike_membrane, which is the outgoing
potential energy to "fire"
'''
import logging
logger = logging.getLogger("neural_web")
from organelle import Organelle

class Dendrite(Organelle):
    pass
