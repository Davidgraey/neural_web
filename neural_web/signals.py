'''
Covering the signal trace - path of the "neural signal"
Charts the IDS of activated neurons
'''
import uuid
from dataclasses import dataclass


class Signal:
    __slots__ = 'timestep', 'id_trace', 'index-trace'
    def __init__(self, timestep: int):
        timestep: int = timestep
        id_trace: list[list[str]] = []
        index_trace: list[list[int]] = []

    def append(self, active_ids: list[str], active_indices: list[int]):
        """
        add a "layer" at a time to the forward pass - track ids and positions.
        Parameters
        ----------
        active_ids :
        active_indices :

        Returns
        -------

        """

        # TODO: type check
        self.id_trace.append(active_ids)
        # TODO: type check
        self.index_trace.append(active_indices)
