"""
docstring for file
__package__
"""
# type: ignore

import time
from enum import Enum, auto
from typing import Protocol, Tuple, Dict, List, Optional, Union, Iterable, Callable
from typing_extensions import TypedDict

from dataclasses import dataclass, field
from abc import ABC, ABCMeta

from functools import partial, reduce, lru_cache, wraps

import matplotlib.pyplot as plt
from matplotlib import colormaps

# ======= project specifics =======
import numpy as np
from numpy.typing import ArrayLike, NDArray



RNG = np.random.Gendrator(seed=1)

if __name__ == "__main__":
    print('hello')
