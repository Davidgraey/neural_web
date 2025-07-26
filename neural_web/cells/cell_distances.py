'''
intercellular distances
'''
import logging
from scipy.spatial.distance import cdist, pdist
from numpy.typing import NDArray
from enum import Enum, member, nonmember
from typing import Callable, Protocol, Optional

from cell_errors import DimensionError

logger = logging.getLogger("neural_web")


def match_shapes(position_1: NDArray,
                 position_2: Optional[NDArray],
                 last_dim_only: bool
                 ) -> bool:
    if last_dim_only:
        try:
            position_1.shape[-1] == position_2.shape[-1]
        except DimensionError as e:
            logger.error(e)
            return False
    else:
        try:
            position_1.shape == position_2.shape
        except DimensionError as e:
            logger.error(e)
            return False
    return True


def calculate_3d_manhattan_distance(position_1: NDArray,
                                    position_2: Optional[NDArray],
                                    ) -> NDArray:
    """
    Manhattan distance calculation (cityblock) -
    Parameters
    ----------
    position_1 : set of 3-D points to calculate distances between
    position_2 : if we provide another set of points, get the distance
    between position_1 and position_2

    Returns
    -------
    the calculated distance metric as np array
    """
    num_samples, num_dimensions = position_1.shape
    if position_2 is None:
        return pdist(position_1, position_2, metric='cityblock').reshape(num_samples, -1)
    else:
        match_shapes(position_1, position_2, last_dim_only=True)
    return cdist(position_1, position_2, metric='cityblock').reshape(-1, num_dimensions)


def calculate_3d_euclidean_distance(position_1: NDArray,
                                    position_2: Optional[NDArray],
                                    ) -> NDArray:
    """

    Parameters
    ----------
    position_1 : set of 3-D points to calculate distances between
    position_2 : if we provide another set of points, get the distance
    between position_1 and position_2

    Returns
    -------
    the calculated distance metric as np array
    """
    num_samples, num_dimensions = position_1.shape
    if position_2 is None:
        return pdist(position_1, position_2, metric='seuclidean').reshape(num_samples, -1)
    else:
        match_shapes(position_1, position_2, last_dim_only=True)
    return cdist(position_1, position_2, metric='seuclidean').reshape(-1, num_dimensions)


def calculate_3d_cosine_distance(position_1: NDArray,
                                    position_2: Optional[NDArray],
                                    ) -> NDArray:
    """

    Parameters
    ----------
    position_1 : set of 3-D points to calculate distances between
    position_2 : if we provide another set of points, get the distance
    between position_1 and position_2

    Returns
    -------
    the calculated distance metric as np array
    """
    num_samples, num_dimensions = position_1.shape
    if position_2 is None:
        return pdist(position_1, position_2, metric='cosine').reshape(num_samples, -1)
    else:
        match_shapes(position_1, position_2, last_dim_only=True)
    return cdist(position_1, position_2, metric='cosine').reshape(-1, num_dimensions)


def calculate_3d_mahalanobis_distance(position_1: NDArray,
                                    position_2: Optional[NDArray]
                                    ) -> NDArray:
    """

    Parameters
    ----------
    position_1 : set of 3-D points to calculate distances between
    position_2 : if we provide another set of points, get the distance
    between position_1 and position_2

    Returns
    -------
    the calculated distance metric as np array
    """
    num_samples, num_dimensions = position_1.shape
    if position_2 is None:
        return pdist(position_1, position_2, metric='mahalanobis').reshape(num_samples, -1)
    else:
        match_shapes(position_1, position_2, last_dim_only=True)
    return cdist(position_1, position_2, metric='mahalanobis').reshape(-1, num_dimensions)


class DistanceMetric(Enum):
    """ enum to type out distance metric calculations"""
    manhattan = calculate_3d_manhattan_distance
    cosine = calculate_3d_cosine_distance
    euclidean = calculate_3d_euclidean_distance
    mahalanobis = calculate_3d_mahalanobis_distance

