"""
Generic or broadly applicable utility functions for neural webs
"""
import time
from typing import Optional, Callable
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import numpy as np

EPSILON = 1e-15

# -------------- Decorators  --------------
def timed_lru_cache(seconds: int, maxsize: int = 128):
    """ from realpython example - uses as @timed_lru_cache"""
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache


def log_timeit(func: Callable, logger: Optional):
    """
    to use with a created log object
    Parameters
    ----------
    func :
    logger :

    Returns
    -------

    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        message = f'Call of {func.__name__}{args} ({kwargs}) Took' \
                  f' {total_time:.4f} seconds'
        if not logger:
            print(message)
        else:
            logger.info(message)

        return result
    return timeit_wrapper


def decay_learning_rate(lr, final_lr = .001):
    # TODO: add in the sigmoid decay
    decay = .75
    return max(lr * decay, final_lr)


def cosine_similarity(u, v):
    '''Angualar similarity between two vectors, u and v
    u, v = word vectors for indiviudal words, output from our word2vec w2v.word_vec('input')
    if highly similar, return is close to 1
    if highly dissimilar, return is close to -1
    '''
    #dist = 0.0
    dot = u @ v
    # L2 norm is the length of the vector
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    cosine_theta = dot / (norm_u * norm_v)
    #1 - cosine_theta?  check scipy.spatial.distance
    #also check scipy.spatial.distance.euclidean
    return cosine_theta


def update_mean_std(mean1, std1, count1, mean2, std2, count2):
    '''**********ARGUMENTS**********
    :param mean1: self.data_type_means
    :param std1: self.data_type_stds
    :param count1: num_before
    :param mean2: this_data_mean
    :param std2: this_data_std
    :param count2: num_new
    **********RETURNS**********
    :return: new mean value, new std value
    '''
    # from www.burtonsys.com/climate/composite_sd.php#python
    countBoth = count1 + count2
    meanBoth = (count1 * mean1 + count2 * mean2) / countBoth
    var1 = std1 ** 2
    var2 = std2 ** 2
    # error sum of squares
    ESS = var1 * (count1 - 1) + var2 * (count2 - 1)
    # total group sum of squares
    TGSS = (mean1 - meanBoth) ** 2 * count1 + (mean2 - meanBoth) ** 2 * count2
    varBoth = (ESS + TGSS) / (countBoth - 1)
    stdBoth = np.sqrt(varBoth)

    return meanBoth, stdBoth


def standardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be standardized
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: standardized data
    '''
    return (X - means) / (stds + EPSILON)


def unstandardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be returned to normal space
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: unstandardized data
    '''
    return (stds - EPSILON) * X + means


def make_patches(incoming_x, patch_size, stride=2):
    '''data: n_samples x n_pixels  (flattened square images)'''

    incoming_x = np.ascontiguousarray(incoming_x)  # make sure X values are contiguous in memory

    n_samples = incoming_x.shape[0]
    image_size = int(np.sqrt(incoming_x.shape[1]))
    n_patches = (image_size - patch_size) // stride + 1

    nb = incoming_x.itemsize  # number of bytes each value

    new_shape = [n_samples, n_patches, n_patches, patch_size, patch_size]

    new_strides = [image_size * image_size * nb,
                   image_size * stride * nb,
                   stride * nb,
                   image_size * nb,
                   nb]
    incoming_x = np.lib.stride_tricks.as_strided(incoming_x, shape=new_shape, strides=new_strides)
    incoming_x = incoming_x.reshape((n_samples, n_patches * n_patches, patch_size * patch_size))

    return incoming_x


def rolling_windows(data, window, num_overlap=0):
    '''**********ARGUMENTS**********
    :param data: multi-dimensional array, last dimension will be segmented into windows
    :param window: number of samples per window - 'size' of each window
    :param num_overlap: number of samples of overlap between consecutive windows
    **********RETURNS**********
    :return: array of [windows, samples, datashape]
    '''
    #shapeUpToLastDim = data.shape[:-1]
    num_samples = data.shape[-1]

    if num_overlap > window:
        print('rollingWindows: num_overlap > window, so setting to window-1')
        num_overlap = window - 1  # shift by one

    num_Shift = window - num_overlap
    nWindows = int((num_samples - window + 1) / num_Shift)
    newShape = data.shape[:-1] + (nWindows, window)
    strides = data.strides[:-1] + (a.strides[-1] * num_Shift, data.strides[-1])

    return np.lib.stride_tricks.as_strided(data, shape=newShape, strides=strides)
