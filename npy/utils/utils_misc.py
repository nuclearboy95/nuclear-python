import functools
import numpy as np
import math
import os
from ..log import *
from .utils_primitive import take


def set_cuda(*args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(arg) for arg in args)


def set_tf_log(level=5):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def lazy_property(f):
    attribute = '_cache_' + f.__name__

    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


def shuffled(x, y=None):
    inds = np.random.permutation(len(x))
    if y is None:
        return take(x, inds)
    else:
        return take(x, inds), take(y, inds)


def failsafe(value=None):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                saye('@failsafe %s() ended with %s.' % (f.__name__, e.__class__.__name__))
                return value
        return wrapper
    return decorator


def sample_multivariate(mu, cov, N, D):
    """

    :param np.ndarray mu:
    :param np.ndarray cov:
    :param int N:
    :param int D:

    :return:
    """
    A = np.linalg.cholesky(cov)
    z = np.random.normal(size=N * D).reshape(D, N)
    return mu[np.newaxis, :] + np.dot(A, z).T


def score2mask(H, W, K, Hs, Ws, scores):
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)
    for H, W, score in zip(Hs, Ws, scores):
        mask[H: H + K, W: W + K] += score
        cnt[H: H + K, W: W + K] += 1
    cnt[cnt == 0] = 1  # avoid divide by zero
    return mask / cnt

