import functools
import numpy as np
import os
import json
from ..log import *


__all__ = ['set_cuda', 'set_tf_log', 'lazy_property', 'failsafe',
           'sample_multivariate', 'score2mask', 'pprint']


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


def failsafe(return_value=None):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                saye('@failsafe %s() ended with %s.' % (f.__name__, e.__class__.__name__))
                return return_value
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


def score2mask(H, W, K, Hs, Ws, scores) -> np.ndarray:
    """

    :param int H:
    :param int W:
    :param int K:
    :param Hs:
    :param Ws:
    :param scores:
    :return:
    """
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)
    for h, w, s in zip(Hs, Ws, scores):
        mask[h: h + K, w: w + K] += s
        cnt[h: h + K, w: w + K] += 1
    cnt[cnt == 0] = 1  # avoid divide by zero
    return mask / cnt


def pprint(obj):
    print(json.dumps(obj, indent=4))
