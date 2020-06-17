import functools
import numpy as np
import os
import json
from .utils_primitive import ranges
from ..calc import cnn_output_size
from ..log import *


__all__ = ['set_cuda', 'set_tf_log', 'lazy_property', 'failsafe',
           'sample_multivariate', 'score2mask', 'pprint', 'avgpool2d',
           'upsample_scoremask', 'upsample_scoremasks']


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


def upsample_scoremask(score_mask, output_shape, K: int, S: int) -> np.ndarray:
    H, W = output_shape
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)

    I, J = score_mask.shape[:2]
    for i, j in ranges(I, J):
        h, w = i * S, j * S

        mask[h: h + K, w: w + K] += score_mask[i, j]
        cnt[h: h + K, w: w + K] += 1

    cnt[cnt == 0] = 1

    return mask / cnt


def upsample_scoremasks(score_masks, output_shape, K: int, S: int) -> np.ndarray:
    N = score_masks.shape[0]
    results = [upsample_scoremask(score_masks[n], output_shape, K, S) for n in range(N)]
    return np.asarray(results)


def pprint(obj):
    print(json.dumps(obj, indent=4))


def avgpool2d(x, K, S):
    H, W = x.shape[:2]
    I = cnn_output_size(H, K, S)
    J = cnn_output_size(W, K, S)
    ret = np.zeros((I, J), dtype=x.dtype)
    for i, j in ranges(I, J):
        h = i * S
        w = j * S
        p = x[h: h + K, w: w + K]
        ret[i, j] = p.mean()
    return ret
