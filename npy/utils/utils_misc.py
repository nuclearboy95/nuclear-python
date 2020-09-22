import functools
import numpy as np
import os
import json

from ..log import *

__all__ = ['set_warning', 'set_cuda', 'set_tf_log',
           'lazy_property', 'failsafe',
           'sample_multivariate', 'pprint', 'get_hostname']


def get_hostname():
    import socket
    return socket.gethostname()


def set_warning():
    import warnings
    warnings.filterwarnings('ignore')


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


def pprint(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=4))
