import functools
import numpy as np
import os
import json
import pandas as pd

from ..log import *
from .datastructure import attrdict

__all__ = ['set_warning', 'set_cuda', 'set_tf_log',
           'sample_multivariate', 'pprint', 'get_hostname',
           'Singleton']


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance


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
