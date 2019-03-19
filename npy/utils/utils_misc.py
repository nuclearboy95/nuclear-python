import functools
import numpy as np
import math
from collections.abc import Iterable
import os


def set_cuda(*args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(arg) for arg in args)


def set_tf_log(level):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(level)


def take(l, inds_or_ind):
    if isinstance(l, np.ndarray):
        return l[inds_or_ind]
    elif isinstance(inds_or_ind, Iterable):
        return [l[i] for i in inds_or_ind]
    else:
        return l[inds_or_ind]


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


def calc_num_batch(num_data, batch_size):
    return int(math.ceil(num_data / batch_size))


def inv_d(d):
    return {v: k for k, v in d.items()}


def append_d_of_l(d_of_l, d):
    for key, value in d.items():
        d_of_l[key].append(value)
