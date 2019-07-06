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


def calc_num_batch(num_data, batch_size):
    return int(math.ceil(num_data / batch_size))


def shuffled(x, y=None):
    inds = np.random.permutation(len(x))
    if y is None:
        return take(x, inds)
    else:
        return take(x, inds), take(y, inds)


def track(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        sayd('%s() called.' % f.__name__)
        result = f(*args, **kwargs)
        return result
    return wrapper


def trackall(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        sayd('%s() called.' % f.__name__)
        result = f(*args, **kwargs)
        sayd('%s() finished.' % f.__name__)
        return result
    return wrapper


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
