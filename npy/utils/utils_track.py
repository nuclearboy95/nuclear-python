import functools
from ..log import *
from ..track import _TRACK_FLAG
import time


__all__ = ['watch', 'watchtime']


def watch(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if _TRACK_FLAG:
            sayd('%s() called.' % f.__name__)
        result = f(*args, **kwargs)
        return result
    return wrapper


def watchtime(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        FLAG = _TRACK_FLAG
        if FLAG:
            sayd('%s() called.' % f.__name__)
            s = time.time()
        result = f(*args, **kwargs)
        if FLAG:
            e = time.time()
            sayd('%s() finished. Took %.2fs' % (f.__name__, e - s))

        return result
    return wrapper
