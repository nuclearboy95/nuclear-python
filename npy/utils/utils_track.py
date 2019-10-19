import functools
from ..log import *
from ..track import TRACK_FLAG, TRACK_TIME_FLAG
import time


__all__ = ['watch']


def watch(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        FLAG = TRACK_FLAG
        if FLAG:
            sayd('%s() called.' % f.__name__)
            s = time.time()

        result = f(*args, **kwargs)

        if FLAG and TRACK_TIME_FLAG:
            e = time.time()
            sayd('%s() finished. Took %.2fs' % (f.__name__, e - s))

        return result
    return wrapper
