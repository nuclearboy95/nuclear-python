import functools
from .log import sayd
import time


TRACK_FLAG = False
TRACK_TIME_FLAG = False


__all__ = ['on', 'off', 'watch']


def on(watch_time=False):
    global TRACK_FLAG
    global TRACK_TIME_FLAG

    TRACK_FLAG = True
    TRACK_TIME_FLAG = watch_time


def off():
    global TRACK_FLAG
    TRACK_FLAG = False


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
