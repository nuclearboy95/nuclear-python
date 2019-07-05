import time
from contextlib import contextmanager
import cProfile
from ..log import sayd


__all__ = ['task']


@contextmanager
def task(blockname='Noname', debug=False, detailed=False, sortby='cumtime'):
    if debug:
        if detailed:
            profiler = cProfile.Profile()
            profiler.enable()

        else:
            sayd('%s start.' % blockname)
            s = time.time()

    yield

    if debug:
        if detailed:
            profiler.disable()
            profiler.print_stats(sortby)

        else:
            e = time.time()
            sayd('%s end. Took %.2fs' % (blockname, e - s))
