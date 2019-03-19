import time
from contextlib import contextmanager
import cProfile


__all__ = ['task']


@contextmanager
def task(blockname='Noname', debug=False, detailed=False, sortby='cumtime'):
    if debug:
        if detailed:
            profiler = cProfile.Profile()
            profiler.enable()

        else:
            print('%s start.' % blockname)
            s = time.time()

    yield

    if debug:
        if detailed:
            profiler.disable()
            profiler.print_stats(sortby)

        else:
            e = time.time()
            print('%s end. Took %.2fs' % (blockname, e - s))
