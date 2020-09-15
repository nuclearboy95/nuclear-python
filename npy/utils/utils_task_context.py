import time
from contextlib import contextmanager
import cProfile
import traceback
from ..log import sayd, saye
import npy

__all__ = ['task', 'sandbox', 'Axof', 'Rowof', 'Colof']


@contextmanager
def task(blockname='Noname', debug=False, detailed=False, sortby='cumtime', verbose=False):
    if debug:
        if detailed:
            profiler = cProfile.Profile()
            profiler.enable()

        else:
            sayd('%s start.' % blockname)
            s = time.time()
    else:
        if verbose:
            sayd(blockname)

    yield

    if debug:
        if detailed:
            profiler.disable()
            profiler.print_stats(sortby)

        else:
            e = time.time()
            sayd('%s end. Took %.2fs' % (blockname, e - s))


@contextmanager
def sandbox(blockname='Noname', send=True):
    npy.log.telegram(4)
    try:
        yield
    except:
        if send:
            saye(traceback.format_exc())


@contextmanager
def Axof(axes, row=None, col=None):
    ax = axes
    if row is not None:
        ax = ax[row]
    if col is not None:
        ax = ax[col]
    yield ax


@contextmanager
def Rowof(axes, row):
    yield axes[row]


@contextmanager
def Colof(axes, col):
    yield axes[col]
