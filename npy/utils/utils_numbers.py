import numpy as np
from math import log, ceil, floor, log10
from functools import partial
from itertools import product


__all__ = ['npw', 'nmax', 'ranges', 'ceil_x', 'ceil_to_1', 'top_kth', 'max_n', 'max95', 'max99']


def top_kth(arr, f):
    if f == 1:
        return arr.max()
    arr_flattened = np.reshape(arr, [-1])
    num = len(arr_flattened)
    num_n = int(f * num)
    arr_sorted = np.sort(arr_flattened)
    return arr_sorted[num_n]


max_n = top_kth
max95 = partial(top_kth, f=0.95)
max99 = partial(top_kth, f=0.99)


def npw(n, power=10):
    try:
        exponent = log(n, power)
    except ValueError:
        print(n, power)
        raise
    exponent = ceil(exponent)
    return power ** exponent


def nmax(n):
    npw10 = npw(n)
    if npw10 / 2 > n:
        return npw10 / 2
    else:
        return npw10


def ranges(*args):
    generators = [range(arg) for arg in args]
    return product(*generators)


def ceil_x(value, d):
    """
    ceil *value* to 10 to the power of *d*.

    ceil_x(5.43, -1) -> 10.0
    ceil_x(5.43, 0) -> 6.0
    ceil_x(5.43, 1) -> 5.5
    ceil_x(5.43, 2) -> 5.43

    :param float value:
    :param int d:
    :return:
    """

    mult = 10 ** d
    return ceil(value * mult) / mult


def ceil_to_1(x):
    """
    Leave only 1 digit.

    :param float x:

    :return:
    """
    if x == 0:
        return 0

    try:
        d = -int(floor(log10(abs(x))))
        return ceil_x(x, d)

    except ValueError:
        print('Input x: {}'.format(x))
        raise
