import numpy as np
from math import log, ceil, floor, log10
from functools import partial
from itertools import product


__all__ = ['npw', 'nmax', 'ranges', 'top_kth', 'max_n', 'max95', 'max99',
           'range_strict', 'ij_range']


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


def ij_range(H, W, K=1, S=1, O=0):
    """

    :param int H:
    :param int W:
    :param int K:
    :param int S:
    :param int O:
    :return:
    """

    for i in range(O, H - K + 1, S):
        for j in range(O, W - K + 1, S):
            yield i, j


def range_strict(low, high, stride):
    for i in range(low, high, stride):
        yield i

    print('range_strict(%d, %d, %d) end.' % (low, high, stride))
    if (high - low) % stride != 0:
        print('i am here', high-stride+1)
        yield high - stride + 1

