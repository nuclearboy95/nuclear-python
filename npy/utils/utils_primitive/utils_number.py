import numpy as np
from math import log, ceil, log10
from functools import partial


__all__ = ['npw', 'nmax', 'top_kth', 'max_n', 'max95', 'max99', 'round_to_significants']


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


def round_to_significants(v: int, n_significants: int) -> int:
    if v == 0:
        return 0

    n_digits = int(log10(abs(v))) + 1
    v = round(v, -(n_digits - n_significants))
    return v
