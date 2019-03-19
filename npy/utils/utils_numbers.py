import numpy as np
from math import log, ceil
from functools import partial
from itertools import product


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
