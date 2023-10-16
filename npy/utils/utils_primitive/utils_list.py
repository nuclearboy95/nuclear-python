import numpy as np
from collections.abc import Iterable


__all__ = ['take', 'isarray', 'isnum', 'shuffled', 'normalize_l']


def take(l, inds_or_ind):
    if isinstance(l, np.ndarray):
        return l[inds_or_ind]
    elif isinstance(inds_or_ind, Iterable):
        return [l[i] for i in inds_or_ind]
    else:
        return l[inds_or_ind]


def isarray(obj) -> bool:
    return isinstance(obj, list) or isinstance(obj, np.ndarray)


def isnum(obj) -> bool:
    return np.issubdtype(type(obj), np.number)


def shuffled(x, y=None):
    inds = np.random.permutation(len(x))
    if y is None:
        return take(x, inds)
    else:
        return take(x, inds), take(y, inds)


def normalize_l(arr):
    only_values = arr[~np.isinf(arr)]
    if len(only_values) == 0:
        return arr
    vmax = np.max(only_values)
    vmin = np.min(only_values)
    dur = vmax - vmin
    if dur == 0:
        dur = 1
    return (arr - vmin) / dur
