import numpy as np
from collections.abc import Iterable


__all__ = ['take']


def take(l, inds_or_ind):
    if isinstance(l, np.ndarray):
        return l[inds_or_ind]
    elif isinstance(inds_or_ind, Iterable):
        return [l[i] for i in inds_or_ind]
    else:
        return l[inds_or_ind]

