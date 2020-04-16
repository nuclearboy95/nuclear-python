import numpy as np
from .basic import *
from .recursive import get_sum, iter_vector


__all__ = ['unflatten_vectors', 'flatten_vectors']


def unflatten_vectors(X, Ns):
    if is_vector(Ns):
        if len(Ns) == 1:
            return [X]
        else:
            Ncum = np.cumsum(Ns)
            return np.split(X, Ncum[:-1])
    else:
        if len(Ns) == 1:
            return [unflatten_vectors(X, Ns[0])]
        else:
            Nsub = [get_sum(Nss) for Nss in Ns]
            Ncum = np.cumsum(Nsub)
            Xsplit = np.split(X, Ncum[:-1])
            return [
                unflatten_vectors(Xone, Nss) for Xone, Nss in zip(Xsplit, Ns)
            ]


def flatten_vectors(X):  # [N1, ..., N2, D] -> [N, D]
    ret = list()
    for vec in iter_vector(X):
        ret.append(vec)
    return np.stack(ret)

