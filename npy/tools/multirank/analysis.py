import numpy as np


__all__ = ['argmin', 'argmax']


def argmin(X):
    ind = np.argmin(X)
    return np.unravel_index(ind, X.shape)


def argmax(X):
    ind = np.argmax(X)
    return np.unravel_index(ind, X.shape)
