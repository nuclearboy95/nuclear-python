import numpy as np


__all__ = ['is_array', 'is_singleton', 'is_vector', 'is_matrix']


def is_array(v):
    return isinstance(v, list) or isinstance(v, np.ndarray)


def is_singleton(v):
    return not is_array(v)


def is_vector(X):
    return is_singleton(X[0])


def is_matrix(X):
    return is_vector(X[0])

