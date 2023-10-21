import numpy as np

__all__ = ['is_array', 'is_singleton', 'is_vector', 'is_matrix', 'is_multirank', 'argmin', 'argmax']


def is_array(v):
    return isinstance(v, list) or isinstance(v, np.ndarray)


def is_singleton(v):
    return not is_array(v)


def is_vector(X):
    if is_singleton(X):
        return False
    return is_singleton(X[0])


def is_matrix(X):
    if is_singleton(X):
        return False
    return is_vector(X[0])


def is_multirank(X):
    if is_singleton(X):
        return False
    return is_matrix(X[0])


#########

def argmin(X):
    ind = np.argmin(X)
    return np.unravel_index(ind, X.shape)


def argmax(X):
    ind = np.argmax(X)
    return np.unravel_index(ind, X.shape)
