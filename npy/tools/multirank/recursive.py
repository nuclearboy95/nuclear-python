import numpy as np
from .basic import *


__all__ = ['get_D', 'get_lengths', 'get_sum', 'get_dtype',
           'equal', 'iter_vector', 'iter_element']


def get_D(X):
    if is_vector(X):
        return len(X)
    else:
        return get_D(X[0])


def get_lengths(X):
    if is_matrix(X):
        return len(X)
    else:
        return [get_lengths(x) for x in X]


def get_sum(X):
    if is_singleton(X):
        return X
    else:
        return sum([get_sum(x) for x in X])


def get_dtype(X):
    if is_singleton(X):
        return type(X)
    else:
        return get_dtype(X[0])


def equal(X1, X2):
    if is_singleton(X1) or is_singleton(X2):
        return X1 == X2
    else:
        if len(X1) != len(X2):
            return False
        return all([equal(x1, x2) for x1, x2 in zip(X1, X2)])


def iter_vector(X):
    if is_vector(X):
        yield X
    else:
        for x in X:
            yield from iter_vector(x)


def iter_element(X):
    if is_singleton(X):
        yield X
    else:
        for x in X:
            yield from iter_element(x)
