import numpy as np
from .utils_list import isarray, isnum


__all__ = ['keys_d_of_l_of_num', 'keys_d_of_num',
           'filter_d_of_l_of_num', 'filter_d_of_num', 'append_d_of_l', 'inv_d']


def keys_d_of_l_of_num(d) -> list:
    """

    :param dict d:
    :return:
    """
    def is_valid_key(k):
        if not isarray(d[k]):
            return False
        if not isnum(d[k][0]):
            return False
        return True

    keys = list(filter(is_valid_key, d.keys()))
    return keys


def keys_d_of_num(d) -> list:
    """

    :param dict d:
    :return:
    """
    def is_valid_key(k):
        return isnum(d[k])

    keys = list(filter(is_valid_key, d.keys()))
    return keys


#############################


def filter_d_of_l_of_num(d) -> dict:
    """
    Filter lists of numbers in a dictionary.

    :param dict d:

    :return:
    """
    keys = keys_d_of_l_of_num(d)
    return {k: d[k] for k in keys}


def filter_d_of_num(d) -> dict:
    """

    Filter numbers in a dictionary.

    :param d:
    :return:
    """
    keys = keys_d_of_num(d)
    return {k: d[k] for k in keys}


def append_d_of_l(d_of_l, d):
    for key, value in d.items():
        d_of_l[key].append(value)


def inv_d(d):
    return {v: k for k, v in d.items()}
