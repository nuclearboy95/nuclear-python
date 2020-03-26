import numpy as np
from .utils_list import isarray, isnum


__all__ = ['keys_d_of_l_of_num', 'keys_d_of_num',
           'filter_d_of_l_of_num', 'filter_d_of_num', 'append_d_of_l', 'inv_d',
           'filter_keys', 'replace_keys', 'drop_keys']


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


def drop_keys(d: dict, keys: list) -> dict:
    return {
        k: v for k, v in d.items() if k not in keys
    }


def replace_keys(d: dict, key_d: dict) -> dict:
    return {
        key_d[k] if k in key_d else k: v
        for k, v in d.items()
    }


def filter_keys(d: dict, keys: list) -> dict:
    return {
        k: v for k, v in d.items() if k in keys
    }


def filter_d_of_l_of_num(d: dict) -> dict:
    """
    Filter lists of numbers in a dictionary.

    :param dict d:

    :return:
    """
    keys = keys_d_of_l_of_num(d)
    return filter_keys(d, keys)


def filter_d_of_num(d: dict) -> dict:
    """

    Filter numbers in a dictionary.

    :param d:
    :return:
    """
    keys = keys_d_of_num(d)
    return filter_keys(d, keys)


def append_d_of_l(d_of_l: dict, d: dict):
    for key, value in d.items():
        d_of_l[key].append(value)
    return d_of_l


def inv_d(d: dict) -> dict:
    return {v: k for k, v in d.items()}
