import numpy as np
from .utils_list import isarray, isnum

__all__ = ['keys_d_of_l_of_num', 'keys_d_of_num',
           'append_d_of_l', 'inv_d', 'replace_keys', 'drop_keys',
           'filter_d_of_l_of_num', 'filter_d_of_num', 'filter_keys', 'filter_empty',
           'dict_eq', 'l_of_dict_eq', 'add_d_of_l', 'merge_l_of_d'
           ]


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


def append_d_of_l(d_of_l: dict, d: dict):
    for key, value in d.items():
        d_of_l[key].append(value)
    return d_of_l


def merge_l_of_d(l_of_d: list) -> dict:
    if len(l_of_d) == 0:
        return {}

    keys = list(l_of_d[0].keys())
    ret = {k: [] for k in keys}
    for d in l_of_d:
        for k in keys:
            ret[k].append(d[k])
    return ret


def add_d_of_l(d_of_l: dict, d_of_l2: dict, reverse=False):
    if d_of_l is None:
        return d_of_l2

    if reverse:
        d_of_l, d_of_l2 = d_of_l2, d_of_l

    for key in d_of_l:
        d_of_l[key] = d_of_l[key] + d_of_l2[key]
    return d_of_l


def inv_d(d: dict) -> dict:
    return {v: k for k, v in d.items()}


####################

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


def filter_empty(d: dict) -> dict:
    keys = list(d.keys())
    for k in keys:
        if d[k] is None:
            del d[k]
    return d


####################

def dict_eq(d1: dict, d2: dict) -> bool:
    if d1.keys() != d2.keys():
        return False
    return all([d1[key] == d2[key] for key in d1])


def l_of_dict_eq(l1: list, l2: list) -> bool:
    if len(l1) != len(l2):
        return False
    return all([dict_eq(d1, d2) for d1, d2 in zip(l1, l2)])
