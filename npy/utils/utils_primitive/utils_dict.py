import numpy as np


__all__ = ['filter_d_of_l_of_num', 'filter_d_of_num', 'append_d_of_l', 'inv_d']


def filter_d_of_l_of_num(d) -> dict:
    """
    Filter lists of numbers in a dictionary.

    :param dict d:

    :return:
    """
    keys = list(d.keys())
    ret = {}

    keys = list(filter(lambda k: isinstance(d[k], list) or isinstance(d[k], np.ndarray), keys))
    ret = {k: np.asarray(d[k]) for k in keys}

    keys = list(filter(lambda k: np.issubdtype(ret[k].dtype, np.number), keys))
    ret = {k: ret[k] for k in keys}

    return ret


def filter_d_of_num(d) -> dict:
    """

    Filter numbers in a dictionary.

    :param d:
    :return:
    """
    keys = list(d.keys())
    keys = list(filter(lambda k: np.issubdtype(type(d[k]), np.number), keys))
    ret = {k: d[k] for k in keys}
    return ret


def append_d_of_l(d_of_l, d):
    for key, value in d.items():
        d_of_l[key].append(value)


def inv_d(d):
    return {v: k for k, v in d.items()}
