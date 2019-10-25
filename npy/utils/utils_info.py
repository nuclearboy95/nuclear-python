import numpy as np


__all__ = ['info', 'prinfo']


def _info_ndarray(obj, name) -> str:
    """

    :param np.ndarray obj:
    :param str name:
    :return:
    """
    if np.prod(obj.shape) != 0:
        return '[np.ndarray] %s shape: %s dtype: %s, max: %.3g  min: %.3g  mean: %.3g  std: %.3g  med: %.3g' % (
            name, obj.shape, obj.dtype, obj.max(), obj.min(), obj.mean(), obj.std(), np.median(obj)
        )

    else:
        return '[np.ndarray] %s shape: %s dtype: %s' % (
            name, obj.shape, obj.dtype
        )


def _info_list(obj, name) -> str:
    """

    :param list obj:
    :param str name:
    :return:
    """
    length = len(obj)
    if length != 0:
        return '[list] %s length: %s, type([0]): %s' % (name, len(obj), type(obj[0]))
    else:
        return '[list] %s length: %s' % (name, len(obj))


def _info_tuple(obj, name) -> str:
    """

    :param tuple obj:
    :param str name:
    :return:
    """
    length = len(obj)
    if length != 0:
        return '[tuple] %s length: %s, type([0]): %s' % (name, len(obj), type(obj[0]))
    else:
        return '[tuple] %s length: %s' % (name, len(obj))


def info(obj, name='') -> str:
    if isinstance(obj, np.ndarray):
        return _info_ndarray(obj, name)
    elif isinstance(obj, list):
        return _info_list(obj, name)
    elif isinstance(obj, tuple):
        return _info_tuple(obj, name)
    return str(obj)


def prinfo(obj, name=''):
    print(info(obj, name))
