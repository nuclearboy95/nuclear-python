import numpy as np


__all__ = ['info']


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


def info(obj, name='') -> str:
    if isinstance(obj, np.ndarray):
        return _info_ndarray(obj, name)
    return str(obj)
