import numpy as np


__all__ = ['info']


def _info_ndarray(obj) -> str:
    """

    :param np.ndarray obj:
    :return:
    """
    if np.prod(obj.shape) != 0:
        return 'np.ndarray  shape: %s  ndtype: %s  max: %s  min: %s  mean: %s  std: %s  med: %s' % (
            obj.shape, obj.dtype, obj.max(), obj.min(), obj.mean(), obj.std(), np.median(obj)
        )

    else:
        return 'np.ndarray  shape: %s  ndtype: %s' % (
            obj.shape, obj.dtype
        )


def info(obj) -> str:
    if isinstance(obj, np.ndarray):
        return _info_ndarray(obj)
    return str(obj)
