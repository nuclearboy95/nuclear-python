import numpy as np

__all__ = ['info', 'prinfo']


def _info_ndarray(obj, name) -> str:
    """

    :param np.ndarray obj:
    :param str name:
    :return:
    """
    res_str = f'[np.ndarray] {name} shape: {obj.shape} dtype: {obj.dtype}'
    if np.prod(obj.shape) != 0:
        res_str += f'\n[np.ndarray] max: {obj.max():.3g}, min: {obj.min():.3g}, mean: {obj.mean():.3g}'

    return res_str


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


def _info_torchtensor(obj, name) -> str:
    """
    :param torch.Tensor obj:
    :param str name:
    :return:
    """
    res_str = f'[torch.Tensor] {name} shape: {obj.shape}, dtype: {obj.dtype}'
    if np.prod(obj.shape) != 0:
        res_str += f'\n[torch.Tensor] max: {obj.max():.3g}, min: {obj.min():.3g}, mean: {obj.mean():.3g}'

    return res_str


def info(obj, name='***') -> str:
    import torch
    if isinstance(obj, np.ndarray):
        return _info_ndarray(obj, name)
    elif isinstance(obj, list):
        return _info_list(obj, name)
    elif isinstance(obj, tuple):
        return _info_tuple(obj, name)
    elif isinstance(obj, torch.Tensor):
        return _info_torchtensor(obj, name)

    return str(obj)


def prinfo(obj, name=''):
    print(info(obj, name))
