import math


__all__ = ['num_batch', 'num_segments']


def num_batch(num_data, batch_size, strict=False) -> int:
    if strict:
        return num_data // batch_size
    else:
        return int(math.ceil(num_data / batch_size))


def num_segments(length, stride, strict=False):
    """

    :param int length:
    :param int stride:
    :param bool strict:
    :return:
    """
    N = 1 + length // stride
    if strict and length % stride != 0:
        N += 1
    return N
