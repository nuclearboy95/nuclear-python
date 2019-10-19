import math


__all__ = ['num_batch', 'num_segments', 'greatest_less_multiple']


def num_batch(num_data, batch_size, strict=False) -> int:
    if strict:
        return num_data // batch_size
    else:
        return int(math.ceil(num_data / batch_size))


def num_segments(length, stride, strict=False) -> int:
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


def greatest_less_multiple(N, d) -> int:
    """
    f(100, 3) = 99

    :param int N:
    :param int d:
    :return:
    """
    return d * (N // d)
