import math
from functools import partial


__all__ = ['num_batch', 'num_segments', 'closest_multiple', 'closest_power',
           'ceil_pow', 'ceil_to_1digit',
           'greatest_less_multiple', 'least_greater_multiple',
           'cnn_output_size'
           ]


def num_batch(num_data, batch_size, strict=False) -> int:
    """
    :param int num_data:
    :param int batch_size:
    :param bool strict:
    """
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


def closest_multiple(N, base, mode='less') -> int:
    """
    :param int N:
    :param int base:
    :param str mode:
    """
    mode = mode.lower()
    assert mode in ['greater', 'less']

    if mode == 'greater':  # least greater multiple
        return base * int(math.ceil(N / base))

    elif mode == 'less':  # greatest less multiple
        return base * (N // base)

    else:
        raise ValueError()


def closest_power(v, base=2, mode='less') -> int:
    """
    :param v:
    :param int base:
    :param str mode:
    """
    mode = mode.lower()
    assert mode in ['greater', 'less']
    assert v >= 0

    log_v = math.log(abs(v), base)

    if mode == 'greater':
        exponent = int(math.ceil(log_v))

    elif mode == 'less':
        exponent = int(math.floor(log_v))

    else:
        raise ValueError()

    power = base ** exponent
    return power


def ceil_pow(v, exponent, base=10) -> float:
    """
    ceil *v* to *base*^*exponent*.

    ceil_x(5.43, -1, base=10) -> 5.5
    ceil_x(5.43, 0, base=10) -> 6.0
    ceil_x(5.43, 1, base=10) -> 10.0
    ceil_x(5.43, 2, base=10) -> 100.0

    :param v:
    :param int exponent:
    :param base:
    """
    power = base ** exponent
    return math.ceil(v / power) * power


def ceil_to_1digit(v) -> float:
    """
    Leave only 1 digit.

    :param float v:

    :return:
    """
    if v == 0:
        return 0

    try:
        exponent = int(math.floor(math.log10(abs(v))))
        return ceil_pow(v, exponent, base=10)

    except ValueError:
        print(f'Input x: {v}')
        raise


def cnn_output_size(H, K, S=1, P=0) -> int:
    """

    :param int H: input_size
    :param int K: filter_size
    :param int S: stride
    :param int P: padding
    :return:
    """
    return 1 + (H - K + 2 * P) // S


greatest_less_multiple = partial(closest_multiple, mode='less')
least_greater_multiple = partial(closest_multiple, mode='greater')
