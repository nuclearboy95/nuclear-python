import numpy as np


__all__ = ['corrupt_labels', 'filter_labels']


def filter_labels(x, y, class_num: int) -> tuple:
    """

    :param np.ndarray x:
    :param np.ndarray y:
    :param int class_num:
    :return:
    """
    m = y < class_num
    x = x[m]
    y = y[m]
    return x, y


def corrupt_labels(y, c: float, class_num: int) -> np.ndarray:
    """

    :param np.ndarray y:  [N]
    :param float c:
    :param int class_num:
    :return:
    """
    N = y.shape[0]
    m = np.zeros(N, dtype=np.bool)
    Nc = int(N * c)
    m[:Nc] = 1
    np.random.shuffle(m)
    y = y.copy()
    y[m] = np.random.randint(class_num, size=Nc)
    return y
