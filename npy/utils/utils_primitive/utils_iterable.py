from itertools import product

__all__ = ['ranges', 'range_strict', 'ij_range']


def printi(l):
    for v in l:
        print(v)


def ranges(*args):
    generators = [range(arg) for arg in args]
    return product(*generators)


def ij_range(H, W, K=1, S=1, O=0):
    """

    :param int H:
    :param int W:
    :param int K:
    :param int S:
    :param int O:
    :return:
    """

    for i in range(O, H - K + 1, S):
        for j in range(O, W - K + 1, S):
            yield i, j


def range_strict(low, high, stride):
    for i in range(low, high, stride):
        yield i

    print('range_strict(%d, %d, %d) end.' % (low, high, stride))
    if (high - low) % stride != 0:
        print('i am here', high - stride + 1)
        yield high - stride + 1
