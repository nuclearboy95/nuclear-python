

__all__ = ['lm', 'lf']


def lm(*args, **kwargs):
    """
    shortcut of list(map(*args, **kwags))

    :param args:
    :param kwargs:
    :return:
    """
    return list(map(*args, **kwargs))


def lf(*args, **kwargs):
    """
    shortcut of list(filter(*args, **kwags))

    :param args:
    :param kwargs:
    :return:
    """
    return list(filter(*args, **kwargs))
