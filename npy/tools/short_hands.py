

__all__ = ['lm', 'lf']


def lm(*args, **kwargs):
    return list(map(*args, **kwargs))


def lf(*args, **kwargs):
    return list(filter(*args, **kwargs))
