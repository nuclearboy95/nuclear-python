import functools
from .basic import get_fmt


__all__ = ['allowable_fmts']


def allowable_fmts(fmts):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(x, *args, **kwargs):
            assert get_fmt(x) in fmts
            return f(x, *args, **kwargs)
        return wrapper

    return decorator
