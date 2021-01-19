import functools
import numpy as np
import os
import json
import pandas as pd

from ..log import *
from .datastructure import attrdict

__all__ = ['lazy_property', 'failsafe',
           'log_function', 'log_function_self',
           'return_array', 'return_dict', 'return_list', 'return_attrdict', 'return_dataframe']


def log_function_self(log_args=True):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            log_str = f'{f.__name__} called.'
            if log_args:
                log_str += ' ('
                arg_str = ', '.join([str(arg) for arg in args[1:]])
                kwargs_str = ', '.join([f'{k}={arg}' for k, arg in kwargs.items()])
                log_str += ', '.join([arg_str, kwargs_str])
                log_str += ')'

            sayd(log_str)
            result = f(*args, **kwargs)
            return result

        return wrapper

    return decorator


def log_function(log_args=True):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            log_str = f'{f.__name__} called.'
            if log_args:
                log_str += ' ('
                arg_str = ', '.join([str(arg) for arg in args])
                kwargs_str = ', '.join([f'{k}={arg}' for k, arg in kwargs.items()])
                log_str += ', '.join([arg_str, kwargs_str])
                log_str += ')'

            sayd(log_str)
            result = f(*args, **kwargs)
            return result

        return wrapper

    return decorator


#############

def return_list(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return list(result)

    return wrapper


def return_dict(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return dict(result)

    return wrapper


def return_dataframe(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return pd.DataFrame(result)

    return wrapper


def return_attrdict(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return attrdict(result)

    return wrapper


def return_array(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        result = f(*args, **kwargs)
        return np.asarray(result)

    return wrapper


#############

def lazy_property(f):
    attribute = '_cache_' + f.__name__

    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


def failsafe(return_value=None):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                return result
            except Exception as e:
                saye('@failsafe %s() ended with %s.' % (f.__name__, e.__class__.__name__))
                return return_value

        return wrapper

    return decorator
