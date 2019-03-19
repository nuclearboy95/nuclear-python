import functools
import itertools
import multiprocessing as mp
import numpy as np


__all__ = ['exhaustive', 'iarg']


def map_wrapper(f):
    def wrapper(args_and_kwargs):
        args, kwargs = args_and_kwargs
        return f(*args, **kwargs)
    return wrapper


def kwargs_spawner(kwargs):
    is_dynamic = {k: isinstance(v, IterableArg) for k, v in kwargs.items()}
    if not any(is_dynamic.values()):
        yield False, kwargs

    else:
        kwargs_static = {k: v for k, v in kwargs.items() if not is_dynamic[k]}
        kwargs_dynamic = {k: v for k, v in kwargs.items() if is_dynamic[k]}
        keys = list(kwargs_dynamic.keys())
        values = [kwargs_dynamic[k] for k in keys]

        for values_generated in itertools.product(*values):
            kwargs_generated = {k: v for k, v in zip(keys, values_generated)}

            # build kwargs
            kwargs0 = kwargs_static.copy()
            kwargs0.update(kwargs_generated)
            yield True, kwargs0


def argument_spawner(args, kwargs):  # generator
    is_dynamic = [isinstance(v, IterableArg) for v in args]
    if not any(is_dynamic):
        for is_dynamic_kwargs, kwargs0 in kwargs_spawner(kwargs):
            yield is_dynamic_kwargs, args, kwargs0

    else:
        args_dynamic = itertools.compress(args, is_dynamic)
        ind_dynamic = np.nonzero(is_dynamic)[0]

        args0 = list(args)
        # TODO reverse iterating order
        for args_generated in itertools.product(*args_dynamic):
            for ind, arg in zip(ind_dynamic, args_generated):
                args0[ind] = arg

            for _, kwargs0 in kwargs_spawner(kwargs):
                yield True, args0, kwargs0


def exhaustive(parallel=False):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = list()

            if parallel:
                params = list(argument_spawner(args, kwargs))

                if len(params) > 1:
                    with mp.Pool() as pool:
                        def wrapper(args_and_kwargs0):
                            args00, kwargs00 = args_and_kwargs0
                            return f(*args00, **kwargs00)

                        # f_wrapped = wrapper(f)
                        args_and_kwargs = [param[1:] for param in params]
                        return pool.map(wrapper, args_and_kwargs)

            for multi, args0, kwargs0 in argument_spawner(args, kwargs):
                result = f(*args0, **kwargs0)

                if multi:
                    results.append(result)
                else:
                    results = result

            return results
        return wrapper
    return decorator


class IterableArg:
    def __iter__(self):
        return NotImplemented


class iarg(IterableArg):
    def __init__(self, low, high=None):
        if high is None:
            if isinstance(low, int):
                self.iterable = range(low)
            else:
                self.iterable = low
        else:
            self.iterable = range(low, high)

    def __iter__(self):
        return iter(self.iterable)
