import time


__all__ = ['clear_line', 'refine_result']


def clear_line(delay=True):
    print('\r', end='', flush=True)
    if delay:
        time.sleep(0.01)


def refine_result(d, prefix=None, i_batch=None) -> dict:
    if prefix is not None:
        d = {prefix + '/' + k: v for k, v in d.items()}

    if i_batch is not None:
        d['i_batch'] = i_batch

    return d
