from .core import get_writer

__all__ = ['add_scalars', 'add_scalar']


##############################################


def add_scalars(d, step=None, name=None, prefix=''):
    for k, v in d.items():
        add_scalar(k, v, step=step, name=name, prefix=prefix)


def add_scalar(key, value, step=None, name=None, prefix=''):
    writer = get_writer(name)

    if prefix:
        key = f'{prefix}/{key}'

    writer.add_scalar(key, value, global_step=step)
