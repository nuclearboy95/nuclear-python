import numpy as np
from functools import partial
from .core import get_writer

__all__ = [
    'add_scalars', 'add_scalar', 'add_hist', 'add_image', 'add_images',
    'add_HWC', 'add_CHW', 'add_NCHW', 'add_NHWC'
]


##############################################


def add_scalars(d, step=None, name=None, prefix=''):
    for k, v in d.items():
        add_scalar(k, v, step=step, name=name, prefix=prefix)


def add_scalar(key, value, step=None, name=None, prefix=''):
    writer = get_writer(name)

    if prefix:
        key = f'{prefix}/{key}'

    writer.add_scalar(key, value, global_step=step)


def add_hist(key, arr, step=None, name=None, prefix=''):
    writer = get_writer(name)

    if prefix:
        key = f'{prefix}/{key}'

    writer.add_histogram(key, arr, global_step=step)


def add_image(key, image, step=None, name=None, prefix='', dataformats='HWC'):
    writer = get_writer(name)

    if prefix:
        key = f'{prefix}/{key}'

    writer.add_image(key, image, global_step=step, dataformats=dataformats)


add_HWC = partial(add_image, dataformats='HWC')
add_CHW = partial(add_image, dataformats='CHW')


def add_images(key, images, step=None, name=None, prefix='', dataformats='NHWC'):
    writer = get_writer(name)

    if prefix:
        key = f'{prefix}/{key}'

    if isinstance(images, list):
        images = np.asarray(images)

    writer.add_images(key, images, global_step=step, dataformats=dataformats)


add_NHWC = partial(add_image, dataformats='NHWC')
add_NCHW = partial(add_image, dataformats='NCHW')
