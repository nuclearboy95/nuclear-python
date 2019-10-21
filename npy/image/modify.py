import numpy as np
from .basic import shape
from .dtype import assure_dtype_uint8

__all__ = ['pad', 'rgb2gray', 'gray2rgb', 'add_border']


def rgb2gray(images, keep_dims=False) -> np.ndarray:
    """

    :param np.ndarray images:
    :param bool keep_dims:
    :return:
    """
    H, W, C = shape(images)
    if C == 1:
        return images

    elif C == 3:
        R, G, B = images[..., 0], images[..., 1], images[..., 2]
        result = 0.2989 * R + 0.5870 * G + 0.1140 * B
        if keep_dims:
            return np.expand_dims(result, axis=-1).astype(images.dtype)
        else:
            return result.astype(images.dtype)

    else:
        raise ValueError()


def gray2rgb(images):
    H, W, C = shape(images)
    if C != 1:
        return images

    if images.shape[-1] != C:
        images = np.expand_dims(images, axis=-1)

    tile_shape = np.ones(len(images.shape), dtype=int)
    tile_shape[-1] = 3
    images = np.tile(images, tile_shape)
    return images


def pad(images, K, shape=None):
    if shape is None:
        shape = images.shape[1:]
    pad_width = ((0, 0), (K, K), (K, K))
    if len(shape) == 3:
        pad_width += ((0, 0),)

    return np.pad(images, pad_width, mode='constant')


def add_border(images, color=(0, 255, 0), border=0.07):
    H, W, C = shape(images)

    if isinstance(border, float):  # if fraction
        border = int(round(min(H, W) * border))

    T = border
    images = images.copy()
    images = assure_dtype_uint8(images)
    images[:, :T, :] = color
    images[:, -T:, :] = color
    images[:, :, :T] = color
    images[:, :, -T:] = color

    return images
