import numpy as np
from .basic import shape, to_NHWC, match_fmt, assure_dtype_uint8
from .decorators import allowable_fmts
from ..constants import *

__all__ = ['pad', 'rgb2gray', 'gray2rgb', 'add_border']


@allowable_fmts([NHWC, HWC])
def rgb2gray(images_or_image, keep_dims=False) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param bool keep_dims:
    :return:
    """
    H, W, C = shape(images_or_image)
    assert C == 3, 'C(%s) should be 3' % C

    R, G, B = images_or_image[..., 0], images_or_image[..., 1], images_or_image[..., 2]
    result = 0.2989 * R + 0.5870 * G + 0.1140 * B

    if keep_dims:
        result = np.expand_dims(result, axis=-1)

    return result.astype(images_or_image.dtype)


@allowable_fmts([NHWC, HWC, NHW])
def gray2rgb(images) -> np.ndarray:
    H, W, C = shape(images)
    assert C == 1, 'C(%s) should be 1' % C

    images = to_NHWC(images)
    tile_shape = np.ones(len(images.shape), dtype=int)
    tile_shape[-1] = 3
    images = np.tile(images, tile_shape)
    return images


@allowable_fmts([NHWC, NHW])
def pad(images, K: int, shape_=None):
    if shape_ is None:
        shape_ = shape(images)
    pad_width = ((0, 0), (K, K), (K, K))
    if len(shape_) == 3:
        pad_width += ((0, 0),)

    return np.pad(images, pad_width, mode='constant')


def add_border(images_or_image, color=(0, 255, 0), border=0.07):
    H, W, C = shape(images_or_image)

    if isinstance(border, float):  # if fraction
        border = int(round(min(H, W) * border))

    T = border
    result = images_or_image.copy()
    result = to_NHWC(result)
    result = assure_dtype_uint8(result)
    result[:, :T, :] = color
    result[:, -T:, :] = color
    result[:, :, :T] = color
    result[:, :, -T:] = color
    result = match_fmt(result, images_or_image)

    return result
