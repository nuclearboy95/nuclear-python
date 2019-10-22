import numpy as np
from .basic import shape, nshape
from .dtype import assure_dtype_uint8

__all__ = ['pad', 'rgb2gray', 'gray2rgb', 'add_border',
           'to_NHWC', 'to_HWC', 'to_NHW']


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


def gray2rgb(images) -> np.ndarray:
    H, W, C = shape(images)
    assert C == 1, 'C(%s) should be 1' % C

    images = to_NHWC(images)
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


def to_NHWC(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    N, H, W, C = nshape(images_or_image)
    result_shape = (N, H, W, C)
    if images_or_image.shape == result_shape:
        return images_or_image
    else:
        return images_or_image.reshape(result_shape)


def to_HWC(images_or_image) -> np.ndarray:
    """
    :param np.ndarray images_or_image:
    :return:
    """
    N, H, W, C = nshape(images_or_image)
    assert N == 1, 'N(%s) should be 1' % C

    result_shape = (H, W, C)
    if images_or_image.shape == result_shape:
        return images_or_image
    else:
        return images_or_image.reshape(result_shape)


def to_NHW(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    N, H, W, C = nshape(images_or_image)
    assert C == 1, 'C(%s) should be 1' % C

    result_shape = (N, H, W)
    if images_or_image.shape == result_shape:
        return images_or_image
    else:
        return images_or_image.reshape(result_shape)
