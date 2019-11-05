import numpy as np
from functools import partial
from ..constants import *


__all__ = ['shape', 'nshape', 'iscolor', 'assure_color_image', 'get_fmt', 'to_fmt',
           'to_NHWC', 'to_HWC', 'to_NHW', 'to_HW', 'match_fmt']


def raise_unexpected_shape(shape_):
    raise ValueError(f'Unexpected shape: {shape_}')


def raise_unexpected_fmt(fmt):
    raise ValueError(f'Unexpected fmt: {fmt}')


def shape(images_or_image) -> tuple:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    shape_ = images_or_image.shape[-3:]
    if len(shape_) <= 1:
        raise_unexpected_shape(shape_)

    elif len(shape_) == 2:  # (H, W)
        H, W = shape_
        return H, W, 1

    elif len(shape_) == 3:
        s1, s2, s3 = shape_
        if s3 in [1, 3, 4]:  # (H, W, C)
            return s1, s2, s3
        else:  # (N, H, W)
            return s2, s3, 1

    else:
        raise_unexpected_shape(shape_)


def nshape(images_or_image) -> tuple:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    H, W, C = shape(images_or_image)
    N = int(np.prod(images_or_image.shape) / (H * W * C))
    return N, H, W, C


def iscolor(images_or_image) -> bool:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    H, W, C = shape(images_or_image)
    return C == 3


def assure_color_image(image):
    def gray2rgb(image_):
        return np.repeat(image_, 3, axis=-1)

    shape_ = image.shape
    if len(shape_) == 4:  # [N, H, W, C]
        C = image.shape[-1]
        if C == 1:  # [N, H, W, 1]
            return gray2rgb(image)
        elif C == 3:  # [N, H, W, 3]
            return image
        else:
            raise_unexpected_shape(image.shape)

    elif len(shape_) == 3:
        if shape_[-1] == 3:  # [H, W, 3]
            return image
        elif shape_[-1] == 1:  # [H, W, 1]
            return gray2rgb(image)
        else:  # [N, H, W]
            return gray2rgb(np.expand_dims(image, axis=-1))

    elif len(shape_) == 2:  # [H, W]
        return gray2rgb(np.expand_dims(image, axis=-1))

    else:
        raise_unexpected_shape(image.shape)


def get_fmt(images_or_image) -> str:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    N, H, W, C = nshape(images_or_image)
    shape_ = images_or_image.shape

    if shape_ == (N, H, W, C):
        return NHWC
    elif shape_ == (H, W):
        return HW
    elif shape_ == (H, W, C):
        return HWC
    elif shape_ == (N, H, W):
        return NHW
    else:
        raise_unexpected_shape(shape_)


def to_fmt(images_or_image, fmt) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param str fmt:
    :return:
    """
    N, H, W, C = nshape(images_or_image)

    if fmt == NHWC:
        result_shape = (N, H, W, C)
    elif fmt == HWC:
        result_shape = (H, W, C)
    elif fmt == NHW:
        result_shape = (N, H, W)
    elif fmt == HW:
        result_shape = (H, W)
    else:
        raise_unexpected_fmt(fmt)

    return images_or_image.reshape(result_shape)


def match_fmt(images_or_image, fmt_images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param np.ndarray fmt_images_or_image:
    :return:
    """
    fmt = get_fmt(fmt_images_or_image)
    return to_fmt(images_or_image, fmt)


to_NHWC = partial(to_fmt, fmt='NHWC')
to_HWC = partial(to_fmt, fmt='HWC')
to_NHW = partial(to_fmt, fmt='NHW')
to_HW = partial(to_fmt, fmt='HW')
