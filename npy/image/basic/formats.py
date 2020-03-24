import numpy as np
from functools import partial
from .shapes import *
from ...constants import *
from ...errors import *

__all__ = ['get_fmt', 'to_fmt', 'to_NHWC', 'to_HWC', 'to_NHW', 'to_HW', 'match_fmt']


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
        raise UnknownImageShapeError(shape_)


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
        raise UnknownImageFormatError(fmt)

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
