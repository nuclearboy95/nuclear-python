import numpy as np
from ...errors import *

__all__ = ['shape', 'nshape', 'iscolor', 'assure_color_image']


def shape(images_or_image) -> tuple:
    """
    :param np.ndarray images_or_image:
    :return:
    """
    shape_ = images_or_image.shape[-3:]
    if len(shape_) <= 1:
        raise UnknownImageShapeError(shape_)

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
        raise UnknownImageShapeError(shape_)


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


def assure_color_image(image):  # FIXME simplify the logic
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
            raise UnknownImageShapeError(image.shape)

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
        raise UnknownImageShapeError(image.shape)
