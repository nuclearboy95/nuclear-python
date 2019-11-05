import numpy as np
from .transform import rescale
from ..constants import *


__all__ = ['assure_dtype', 'assure_dtype_float32', 'assure_dtype_uint8',
           'to_float32', 'to_uint8']


def raise_unknown_float_image(min_v, max_v):
    raise ValueError(f'Unknown float image range. Min: {min_v}, Max: {max_v}')


def raise_unknown_image_dtype(dtype):
    raise ValueError(f'Unknown image dtype: {dtype}')


def raise_unknown_imagetype(imagetype):
    raise ValueError(f'Unknown imagetype: {imagetype}')


#############################################

def get_imagetype(images_or_image) -> str:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    max_v = images_or_image.max()
    min_v = images_or_image.min()
    dtype = images_or_image.dtype
    return imagetype_of_minmax(dtype, min_v, max_v)


def minmax_of_imagetype(imagetype) -> tuple:
    """

    :param str imagetype:
    :return:
    """

    if imagetype in [FLOAT32_0_1, FLOAT64_0_1, FLOAT32_m1_1, FLOAT64_m1_1]:
        max_v = 1.
    elif imagetype in [FLOAT32_0_255, FLOAT64_0_255, FLOAT32_m256_255, FLOAT64_m256_255]:
        max_v = 255.
    elif imagetype in [UINT8_0_255]:
        max_v = 255
    else:
        raise_unknown_imagetype(imagetype)

    if imagetype in [FLOAT32_0_1, FLOAT64_0_1, FLOAT32_0_255, FLOAT64_0_255]:
        min_v = 0.
    elif imagetype in [FLOAT32_m256_255, FLOAT64_m256_255]:
        min_v = -256.
    elif imagetype in [FLOAT32_m1_1, FLOAT64_m1_1]:
        min_v = -1.
    elif imagetype in [UINT8_0_255]:
        min_v = 0
    else:
        raise_unknown_imagetype(imagetype)

    return min_v, max_v


def imagetype_of_minmax(dtype, min_v, max_v) -> str:
    if dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1)
                if dtype == np.float32:
                    return FLOAT32_0_1
                else:
                    return FLOAT64_0_1

            elif -1 <= min_v <= 0:  # [-1, 1)
                if dtype == np.float32:
                    return FLOAT32_m1_1
                else:
                    return FLOAT64_m1_1
            else:
                raise_unknown_float_image(min_v, max_v)

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                if dtype == np.float32:
                    return FLOAT32_0_255
                else:
                    return FLOAT64_0_255

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                if dtype == np.float32:
                    return FLOAT32_m256_255
                else:
                    return FLOAT64_m256_255
            else:
                raise_unknown_float_image(min_v, max_v)
        else:
            raise_unknown_float_image(min_v, max_v)

    elif dtype in [np.uint8]:
        return UINT8_0_255

    else:
        raise_unknown_image_dtype(dtype)


#############################################


def assure_dtype(images_or_image, min_to=0, max_to=1) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param min_to:
    :param max_to:
    :return:
    """
    if images_or_image.dtype in [np.float32, np.float64]:
        return assure_dtype_float32(images_or_image, min_to, max_to)

    elif images_or_image.dtype in [np.uint8]:
        return assure_dtype_uint8(images_or_image)

    else:
        raise_unknown_image_dtype(images_or_image.dtype)


def assure_dtype_float32(images_or_image, min_to=0., max_to=1.) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param float min_to:
    :param float max_to:
    :return:
    """
    imagetype_from = get_imagetype(images_or_image)
    imagetype_to = imagetype_of_minmax(np.float32, min_to, max_to)

    min_from, max_from = minmax_of_imagetype(imagetype_from)
    min_to, max_to = minmax_of_imagetype(imagetype_to)

    return rescale(images_or_image,
                   min_from=min_from, max_from=max_from, min_to=min_to, max_to=max_to,
                   dtype=np.float32)


def assure_dtype_uint8(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    imagetype_from = get_imagetype(images_or_image)
    imagetype_to = imagetype_of_minmax(np.uint8, 0, 255)

    min_from, max_from = minmax_of_imagetype(imagetype_from)
    min_to, max_to = minmax_of_imagetype(imagetype_to)

    return rescale(images_or_image,
                   min_from=min_from, max_from=max_from, min_to=min_to, max_to=max_to,
                   dtype=np.uint8)


to_float32 = assure_dtype_float32
to_uint8 = assure_dtype_uint8
