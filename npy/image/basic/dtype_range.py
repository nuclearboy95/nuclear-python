import numpy as np
from typing import Type
from ..transform import rescale
from ...constants import *
from ...errors import *


__all__ = ['assure_dtype', 'assure_dtype_float32', 'assure_dtype_uint8',
           'to_float32', 'to_uint8']


def get_dtyperange(images_or_image) -> Type[NPYDtypeRange]:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    max_v = images_or_image.max()
    min_v = images_or_image.min()
    dtype = images_or_image.dtype
    return dtyperange_of_minmax(dtype, min_v, max_v)


def dtyperange_of_minmax(dtype, min_v, max_v) -> Type[NPYDtypeRange]:
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
                raise UnknownImageFloatRangeError(min_v, max_v)

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
                raise UnknownImageFloatRangeError(min_v, max_v)
        else:
            raise UnknownImageFloatRangeError(min_v, max_v)

    elif dtype in [np.uint8]:
        return UINT8_0_255

    else:
        raise UnknownImageDtypeError(dtype)


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
        raise UnknownImageDtypeError(images_or_image.dtype)


def assure_dtype_float32(images_or_image, min_to=0., max_to=1.) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param float min_to:
    :param float max_to:
    :return:
    """
    imagetype_from = get_dtyperange(images_or_image)
    imagetype_to = dtyperange_of_minmax(np.float32, min_to, max_to)

    min_from, max_from = imagetype_from.get_minmax()
    min_to, max_to = imagetype_to.get_minmax()

    return rescale(images_or_image,
                   min_from=min_from, max_from=max_from, min_to=min_to, max_to=max_to,
                   dtype=np.float32)


def assure_dtype_uint8(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    imagetype_from = get_dtyperange(images_or_image)
    imagetype_to = dtyperange_of_minmax(np.uint8, 0, 255)

    min_from, max_from = imagetype_from.get_minmax()
    min_to, max_to = imagetype_to.get_minmax()

    return rescale(images_or_image,
                   min_from=min_from, max_from=max_from, min_to=min_to, max_to=max_to,
                   dtype=np.uint8)


to_float32 = assure_dtype_float32
to_uint8 = assure_dtype_uint8
