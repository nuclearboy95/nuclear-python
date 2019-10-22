import numpy as np
from .transform import rescale


__all__ = ['assure_dtype', 'assure_dtype_float32', 'assure_dtype_uint8',
           'to_float32', 'to_uint8']


def assure_dtype(image, min_to=0, max_to=1):
    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(image.dtype))

    if image.dtype in [np.float32, np.float64]:
        return assure_dtype_float32(image, min_to, max_to)

    elif image.dtype in [np.uint8]:
        return assure_dtype_uint8(image)

    else:
        raise_unknown_image_dtype()


def assure_dtype_float32(images_or_image, min_to=0., max_to=1.) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :param float min_to:
    :param float max_to:
    :return:
    """
    img = images_or_image

    def raise_unknown_float_image():
        raise ValueError('Unknown float image range. Min: {}, Max: {}'.format(min_v, max_v))

    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(img.dtype))

    max_v = img.max()
    min_v = img.min()
    if img.dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1) (Normal)
                return img.astype(np.float32)

            elif -1 <= min_v <= 0:  # Presumably [-1, 1)
                return rescale(img, min_from=-1, max_from=1,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')
            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                return rescale(img, min_from=0, max_from=255,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                return rescale(img, min_from=-256, max_from=255,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')
            else:
                raise_unknown_float_image()
        else:
            raise_unknown_float_image()

    elif img.dtype in [np.uint8]:
        return rescale(img, min_from=0, max_from=255,
                       min_to=min_to, max_to=max_to,
                       dtype='float32')

    else:
        raise_unknown_image_dtype()


def assure_dtype_uint8(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    img = images_or_image

    def raise_unknown_float_image():
        raise ValueError('Unknown float image range. Min: {}, Max: {}'.format(min_v, max_v))

    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(img.dtype))

    max_v = img.max()
    min_v = img.min()
    if img.dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1)
                min_v, max_v = 0, 1

            elif -1 <= min_v <= 0:  # Presumably [-1, 1)
                min_v, max_v = -1, 1

            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                min_v, max_v = 0, 255

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                min_v, max_v = -256, 255

            else:
                raise_unknown_float_image()

        else:
            raise_unknown_float_image()

        return rescale(img,
                       min_from=min_v, max_from=max_v,
                       min_to=0, max_to=255,
                       dtype='uint8')

    elif img.dtype in [np.uint8]:
        return img

    else:
        raise_unknown_image_dtype()


def to_float32(images_or_image) -> np.ndarray:
    """

    :param np.ndarray images_or_image:
    :return:
    """
    return assure_dtype_float32(images_or_image)


def to_uint8(images_or_image) -> np.ndarray:
    return assure_dtype_uint8(images_or_image)
