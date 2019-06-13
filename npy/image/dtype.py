import numpy as np
from .transform import rescale


def assure_image_dtype(image, min_to=0, max_to=1):
    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(image.dtype))

    if image.dtype in [np.float32, np.float64]:
        return assure_image_dtype_float32(image, min_to, max_to)

    elif image.dtype in [np.uint8]:
        return assure_image_dtype_uint8(image)

    else:
        raise_unknown_image_dtype()


def assure_image_dtype_float32(image, min_to=0, max_to=1):
    def raise_unknown_float_image():
        raise ValueError('Unknown float image range. Min: {}, Max: {}'.format(min_v, max_v))

    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(image.dtype))

    max_v = image.max()
    min_v = image.min()
    if image.dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1) (Normal)
                return image
            elif -1 <= min_v <= 0:  # Presumably [-1, 1)
                return rescale(image, min_from=-1, max_from=1,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')
            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                return rescale(image, min_from=0, max_from=255,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                return rescale(image, min_from=-255, max_from=255,
                               min_to=min_to, max_to=max_to,
                               dtype='float32')
            else:
                raise_unknown_float_image()
        else:
            raise_unknown_float_image()

    elif image.dtype in [np.uint8]:
        return rescale(image, min_from=0, max_from=255,
                       min_to=min_to, max_to=max_to,
                       dtype='float32')

    else:
        raise_unknown_image_dtype()


def assure_image_dtype_uint8(image):
    def raise_unknown_float_image():
        raise ValueError('Unknown float image range. Min: {}, Max: {}'.format(min_v, max_v))

    def raise_unknown_image_dtype():
        raise ValueError('Unknown image dtype: {}'.format(image.dtype))

    max_v = image.max()
    min_v = image.min()
    if image.dtype in [np.float32, np.float64]:
        if 0 <= max_v <= 1:
            if 0 <= min_v <= 1:  # [0, 1) (Normal)
                return image
            elif -1 <= min_v <= 0:  # Presumably [-1, 1)
                return rescale(image, min_from=-1, max_from=1,
                               min_to=0, max_to=255,
                               dtype='uint8')
            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                return rescale(image, min_from=0, max_from=255,
                               min_to=0, max_to=255,
                               dtype='uint8')

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                return rescale(image, min_from=-256, max_from=255,
                               min_to=0, max_to=255,
                               dtype='uint8')
            else:
                raise_unknown_float_image()
        else:
            raise_unknown_float_image()

    elif image.dtype in [np.uint8]:
        return image

    else:
        raise_unknown_image_dtype()


def ij_range(H, W, K=1, S=1, O=0):
    """

    :param int H:
    :param int W:
    :param int K:
    :param int S:
    :param int O:
    :return:
    """

    for i in range(O, H - K + 1, S):
        for j in range(O, W - K + 1, S):
            yield i, j
