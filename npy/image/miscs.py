import numpy as np
from .basic import shape


def flatten_image_list(images, show_shape) -> np.ndarray:
    """

    :param images:
    :param tuple show_shape:
    :return:
    """
    N = np.prod(show_shape)

    if isinstance(images, list):
        images = np.array(images)

    for i in range(len(images.shape)):  # find axis.
        if N == np.prod(images.shape[:i]):
            img_shape = images.shape[i:]
            new_shape = (N,) + img_shape
            return np.reshape(images, new_shape)

    else:
        raise ValueError('Cannot distinguish images. imgs shape: %s, show_shape: %s' % (images.shape, show_shape))


def merge(images, show_shape, order='row') -> np.ndarray:
    """

    :param np.ndarray images:
    :param tuple show_shape:
    :param str order:

    :return:
    """
    images = flatten_image_list(images, show_shape)
    H, W, C = shape(images)
    I, J = show_shape
    result = np.zeros((I * H, J * W, C), dtype=images.dtype)

    for k, img in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = np.squeeze(img)

    return result
