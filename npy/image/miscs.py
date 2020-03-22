import numpy as np
from .basic import shape
import io
import imageio
from math import ceil
from itertools import product
from ..calc import num_segments


__all__ = ['flatten_image_list', 'merge', 'image_to_bytes', 'to_patches', 'to_patch']


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

    for k, image in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        target_shape = result[i * H: (i + 1) * H, j * W: (j + 1) * W].shape
        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = image.reshape(target_shape)

    return result


def image_to_bytes(image, format='png'):
    buf = io.BytesIO()
    imageio.imsave(buf, image, format=format)
    return buf


def to_patch(image, K, S=1, HO=0, WO=0, result=None, return_indexes=False, strict=False):
    H, W, C = shape(image)

    NH = num_segments(H - K + 1 - HO, S, strict=strict)
    NW = num_segments(W - K + 1 - WO, S, strict=strict)
    N = NH * NW

    result_shape = (N, K, K, C)
    Hs = np.empty([N], dtype=int)
    Ws = np.empty([N], dtype=int)

    if result is None:
        result = np.empty(result_shape, dtype=image.dtype)

    else:
        if result_shape != result.shape:
            raise ValueError()

    if strict:
        from ..utils import range_strict
        r = range_strict
    else:
        r = range

    for i, (h, w) in enumerate(product(r(HO, H - K + 1, S),
                                       r(WO, W - K + 1, S))):
        result[i] = image[h: h + K, w: w + K].reshape(result[i].shape)
        Hs[i] = h
        Ws[i] = w

    if return_indexes:
        return result, Hs, Ws
    else:
        return result


def to_patches(images, K, S=1, HO=0, WO=0):
    patches = [to_patch(image, K, S, HO, WO) for image in images]
    return np.concatenate(patches, axis=0)
