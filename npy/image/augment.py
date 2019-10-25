import numpy as np


__all__ = ['rotation', 'horizontal_flip']


def rotation(images) -> np.ndarray:
    from npy.image import nshape
    N, H, W, C = nshape(images)
    result = np.empty((N * 4, H, W, C), dtype=images.dtype)
    result[::4] = images
    result[1::4] = np.rot90(images, k=1, axes=(1, 2))
    result[2::4] = np.rot90(images, k=2, axes=(1, 2))
    result[3::4] = np.rot90(images, k=3, axes=(1, 2))

    return result


def horizontal_flip(images) -> np.ndarray:
    from npy.image import nshape
    N, H, W, C = nshape(images)
    result = np.empty((N * 2, H, W, C), dtype=images.dtype)
    result[::2] = images
    result[1::2] = images[:, :, ::-1, :]

    return result
