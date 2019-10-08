import numpy as np
from ..constants import MEAN_IMAGENET, STD_IMAGENET
from PIL import Image
from .basic import shape


__all__ = ['pad', 'preprocess_imagenet', 'unpreprocess_imagenet', 'resize_imagenet', 'rescale',
           'normalize', 'resize', 'scale', 'resizes', 'rgb2gray']


def pad(images, K, shape=None):
    if shape is None:
        shape = images.shape[1:]
    pad_width = ((0, 0), (K, K), (K, K))
    if len(shape) == 3:
        pad_width += ((0, 0),)

    return np.pad(images, pad_width, mode='constant')


def preprocess_imagenet(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= np.array(MEAN_IMAGENET)
    x /= np.array(STD_IMAGENET)
    return x


def unpreprocess_imagenet(x):
    x *= np.array(STD_IMAGENET)
    x += np.array(MEAN_IMAGENET)
    x *= 255.
    np.clip(x, 0, 255, out=x)
    x = x.astype(np.uint8)
    return x


def resize_imagenet(img):
    # crop imgs to 224 x 224
    H, W = img.shape[:2]
    if (H, W) == (224, 224):
        return img
    scale = 1.15
    r = max(224. / H, 224. / W)
    Hp = int(H * r * scale)
    Wp = int(W * r * scale)
    img2 = resize(img, (Hp, Wp))

    Hcut = Hp - 224
    Wcut = Wp - 224

    Htop = Hcut // 2
    Wleft = Wcut // 2

    img2 = img2[Htop: Htop + 224, Wleft: Wleft + 224, ...]
    return img2


def rescale(img, min_from=-1, max_from=1, min_to=0, max_to=255, dtype='float32'):
    len_from = max_from - min_from
    len_to = max_to - min_to
    img = (img.astype(np.float32) - min_from) * len_to / len_from + min_to
    return img.astype(dtype)


def normalize(img):
    result = img.copy()
    result -= result.min()
    result /= result.max()
    return result


def resize(image, shape):
    return np.array(Image.fromarray(image).resize(shape[::-1]))


def resizes(images, new_shape):
    new_shapes = (images.shape[0],) + tuple(new_shape)
    result = np.zeros(new_shapes, dtype=images.dtype)
    for i, img in enumerate(images):
        result[i] = resize(img, new_shape)
    return result


def scale(image, s):
    H, W = shape(image)[:2]
    h = int(H * s)
    w = int(W * s)
    return resize(image, (h, w))


def rgb2gray(images):
    H, W, C = shape(images)
    if C == 1:
        return images

    elif C == 3:
        R, G, B = images[..., 0], images[..., 1], images[..., 2]
        return 0.2989 * R + 0.5870 * G + 0.1140 * B

    else:
        raise ValueError()

