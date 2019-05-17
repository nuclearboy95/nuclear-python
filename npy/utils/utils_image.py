import numpy as np
from scipy.misc import imresize
from ..constants import MEAN_IMAGENET, STD_IMAGENET


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


def img_shape(img):
    shape = img.shape[-3:]
    if len(shape) <= 1:
        raise ValueError('Unexpected shape: {}'.format(shape))

    elif len(shape) == 2:
        H, W = shape
        return H, W, 1

    elif len(shape) == 3:
        H, W, C = shape
        if C in [1, 3]:
            return H, W, C
        else:
            raise ValueError('Unexpected shape: {}'.format(shape))

    else:
        raise ValueError('Unexpected shape: {}'.format(shape))


def pad(images, K, shape=None):
    if shape is None:
        shape = images.shape[1:]
    pad_width = ((0, 0), (K, K), (K, K))
    if len(shape) == 3:
        pad_width += ((0, 0),)

    return np.pad(images, pad_width, mode='constant')


def iscolor(img):
    return img_shape(img)[-1] == 3


def resize_imagenet(img):
    # crop imgs to 224 x 224
    H, W = img.shape[:2]
    if (H, W) == (224, 224):
        return img
    scale = 1.15
    r = max(224. / H, 224. / W)
    Hp = int(H * r * scale)
    Wp = int(W * r * scale)
    img2 = imresize(img, (Hp, Wp))

    Hcut = Hp - 224
    Wcut = Wp - 224

    Htop = Hcut // 2
    Wleft = Wcut // 2

    img2 = img2[Htop: Htop + 224, Wleft: Wleft + 224, ...]
    return img2


def normalized(img):
    result = img.copy()
    result -= result.min()
    result /= result.max()
    return result


def rescale_img(img, min_from=-1, max_from=1, min_to=0, max_to=255, dtype='float32'):
    len_from = max_from - min_from
    len_to = max_to - min_to
    return ((img.astype(np.float32) - min_from) * len_to / len_from + min_to).astype(dtype)


def assure_color_image(image):
    def raise_unknown_image_shape():
        raise ValueError('Unknown image shape. Shape: {}'.format(image.shape))

    def gray2rgb(image_):
        return np.repeat(image_, 3, axis=-1)

    shape = image.shape
    if len(shape) == 4:  # [N, H, W, C]
        C = image.shape[-1]
        if C == 1:  # [N, H, W, 1]
            return gray2rgb(image)
        elif C == 3:  # [N, H, W, 3]
            return image
        else:
            raise_unknown_image_shape()

    elif len(shape) == 3:
        if shape[-1] == 3:  # [H, W, 3]
            return image
        elif shape[-1] == 1:  # [H, W, 1]
            return gray2rgb(image)
        else:  # [N, H, W]
            return gray2rgb(np.expand_dims(image, axis=-1))

    elif len(shape) == 2:  # [H, W]
        return gray2rgb(np.expand_dims(image, axis=-1))

    else:
        raise_unknown_image_shape()


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
                return rescale_img(image, min_from=-1, max_from=1,
                                   min_to=min_to, max_to=max_to,
                                   dtype='float32')
            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                return rescale_img(image, min_from=0, max_from=255,
                                   min_to=min_to, max_to=max_to,
                                   dtype='float32')

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                return rescale_img(image, min_from=-255, max_from=255,
                                   min_to=min_to, max_to=max_to,
                                   dtype='float32')
            else:
                raise_unknown_float_image()
        else:
            raise_unknown_float_image()

    elif image.dtype in [np.uint8]:
        return rescale_img(image, min_from=0, max_from=255,
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
                return rescale_img(image, min_from=-1, max_from=1,
                                   min_to=0, max_to=255,
                                   dtype='uint8')
            else:
                raise_unknown_float_image()

        elif 0 <= max_v <= 255:
            if 0 <= min_v <= 255:  # Presumably [0, 255)
                return rescale_img(image, min_from=0, max_from=255,
                                   min_to=0, max_to=255,
                                   dtype='uint8')

            elif -256 <= min_v <= 0:  # Presumably [-256, 255)
                return rescale_img(image, min_from=-256, max_from=255,
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
