import numpy as np


def shape(image):
    shape_ = image.shape[-3:]
    if len(shape_) <= 1:
        raise ValueError('Unexpected shape: {}'.format(shape_))

    elif len(shape_) == 2:
        H, W = shape_
        return H, W, 1

    elif len(shape_) == 3:
        H, W, C = shape_
        if C in [1, 3]:
            return H, W, C
        else:
            raise ValueError('Unexpected shape: {}'.format(shape_))

    else:
        raise ValueError('Unexpected shape: {}'.format(shape_))


def nshape(image):
    H, W, C = shape(image)
    N = int(np.prod(image.shape) / (H * W * C))
    return N, H, W, C


def iscolor(img):
    return shape(img)[-1] == 3


def assure_color_image(image):
    def raise_unknown_image_shape():
        raise ValueError('Unknown image shape. Shape: {}'.format(image.shape))

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
            raise_unknown_image_shape()

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
        raise_unknown_image_shape()
