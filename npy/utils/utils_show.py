import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imresize
from .utils_image import rescale_img, assure_image_dtype, assure_color_image, img_shape


def flatten_imgs(imgs, show_shape):
    num_img = np.prod(show_shape)

    if isinstance(imgs, list):
        imgs = np.array(imgs)

    for i in range(len(imgs.shape)):
        if num_img == np.prod(imgs.shape[:i]):
            img_shape = imgs.shape[i:]
            new_shape = (num_img,) + img_shape
            return np.reshape(imgs, new_shape)

    else:
        raise ValueError('Cannot distinguish images. imgs shape: %s, show_shape: %s' % (imgs.shape, show_shape))


def merge_imgs(imgs, show_shape, order='row'):
    """

    :param np.ndarray imgs:
    :param tuple show_shape:
    :param str order:
    :return:
    """
    imgs = flatten_imgs(imgs, show_shape)
    H, W, C = img_shape(imgs)
    I, J = show_shape
    result = np.zeros((I * H, J * W, C), dtype=imgs.dtype)

    for k, img in enumerate(imgs):
        if order == 'row':
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        result[i * H: (i + 1) * H, j * W: (j + 1) * W] = np.squeeze(img)

    return result


def showimg(image, ax=None, min_a=None, max_a=None, inverse=True):
    if ax is None:
        ax = plt.subplot()

    if image.shape[-1] == 1:
        is_gray = True
        image = np.squeeze(image, -1)
    elif len(image.shape) == 2:
        is_gray = True
    else:
        is_gray = False

    if max_a is None:
        max_a = image.max()
    if min_a is None:
        min_a = image.min()

    image = rescale_img(image, min_from=min_a, max_from=max_a, dtype='uint8')
    if is_gray:
        if inverse:
            ax.imshow(image, cmap='gray_r')
        else:
            ax.imshow(image, cmap='gray')

    else:
        ax.imshow(image)

    ax.set_axis_off()
    return ax


def showimgs(imgs, title=None):
    num_img = len(imgs)
    fig, ax = plt.subplots(1, num_img, figsize=(num_img, 1))

    for i, img in enumerate(imgs):
        ax[i].set_axis_off()
        showimg(img, ax[i])

    if title is not None:
        plt.title(title)
    return fig


def saveimg(img, path, min_a=-1, max_a=1):
    fig, ax = plt.subplots()
    showimg(img, ax, min_a=min_a, max_a=max_a)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def saveimgs(imgs, title, path, min_a=-1, max_a=1):
    imgs = rescale_img(imgs, min_from=min_a, max_from=max_a, dtype='uint8')
    fig = showimgs(imgs, title)
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)


def resize_batch(imgs, new_shape):
    new_shapes = (imgs.shape[0],) + tuple(new_shape)
    result = np.zeros(new_shapes, dtype=imgs.dtype)
    for i, img in enumerate(imgs):
        result[i] = imresize(img, new_shape)
    return result


def show_image(ax, data, title='', show_axis=False, interpolation=None):
    if data is None:
        return

    # 1. Channel : Color or Gray?
    data = assure_color_image(data)

    # 2. Dtype: To Float[0, 1] or uint8[0, 255]
    data = assure_image_dtype(data)

    if data is not None:
        ax.imshow(data, interpolation=interpolation)

    if title:
        ax.set_title(title)

    if not show_axis:
        ax.set_axis_off()
