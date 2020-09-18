import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import PIL
import io
from ..utils import max_n
from ..calc import ceil_to_1digit
from .basic import nshape, shape, assure_dtype
from .miscs import flatten_image_list, merge

__all__ = [
    'show_heatmap', 'show', 'shows', 'shows_merged',
    'plot_to_image', 'blank'
]


def plot_to_image(figure) -> np.ndarray:
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = np.array(PIL.Image.open(buf))
    return image


def show(data=None, ax=None, title='', show_axis=False, interpolation=None, inverse=False):
    if data is None:
        if ax is not None:
            ax.set_axis_off()
        return

    if ax is None:
        fig, ax = plt.subplots()

    # 2. Dtype: To Float[0, 1] or uint8[0, 255]
    data = assure_dtype(data)

    H, W, C = shape(data)

    if data is not None:
        if C == 1:
            cmap = 'gray_r' if inverse else 'gray'
            ax.imshow(data, interpolation=interpolation, cmap=cmap)

        else:
            ax.imshow(data, interpolation=interpolation)

    if title:
        ax.set_title(title)

    if not show_axis:
        ax.set_axis_off()

    return ax


def blank(ax, show_axis=False):
    ax.imshow(np.full((256, 256), 255, dtype=np.uint8), cmap='gray', vmax=255, vmin=0)
    if show_axis:
        ax.set_yticks([])
        ax.set_xticks([])

    else:
        ax.set_axis_off()


def shows(images, show_shape=None, order='row'):
    """
    Show multiple images in splitted axes.

    :param np.ndarray images:
    :param tuple show_shape:
    :param str order:
    :return:
    """
    if isinstance(images, list):
        images = np.asarray(images)

    N, H, W, C = nshape(images)
    if show_shape is None:
        show_shape = (N, 1)

    I, J = show_shape
    images = flatten_image_list(images, show_shape)
    fig, axes = plt.subplots(I, J, figsize=(J, I))

    for k, image in enumerate(images):
        if order.lower().startswith('row'):
            i = k // J
            j = k % J
        else:
            i = k % I
            j = k // I

        if I != 1 and J != 1:
            ax = axes[i][j]
        elif I == 1:
            ax = axes[j]
        else:
            ax = axes[i]

        show(data=image, ax=ax)

    return fig


def shows_merged(images, ax=None, show_shape=None, order='row', title=None):
    """
    Show multiple images in a single axis.

    :param np.ndarray images:
    :param ax:
    :param tuple show_shape:
    :param str order:
    :param str title:
    :return:
    """
    if isinstance(images, list):
        images = np.asarray(images)

    N, H, W, C = nshape(images)
    if show_shape is None:
        show_shape = (N, 1)

    image = merge(images, show_shape, order=order)
    return show(data=image, ax=ax, title=title)


def show_heatmap(ax, data, title='', colorbar=True, percentile=1.,
                 nonnegative=False, thres=1e-5, vmax=None, bg=None, bg_alpha=0.6,
                 cmap=None, cb_legend=False, interpolation='nearest'):
    """

    :param matplotlib.axes._subplots.AxesSubplot ax:
    :param np.ndarray data:
    :param str title:
    :param bool colorbar:
    :param float percentile:
    :param bool nonnegative:
    :param float thres:
    :param float vmax:
    :param np.ndarray bg:
    :param float bg_alpha:
    :param str cmap:
    :param bool cb_legend:
    :param str interpolation:
    :return:
    """
    if ax is None:
        ax = plt.subplot()
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    if data is None:
        return

    top = max_n(abs(data), percentile)
    # top = nmax(top)
    top = ceil_to_1digit(top)
    top = max(top, thres)

    if vmax is None:
        vmax = top
    if nonnegative:
        vmin = 0
        cmap = cmap or 'gray'
    else:
        vmin = -vmax
        cmap = cmap or 'RdBu_r'

    if bg is not None:
        show(data=bg, ax=ax)
    else:
        bg_alpha = None
    mappable = ax.imshow(data, cmap=cmap, interpolation=interpolation, vmax=vmax, vmin=vmin, alpha=bg_alpha)
    if colorbar:
        cb = plt.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04)
        cb.formatter.set_powerlimits((-1, 1))
        cb.locator = ticker.MaxNLocator(nbins=2)
        cb.ax.tick_params(labelsize=10)
        cb.ax.yaxis.get_offset_text().set(size=8)
        cb.update_ticks()
        if not cb_legend:
            cb.ax.set_axis_off()
