import matplotlib.pyplot as plt
from .utils_numbers import max_n, ceil_to_1
from .utils_show import show_image
from matplotlib import ticker


__all__ = ['show_heatmap']


def show_heatmap(ax, data, title='', colorbar=True, percentile=1.,
                 nonnegative=False, thres=1e-5, vmax=None, bg=None, bg_alpha=0.6,
                 cmap=None, cb_legend=False):
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
    top = ceil_to_1(top)
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
        show_image(ax, bg)
    else:
        bg_alpha = None
    mappable = ax.imshow(data, cmap=cmap, interpolation='nearest', vmax=vmax, vmin=vmin, alpha=bg_alpha)
    if colorbar:
        cb = plt.colorbar(mappable, ax=ax, fraction=0.045, pad=0.04)
        cb.formatter.set_powerlimits((-1, 1))
        cb.locator = ticker.MaxNLocator(nbins=2)
        cb.ax.tick_params(labelsize=10)
        cb.ax.yaxis.get_offset_text().set(size=8)
        cb.update_ticks()
        if not cb_legend:
            cb.ax.set_axis_off()

