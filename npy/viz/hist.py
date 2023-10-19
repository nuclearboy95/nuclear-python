import numpy as np
import matplotlib.pyplot as plt
from ..utils import multirank, index_l_of_d

__all__ = ['plot', 'plot_std_of_each_dim', 'plot_norm_of_each_vec']


def plot(X, ax=None, bins=50, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    if multirank.is_matrix(X):  # X [C, N]
        C = len(X)
        for c in range(C):
            _kwargs = index_l_of_d(kwargs, c)
            ax.hist(X[c], bins=bins, **_kwargs)

    else:  # X [N]
        ax.hist(X, bins=bins, **kwargs)
    return ax


def plot_std_of_each_dim(X, ax=None, bins=50, **kwargs):
    if multirank.is_multirank(X):  # X [C, N, D]
        C = len(X)
        stds = [np.std(X[c], axis=0) for c in range(C)]
    else:  # X [C, N]
        stds = np.std(X, axis=0)

    return plot(stds, ax=ax, bins=bins, **kwargs)


def plot_norm_of_each_vec(X, ax=None, bins=50, **kwargs):
    if multirank.is_multirank(X):  # X [C, N, D]
        C = len(X)
        norms = [np.linalg.norm(X[c], axis=-1) for c in range(C)]
    else:  # X [C, N]
        norms = np.linalg.norm(X, axis=-1)

    return plot(norms, ax=ax, bins=bins, **kwargs)
