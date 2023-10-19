import matplotlib.pyplot as plt
from ..utils import multirank

__all__ = ['tsne', 'plot_tsne']


def tsne(X, **kwargs):
    from MulticoreTSNE import MulticoreTSNE as TSNE
    model = TSNE(**kwargs)

    if multirank.is_multirank(X):
        Ns = multirank.recursive.get_lengths(X)
        X = multirank.flatten_vectors(X)

        Y = model.fit_transform(X)
        Y = multirank.unflatten_vectors(Y, Ns)

    else:
        Y = model.fit_transform(X)
    return Y


def plot_tsne(X, n=4, ax=None, s=10, **kwargs):  # [C, N, D]
    emb = tsne(X, n_jobs=n)

    if ax is None:
        fig, ax = plt.subplots()

    if multirank.is_multirank(X):
        C = len(emb)
        for c in range(C):
            print(c, len(emb[c]))

            _kwargs = {k: v[c] if isinstance(v, list) else v for k, v in kwargs.items()}
            ax.scatter(*emb[c].T, s=s, **_kwargs)
        return ax

    else:
        ax.scatter(*emb.T, s=s, **kwargs)

    return ax
