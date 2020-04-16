from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from npy import task, load_binary
from .multirank.modify import flatten_vectors, unflatten_vectors
from .multirank.recursive import get_lengths


__all__ = ['tsne_multirank', 'tsne']


def tsne(X, **kwargs):
    model = TSNE(**kwargs)
    Y = model.fit_transform(X)
    return Y


def tsne_multirank(X, **kwargs):
    Ns = get_lengths(X)
    X = flatten_vectors(X)

    Y = tsne(X, **kwargs)
    Y = unflatten_vectors(Y, Ns)
    return Y


def draw_tsne(X, n=4):  # [C, N, D]
    emb = tsne_multirank(X, n_jobs=n)
    C = len(emb)

    for c in range(C):
        print(c, len(emb[c]))
        plt.scatter(*emb[c].T, s=10)


def test():
    X_cls0 = np.zeros((14, 60))
    X_cls1 = np.zeros((16, 60)) + 1
    X_cls2 = np.zeros((20, 60)) + 2

    X = [X_cls0, X_cls1, X_cls2]
    draw_tsne(X)
    plt.show()
