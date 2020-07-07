import numpy as np


__all__ = ['calc_id']


def get_NN_1_and_2(vecs) -> np.ndarray:
    from sklearn.neighbors import KDTree
    kdt = KDTree(vecs)
    dists, inds = kdt.query(vecs, k=3)
    return dists[:, 1:]


def calc_id(vecs) -> float:  # [N, D]
    NN12 = get_NN_1_and_2(vecs)  # [N, 2]
    mu = NN12[:, 1] / NN12[:, 0]
    mu = mu[~np.isnan(mu)]
    mu = mu[~np.isinf(mu)]
    intrinsic_dimension = 1 / np.log(mu).mean()

    return intrinsic_dimension
