import numpy as np


__all__ = ['calc_cosine_similaritys',
           'min_cosine', 'mean_cosine', 'max_cosine']


def calc_cosine_similaritys(source, targets):
    # source [D]
    # targets [N, D]
    cosine_similarity = np.dot(targets, source) / np.linalg.norm(targets, axis=-1) / np.linalg.norm(source)
    return cosine_similarity


def min_cosine(source, targets):
    # source [D]
    # targets [N, D]
    cosine_similarity = calc_cosine_similaritys(source, targets)
    return cosine_similarity.min()


def mean_cosine(source, targets):
    # source [D]
    # targets [N, D]
    cosine_similarity = calc_cosine_similaritys(source, targets)
    return cosine_similarity.mean()


def max_cosine(source, targets):
    # source [D]
    # targets [N, D]
    cosine_similarity = calc_cosine_similaritys(source, targets)
    return cosine_similarity.max()
