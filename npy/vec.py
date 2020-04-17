import numpy as np


__all__ = ['calc_cosine_similaritys', 'calc_cosine_similarity_pair',
           'min_cosine', 'mean_cosine', 'max_cosine']


def calc_cosine_similaritys(source, targets):
    # source [D]
    # targets [N, D]
    dot_products = np.dot(targets, source)
    cosine_similarity = dot_products / np.linalg.norm(targets, axis=-1) / np.linalg.norm(source)
    return cosine_similarity


def calc_cosine_similarity_pair(sources, targets):
    # sources [N, D]
    # targets [N, D]
    dot_products = np.asarray([np.dot(s, t) for s, t in zip(sources, targets)])
    norm_t = np.linalg.norm(targets, axis=-1)
    norm_s = np.linalg.norm(sources, axis=-1)
    cosine_similarity = dot_products / norm_t / norm_s
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
