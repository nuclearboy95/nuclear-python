import numpy as np
import tensorflow as tf


__all__ = ['mix_datasets', 'repeat_elementwise', 'from_dataset']


def mix_datasets(datasets, lengths, fit_longest=True):
    if fit_longest:  # fit longest.
        lengths = np.array(lengths)
        weights = lengths / lengths.sum()
        d = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
        return d

    else:  # fit shortest.
        d = tf.data.Dataset.zip(tuple(datasets))
        d = d.flat_map(lambda x, y: tf.data.Dataset.concatenate(
            tf.data.Dataset.from_tensors(x),
            tf.data.Dataset.from_tensors(y)
        )
                       )
        return d


def repeat_elementwise(dataset, count) -> tf.data.Dataset:
    """
    Repeat element of each dataset by *count*.

    :param tf.data.Dataset dataset:
    :param int count: number_of_repeats
    :return:
    """
    return dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(count)
    )


def from_dataset(x, y) -> tf.data.Dataset:
    """

    :param np.ndarray x:
    :param np.ndarray y:

    :return:
    """
    assert x.shape[0] == y.shape[0], 'Number of data should be the same'
    N = x.shape[0]
    shape_x = x.shape[1:]
    shape_y = y.shape[1:]

    def index(i):
        return x[i], y[i]

    def shaper(x_, y_):
        return tf.reshape(x_, shape_x), tf.reshape(y_, shape_y)

    d = tf.data.Dataset.range(N)
    d = d.map(lambda i: tf.py_func(index, [i], [x.dtype, y.dtype]))
    d = d.map(shaper)

    return d
