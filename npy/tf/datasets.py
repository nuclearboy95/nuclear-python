import numpy as np
import tensorflow as tf


__all__ = ['mix_datasets', 'repeat_elementwise']


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


def repeat_elementwise(dataset, count):
    return dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensors(x).repeat(count)
    )
