import numpy as np
import tensorflow as tf


__all__ = ['iterator_like']


def iterator_like(iterator, handle_ph=None) -> tuple:
    """

    :param tf.data.Iterator iterator:
    :param tf.Tensor handle_ph:
    :return:
    """

    if handle_ph is None:
        handle_ph = tf.placeholder(tf.string, [])

    iterator2 = tf.data.Iterator.from_string_handle(handle_ph, iterator.output_types,
                                                    output_shapes=iterator.output_shapes)

    return handle_ph, iterator2
