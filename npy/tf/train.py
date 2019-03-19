import tensorflow as tf
import numpy as np
from ..utils import d_of_l, append_d_of_l
from itertools import count
from tqdm import tqdm


__all__ = ['run_dict', 'run_op']


def run_dict(sess, ops, steps=None, verbose=True, hook=None, feed_dict=None):
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param hook:
    :return:
    """
    i_batch_g = range(steps) if steps is not None else count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    results = d_of_l()
    for i_batch in i_batch_g:
        try:
            result_one = sess.run(ops, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            break

        append_d_of_l(results, result_one)

        if hook is not None:
            if 'i_batch' not in result_one:
                result_one['i_batch'] = i_batch

            hook(result_one)
    return results.as_dict()


def run_op(sess, op, steps=None, verbose=True, hook=None, feed_dict=None):
    key = 'run_op'
    ops = {key: op}
    result_d = run_dict(sess, ops, steps=steps, verbose=verbose, hook=hook, feed_dict=feed_dict)
    return result_d[key]
