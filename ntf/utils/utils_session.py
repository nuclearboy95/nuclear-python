from npy import d_of_l, append_d_of_l
from itertools import count
from tqdm import tqdm
import tensorflow as tf


__all__ = ['runner', 'run_dict', 'run_op', 'config']


def config():
    con = tf.ConfigProto()
    con.gpu_options.allow_growth = True
    return con


def runner(sess, ops, steps=None, verbose=True, feed_dict=None):
    i_batch_g = range(steps) if steps is not None else count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    for i_batch in i_batch_g:
        try:
            result_batch = sess.run(ops, feed_dict=feed_dict)
            result_batch.update({'i_batch': i_batch})
            yield (i_batch, result_batch)
        except tf.errors.OutOfRangeError:
            raise StopIteration


def run_dict(sess, ops, steps=None, verbose=True, hook=None, feed_dict=None) -> dict:
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param hook:
    :param dict feed_dict:
    :return:
    """
    results = d_of_l()
    for i_batch, result_one in runner(sess, ops, steps=steps, verbose=verbose, feed_dict=feed_dict):
        append_d_of_l(results, result_one)
        if hook is not None:
            hook(result_one)
    return results.as_dict()


def run_op(sess, op, steps=None, verbose=True, hook=None, feed_dict=None) -> list:
    key = 'run_op'
    ops = {key: op}
    result_d = run_dict(sess, ops, steps=steps, verbose=verbose, hook=hook, feed_dict=feed_dict)
    return result_d[key]
