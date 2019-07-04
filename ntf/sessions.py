import numpy as np
from npy import d_of_l, append_d_of_l
from itertools import count
from tqdm import tqdm
import tensorflow as tf
from .tb_tools import add_summary_values
from npy import task


__all__ = ['runner', 'run_dict', 'run_op', 'hook_generator']


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


def hook_generator(keys, ln=False):
    end = '\n' if ln else ''

    def hook(result_one):
        fmt_strs = list()
        if 'i_batch' in result_one:
            fmt_strs.append('Batch #{i_batch:04d}')
        for key in keys:
            fmt_strs.append('%s: {%s:.3f}' % (key, key))
        fmt_str = '\r' + ', '.join(fmt_strs)

        d = {}
        for k, v in result_one.items():
            if isinstance(v, np.ndarray) or isinstance(v, list):
                try:
                    d[k] = np.mean(v)
                except:
                    pass
            else:
                d[k] = v
        print(fmt_str.format(**d), end=end)

    return hook
