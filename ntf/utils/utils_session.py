from itertools import count
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import npy
from npy import d_of_l


__all__ = ['runner', 'runner_train', 'runner_test',
           'run_dict', 'run_dict_train', 'run_dict_test',
           'run_op', 'config', 'batch']


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
            if isinstance(result_batch, dict):
                result_batch.update({'i_batch': i_batch})
            yield (i_batch, result_batch)

        except tf.errors.OutOfRangeError:
            return


def runner_train(sess, ops, steps=None, verbose=True, feed_dict=None, cbk=None):
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param dict feed_dict:
    :param ntf.hooks.HistoryCallback cbk:
    :return:
    """
    i_batch_g = range(steps) if steps is not None else count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    for i_batch in i_batch_g:
        try:
            if cbk is not None:
                cbk.on_train_batch_begin(i_batch)
            result_batch = sess.run(ops, feed_dict=feed_dict)
            if cbk is not None:
                cbk.on_train_batch_end(i_batch, result_batch)

            if isinstance(result_batch, dict):
                result_batch.update({'i_batch': i_batch})
            yield (i_batch, result_batch)

        except tf.errors.OutOfRangeError:
            return


def runner_test(sess, ops, steps=None, verbose=True, feed_dict=None, cbk=None):
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param dict feed_dict:
    :param ntf.hooks.HistoryCallback cbk:
    :return:
    """
    i_batch_g = range(steps) if steps is not None else count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    for i_batch in i_batch_g:
        try:
            if cbk is not None:
                cbk.on_test_batch_begin(i_batch)
            result_batch = sess.run(ops, feed_dict=feed_dict)
            if cbk is not None:
                cbk.on_test_batch_end(i_batch, result_batch)

            if isinstance(result_batch, dict):
                result_batch.update({'i_batch': i_batch})
            yield (i_batch, result_batch)

        except tf.errors.OutOfRangeError:
            return


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
        results.appends(result_one)

        if hook is not None:
            hook(result_one)

    return results.as_dict()


def run_dict_train(sess, ops, steps=None, verbose=True, cbk=None, feed_dict=None) -> dict:
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param ntf.hooks.HistoryCallback cbk:
    :param dict feed_dict:
    :return:
    """
    results = d_of_l()
    for i_batch, result in runner_train(sess, ops, steps=steps, verbose=verbose, feed_dict=feed_dict, cbk=cbk):
        results.appends(result)

    return results.as_dict()


def run_dict_test(sess, ops, steps=None, verbose=True, cbk=None, feed_dict=None) -> dict:
    """

    :param tf.Session sess:
    :param dict ops:
    :param int steps:
    :param bool verbose:
    :param ntf.hooks.HistoryCallback cbk:
    :param dict feed_dict:
    :return:
    """
    results = d_of_l()
    for i_batch, result in runner_test(sess, ops, steps=steps, verbose=verbose, feed_dict=feed_dict, cbk=cbk):
        results.appends(result)

    return results.as_dict()


def run_op(sess, op, steps=None, verbose=True, hook=None, feed_dict=None) -> list:
    key = 'run_op'
    ops = {key: op}
    result_d = run_dict(sess, ops, steps=steps, verbose=verbose, hook=hook, feed_dict=feed_dict)
    return result_d[key]


def batch(data, batch_size, N=None, strict=False, shuffle=False):
    if N is None:
        N = data.shape[0]

    if shuffle:
        inds = np.random.permutation(N)
    else:
        inds = np.arange(N)

    if isinstance(data, tuple):
        for i_batch in range(npy.calc.num_batch(N, batch_size, strict=strict)):
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            d_batch = tuple(v[inds_batch] for v in data)
            yield i_batch, d_batch

    else:
        for i_batch in range(npy.calc.num_batch(N, batch_size, strict=strict)):
            inds_batch = inds[i_batch * batch_size: (i_batch + 1) * batch_size]
            x_batch = data[inds_batch]
            yield i_batch, x_batch
