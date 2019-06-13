import tensorflow as tf
import numpy as np
from npy import d_of_l, append_d_of_l
from itertools import count
from tqdm import tqdm
from .misc import abs_max, norms2


__all__ = ['runner', 'run_dict', 'run_op', 'hook_generator', 'minimize_clipped', 'minimize']


def minimize_clipped(optimizer, loss, norm=1., return_max=False):
    gvs = optimizer.compute_gradients(loss)
    capped = [(tf.clip_by_norm(grad, norm), v) for grad, v in gvs]
    capped_g = [grad for grad, v in capped]
    max_v = abs_max(capped_g)
    # norm2 = norms2(capped_g)
    train_op = optimizer.apply_gradients(capped)
    if return_max:
        return train_op, max_v
    else:
        return train_op


def minimize(optimizer, loss, norm=None, return_grads_norm=False,
             return_grads_max=False, return_vars_norm=False,
             return_vars_max=False):
    """

    :param tf.train.Optimizer optimizer:
    :param tf.Tensor loss:
    :param float norm:
    :param bool return_grads_norm:
    :param bool return_grads_max:
    :param bool return_vars_norm:
    :param bool return_vars_max:

    :return:
    """
    grads_and_vars = optimizer.compute_gradients(loss)
    if norm is not None:
        grads_and_vars = [(tf.clip_by_norm(grad, norm), v)
                          for grad, v in grads_and_vars]

    grads = [grad for grad, v in grads_and_vars]
    vs = [v for grad, v in grads_and_vars]
    train_op = optimizer.apply_gradients(grads_and_vars)
    ret = (train_op,)

    if return_grads_norm:
        norm2 = norms2(grads)
        ret += (norm2,)

    if return_grads_max:
        max_g = abs_max(grads)
        ret += (max_g,)

    if return_vars_norm:
        norm2_v = norms2(vs)
        ret += (norm2_v,)

    if return_vars_max:
        max_v = abs_max(vs)
        ret += (max_v,)

    return ret


def runner(sess, ops, steps=None, verbose=True, feed_dict=None):
    i_batch_g = range(steps) if steps is not None else count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    for i_batch in i_batch_g:
        try:
            result = sess.run(ops, feed_dict=feed_dict)
            yield (i_batch, result)
        except tf.errors.OutOfRangeError:
            raise StopIteration


def run_dict(sess, ops, steps=None, verbose=True, hook=None, feed_dict=None):
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
            if 'i_batch' not in result_one:
                result_one['i_batch'] = i_batch

            hook(result_one)
    return results.as_dict()


def run_op(sess, op, steps=None, verbose=True, hook=None, feed_dict=None):
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

