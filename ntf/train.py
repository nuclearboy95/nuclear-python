import tensorflow as tf

from .utils import abs_max, norms2
from .metrics import accuracy


__all__ = ['minimize_clipped', 'minimize', 'make_train_ops', 'make_metric_ops']


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


def make_train_ops(optimizer, loss, norm=1., train=True, var_list=None, global_step=None,
                   return_grads_norm=True, return_grads_max=True,
                   return_vars_norm=True, return_vars_max=True) -> dict:
    """

    :param tf.train.Optimizer optimizer:
    :param tf.Tensor loss:
    :param float norm:
    :param bool train:
    :param list var_list:
    :param tf.Variable global_step:
    :param bool return_grads_norm:
    :param bool return_grads_max:
    :param bool return_vars_norm:
    :param bool return_vars_max:

    :return:
    """
    grads_and_vars = optimizer.compute_gradients(loss, var_list=var_list)
    grads = [grad for grad, v in grads_and_vars]
    vs = [v for grad, v in grads_and_vars]

    if norm is not None:
        grads, _ = tf.clip_by_global_norm(grads, norm)
        grads_and_vars = [(grad, v) for grad, v in zip(grads, vs)]

    mode = 'train' if train else 'test'
    ret = {}

    if True:
        key = f'{mode}/Loss'
        ret.update({key: tf.reduce_mean(loss)})

    if train:
        ret.update({':train_op': optimizer.apply_gradients(grads_and_vars, global_step=global_step)})

    if return_grads_norm:
        key = f'{mode}_monitor/grad_norm'
        ret.update({key: norms2(grads)})

    if return_grads_max:
        key = f'{mode}_monitor/grad_max'
        ret.update({key: abs_max(grads)})

    if return_vars_norm:
        key = f'{mode}_monitor/vars_norm'
        ret.update({key: norms2(vs)})

    if return_vars_max:
        key = f'{mode}_monitor/vars_max'
        ret.update({key: abs_max(vs)})

    return ret


def make_metric_ops(labels, preds, train=True, num_class=None,
                    return_acc=True,
                    return_class_acc=True, return_batch_size=True) -> dict:
    ret = {}
    mode = 'train' if train else 'test'

    if return_acc:
        key = f'{mode}/Acc'
        ret.update({key: accuracy(labels=labels, preds=preds)})

    if return_class_acc:
        if num_class is not None:
            for c in range(num_class):
                key = f'{mode}/Acc_{c}'
                ret.update({key: accuracy(labels=labels, preds=preds, label=c)})

    if return_batch_size:
        ret.update({'batch_size': tf.shape(labels)[0]})

    return ret
