import tensorflow as tf


__all__ = ['choice_deep', 'abs_max', 'norms2']


def choice_deep(N, p, shape):
    if len(shape) == 0:
        raise ValueError('Invalid shape:', shape)

    if len(shape) == 1:
        dist = tf.distributions.Categorical(probs=p)
        return dist.sample(N)

    else:
        l = list()
        for _ in range(shape[0]):
            l.append(choice_deep(N, p[0], shape[1:]))
        return tf.stack(l)


def abs_max(tensors):
    maxs = [tf.reduce_max(tf.abs(tensor)) for tensor in tensors]
    max_v = tf.reduce_max([0] + maxs)
    return max_v


def norms2(tensors):
    squares = [tf.reduce_sum(tf.square(tensor)) for tensor in tensors]
    square = tf.reduce_sum(squares)
    return tf.sqrt(square)
