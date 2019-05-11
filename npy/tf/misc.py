import tensorflow as tf


__all__ = ['choice_deep']


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
