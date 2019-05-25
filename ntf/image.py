import tensorflow as tf
from .constants import MEAN_IMAGENET, STD_IMAGENET


__all__ = ['preprocess_imagenet', 'unpreprocess_imagenet']


def preprocess_imagenet(x):
    x = tf.cast(x, tf.float32)
    x /= 255.
    x -= tf.constant(MEAN_IMAGENET)
    x /= tf.constant(STD_IMAGENET)
    return x


def unpreprocess_imagenet(x):
    x *= tf.constant(STD_IMAGENET)
    x += tf.constant(MEAN_IMAGENET)
    x *= 255.
    x = tf.saturate_cast(x, tf.uint8)
    return x
