import tensorflow as tf
from ..constants import MEAN_IMAGENET, STD_IMAGENET


__all__ = ['preprocess_imagenet']


def preprocess_imagenet(x):
    x = tf.cast(x, tf.float32)
    x /= 255.
    x -= tf.constant(MEAN_IMAGENET)
    x /= tf.constant(STD_IMAGENET)
    return x
