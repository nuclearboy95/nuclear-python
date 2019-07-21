import tensorflow as tf


__all__ = ['Standardize']


class Standardize(tf.keras.layers.Layer):
    def __init__(self, mean=0., std=1.):
        super(Standardize, self).__init__()
        self._mean = mean
        self._std = std

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        inputs -= self._mean
        inputs /= self._std
        return inputs
