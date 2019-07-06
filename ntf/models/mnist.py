import tensorflow as tf
from .model import Model


__all__ = ['MNISTModel']


def preprocess(x):
    x = tf.cast(x, tf.float32)
    x /= 255.
    return x


class MNISTModel(Model):
    def __init__(self, x):
        h = preprocess(x)
        self.preprocessed = h

        with tf.variable_scope(self.name):
            with tf.variable_scope('block1'):
                h = tf.layers.conv2d(h, 8, 3, activation=tf.nn.relu)
                h = tf.layers.conv2d(h, 8, 3, activation=tf.nn.relu)
                h = tf.layers.max_pooling2d(h, 2, 2)

            with tf.variable_scope('block2'):
                h = tf.layers.conv2d(h, 16, 3, activation=tf.nn.relu)
                h = tf.layers.conv2d(h, 16, 3, activation=tf.nn.relu)
                h = tf.layers.max_pooling2d(h, 2, 2)

            with tf.variable_scope('classifier'):
                h = tf.layers.flatten(h)
                h = tf.layers.dense(h, 64, activation=tf.nn.relu)
                h = tf.layers.dense(h, 10, activation=None)
                self.logits = h

                h = tf.nn.softmax(h)
                self.probs = h
