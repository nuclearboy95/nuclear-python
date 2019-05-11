import tensorflow as tf
from .. import preprocess_imagenet
import os


__all__ = ['AlexNet']


class AlexNet:
    def __init__(self, x):
        h = preprocess_imagenet(x)

        with tf.variable_scope('alexnet'):
            with tf.variable_scope('feature'):
                with tf.variable_scope('block1'):
                    h = tf.pad(h, [[0, 0], [2, 2], [2, 2], [0, 0]])
                    h = tf.layers.conv2d(h, 64, 11, 4, activation=tf.nn.relu, padding='valid')
                    h = tf.layers.max_pooling2d(h, 3, 2)

                with tf.variable_scope('block2'):
                    h = tf.layers.conv2d(h, 192, 5, 1, activation=tf.nn.relu, padding='same')
                    h = tf.layers.max_pooling2d(h, 3, 2)

                with tf.variable_scope('block3'):
                    h = tf.layers.conv2d(h, 384, 3, 1, activation=tf.nn.relu, padding='same')

                with tf.variable_scope('block4'):
                    h = tf.layers.conv2d(h, 256, 3, 1, activation=tf.nn.relu, padding='same')

                with tf.variable_scope('block5'):
                    h = tf.layers.conv2d(h, 256, 3, 1, activation=tf.nn.relu, padding='same')
                    h = tf.layers.max_pooling2d(h, 3, 2)

                self.feature = h
            with tf.variable_scope('avgpool'):
                pass

            with tf.variable_scope('classifier'):
                h = tf.transpose(h, [0, 3, 1, 2])
                h = tf.layers.flatten(h)
                self.pooled = h
                with tf.variable_scope('block1'):
                    h = tf.layers.dense(h, 4096, activation=tf.nn.relu)
                    h = tf.layers.dropout(h)

                with tf.variable_scope('block2'):
                    h = tf.layers.dense(h, 4096, activation=tf.nn.relu)
                    h = tf.layers.dropout(h)

                with tf.variable_scope('block3'):
                    self.logits = tf.layers.dense(h, 1000)

                self.probs = tf.nn.softmax(self.logits)

    def load(self, sess, ckpt_path):
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alexnet')
        saver = tf.train.Saver(var_list=vs)
        saver.restore(sess, ckpt_path)

    def save(self, sess, ckpt_path):
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alexnet')
        saver = tf.train.Saver(var_list=vs)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        saver.save(sess, ckpt_path)