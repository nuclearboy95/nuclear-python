from npy.ns import *
import tensorflow as tf
from ..models import Model
import ntf


__all__ = ['mnist']


class MNISTModel(Model):
    def __init__(self, x):
        h = x
        with tf.variable_scope('MNISTModel'):
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


class Iterators:
    @staticmethod
    def get_iterator(x, y, batch_size=128):
        def foo(images, labels):
            images = tf.cast(images, tf.float32)
            images /= 255.
            images = tf.reshape(images, [-1, 28, 28, 1])
            labels = tf.one_hot(labels, depth=10)
            return images, labels

        ds = ntf.data.from_dataset(x, y)
        ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        ds = ds.map(foo)
        ds = ds.prefetch(1)

        return ds.make_initializable_iterator()


def mnist():
    set_cuda()
    set_tf_log(5)
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    with task('Make Iterators'):
        it_train = Iterators.get_iterator(x_train[:1000], y_train[:1000])
        it_test = Iterators.get_iterator(x_test, y_test)

        handle = tf.placeholder(tf.string, [])
        it = tf.data.Iterator.from_string_handle(handle,
                                                 (tf.float32, tf.float32),
                                                 ((None, 28, 28, 1), (None, 10)))

    with task('Build Graph'):
        X, y = it.get_next()
        model = MNISTModel(X)
        loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y)
        optimizer = tf.train.AdamOptimizer(1e-3)
        ops = ntf.train.make_train_ops(optimizer, loss_op)
        ops.update(ntf.train.make_metric_ops(labels=y, preds=model.logits))

    with tf.Session(config=ntf.config()) as sess:
        train_handle = sess.run(it_train.string_handle())
        test_handle = sess.run(it_test.string_handle())

        sess.run(tf.global_variables_initializer())

        for i_epoch in range(10):
            sess.run(it_train.initializer)

            result = ntf.run_dict(sess, ops, verbose=False, feed_dict={handle: train_handle}, hook=ntf.hooks.on_batch)
            print('\r', end='', flush=True)
            ntf.hooks.on_epoch(result, i_epoch)
