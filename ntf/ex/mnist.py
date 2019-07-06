from npy.ns import *
import tensorflow as tf
from ..models import MNISTModel
import ntf


__all__ = ['mnist']


class Iterators:
    @staticmethod
    def get_iterator(x, y, batch_size=128):
        def foo(images, labels):
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
    set_tf_log(5)
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    with task('Make Iterators'):
        it_train = Iterators.get_iterator(x_train, y_train)
        it_test = Iterators.get_iterator(x_test, y_test)

        handle_ph, it = ntf.data.iterator_like(it_train)

    with task('Build Graph'):
        X, y = it.get_next()
        model = MNISTModel(X)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=y)
        optimizer = tf.train.AdamOptimizer(1e-3)

        train_ops = ntf.train.make_train_ops(optimizer, loss)
        train_ops.update(ntf.train.make_metric_ops(labels=y, preds=model.logits))

        test_ops = ntf.train.make_train_ops(optimizer, loss, train=False)
        test_ops.update(ntf.train.make_metric_ops(labels=y, preds=model.logits, train=False))

    with tf.Session(config=ntf.config()) as sess:
        train_handle = sess.run(it_train.string_handle())
        test_handle = sess.run(it_test.string_handle())

        sess.run(tf.global_variables_initializer())

        for i_epoch in range(10):

            with task('Train'):
                sess.run(it_train.initializer)

                result = ntf.run_dict(sess, train_ops, verbose=False, feed_dict={handle_ph: train_handle},
                                      hook=ntf.hooks.on_batch)
                print('\r', end='', flush=True)
                ntf.hooks.on_epoch(result, i_epoch)

            with task('Test'):
                sess.run(it_test.initializer)
                result = ntf.run_dict(sess, test_ops, verbose=False,
                                      feed_dict={handle_ph: test_handle})

                print('\r', end='', flush=True)
                ntf.hooks.on_epoch(result, i_epoch)
