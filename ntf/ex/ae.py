from npy import task, set_tf_log, set_cuda
import tensorflow as tf
from tensorflow import keras
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


def get_encoder():
    with tf.variable_scope('Enc'):
        model = keras.models.Sequential(name='Encoder')
        model.add(ntf.keras.layers.Standardize(std=255., input_shape=(28, 28, 1)))
        model.add(keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.Conv2D(8, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D())

        model.add(keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.Conv2D(16, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D())

        model.add(keras.layers.Conv2D(4, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.Conv2D(4, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPool2D())

        return model


def get_decoder():
    with tf.variable_scope('Dec'):
        model = keras.models.Sequential(name='Decoder')
        model.add(ntf.keras.layers.Standardize(input_shape=(3, 3, 4)))
        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2DTranspose(4, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.Conv2DTranspose(4, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2DTranspose(16, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.Conv2DTranspose(16, 3, padding='same', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.UpSampling2D())
        model.add(keras.layers.Conv2DTranspose(8, 3, padding='valid', activation=tf.nn.relu))
        model.add(keras.layers.Conv2DTranspose(8, 3, padding='valid', activation=tf.nn.relu))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2DTranspose(1, 3, padding='same', activation=tf.nn.sigmoid))
        model.add(keras.layers.Lambda(lambda x: x * 255.))
        return model


def main():
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data

    with task('Make Iterators'):
        it_train = Iterators.get_iterator(x_train, y_train)
        it_test = Iterators.get_iterator(x_test, y_test)

        handle_ph, it = ntf.data.iterator_like(it_train)

    with task('Build Graph'):
        X, y = it.get_next()
        enc = get_encoder()
        dec = get_decoder()
        latent = enc(X)
        recon = dec(latent)

        diff = tf.cast(X, tf.float32) - recon

        loss = tf.reduce_mean(tf.square(diff / 255.), axis=[1, 2, 3])
        optimizer = tf.train.AdamOptimizer(1e-3)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops = ntf.train.make_train_ops(optimizer, loss)

        test_ops = ntf.train.make_train_ops(optimizer, loss, train=False)

    with tf.Session(config=ntf.config()) as sess:
        train_handle = sess.run(it_train.string_handle())
        test_handle = sess.run(it_test.string_handle())

        sess.run(tf.global_variables_initializer())

        for i_epoch in range(10):

            with task('Train'):
                sess.run(it_train.initializer)
                keras.backend.set_learning_phase(True)

                result = ntf.run_dict(sess, train_ops, verbose=False, hook=ntf.hooks.on_batch,
                                      feed_dict={handle_ph: train_handle})
                print('\r', end='', flush=True)
                ntf.hooks.on_epoch(result, i_epoch)

            with task('Test'):
                sess.run(it_test.initializer)
                keras.backend.set_learning_phase(False)
                result = ntf.run_dict(sess, test_ops, verbose=False, feed_dict={handle_ph: test_handle})

                print('\r', end='', flush=True)
                ntf.hooks.on_epoch(result, i_epoch)


if __name__ == '__main__':
    set_cuda(0)
    set_tf_log()
    main()
