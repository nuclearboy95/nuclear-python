import numpy as np
from npy import task


__all__ = ['loop']


def loop(fraction=None):
    with task('Import', debug=True):
        import tensorflow as tf

    with task('Build graph'):
        x = tf.placeholder(tf.float32, [None, None])
        y = tf.matmul(x, x)

    with task('Compute', debug=True):
        config = tf.ConfigProto()
        if fraction is not None:
            config.gpu_options.per_process_gpu_memory_fraction = fraction
        with tf.Session(config=config) as sess:
            while True:
                _ = sess.run(y, feed_dict={x: np.random.random((1024, 1024)).astype(np.float32)})
