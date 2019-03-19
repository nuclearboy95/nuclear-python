import tensorflow as tf


__all__ = ['accuracy', 'accuracy_per_class']


def accuracy(labels, preds):
    return tf.reduce_mean(
        tf.cast(tf.equal(
            tf.cast(tf.argmax(preds, axis=-1), tf.int32),
            tf.cast(tf.argmax(labels, axis=-1), tf.int32)
        ), tf.float32)
    )


def accuracy_per_class(labels, preds, label):
    acc = tf.reduce_mean(
            tf.boolean_mask(
                tf.cast(
                    tf.equal(
                        tf.cast(tf.argmax(preds, axis=-1), tf.int32),
                        tf.cast(tf.argmax(labels, axis=-1), tf.int32)
                    ),
                    tf.float32
                ),
                tf.equal(
                    tf.argmax(labels, axis=-1),
                    label
                )
            )
        )
    return tf.where(tf.is_nan(acc), 0., acc)
