import tensorflow as tf


__all__ = ['accuracy']


def accuracy(preds, labels=None, labels_sparse=None, class_=None):
    if labels_sparse is not None:
        labels_a = tf.cast(labels_sparse, tf.int32)
    else:
        labels_a = tf.cast(tf.argmax(labels, axis=-1), tf.int32)
    preds_a = tf.cast(tf.argmax(preds, axis=-1), tf.int32)

    correct = tf.cast(tf.equal(preds_a, labels_a), tf.float32)

    if class_ is not None:
        in_class = tf.equal(labels_a, class_)
        correct = tf.boolean_mask(correct, in_class)

    acc = tf.reduce_mean(correct)
    return tf.where(tf.is_nan(acc), 0., acc)
