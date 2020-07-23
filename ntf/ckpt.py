import tensorflow as tf


__all__ = ['rename', 'load_variables']


def rename(checkpoint_dir, name_f):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            v = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            new_name = name_f(var_name)
            if new_name is not None:
                v2 = tf.Variable(v, name=new_name)

        else:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)


def load_variables(checkpoint_dir) -> dict:
    d = dict()
    for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
        v = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
        d[var_name] = v

    return d
