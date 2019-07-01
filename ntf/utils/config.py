import tensorflow as tf


__all__ = ['config']


def config():
    con = tf.ConfigProto()
    con.gpu_options.allow_growth = True
    return con
