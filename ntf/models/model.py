import tensorflow as tf
import os
from npy.utils import lazy_property


__all__ = ['Model']


class Model:
    @property
    def name(self):
        return self.__class__.__name__

    @lazy_property
    def saver(self):
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return tf.train.Saver(var_list=vs)

    def save(self, sess, ckpt_path=None, name=None):
        if name is None:
            name = self.name

        if ckpt_path is None:
            ckpt_path = './ckpts/%s/%s' % (name, name)

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.saver.save(sess, ckpt_path)

    def load(self, sess, ckpt_path=None, name=None):
        if name is None:
            name = self.name

        if ckpt_path is None:
            ckpt_path = './ckpts/%s/%s' % (name, name)

        self.saver.restore(sess, ckpt_path)
