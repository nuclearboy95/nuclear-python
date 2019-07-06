import tensorflow as tf
import os
from npy.utils import lazy_property
from npy.ns import sayd, sayw


__all__ = ['Model']


class Model:
    @property
    def name(self):
        return self.__class__.__name__

    @lazy_property
    def is_training(self):
        return tf.placeholder(tf.bool, [], name='is_training')

    @lazy_property
    def saver(self):
        vs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return tf.train.Saver(var_list=vs)

    def save(self, sess, ckpt_path=None, save_name=None, verbose=True):
        if save_name is None:
            save_name = self.name

        if ckpt_path is None:
            ckpt_path = './ckpts/%s/%s.ckpt' % (save_name, save_name)

        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        self.saver.save(sess, ckpt_path)

        if verbose:
            sayd("Model [{name}] saved to '{ckpt_path}'.".format(name=self.name, ckpt_path=ckpt_path))

    def load(self, sess, ckpt_path=None, save_name=None, ignore=False, verbose=True):
        if save_name is None:
            save_name = self.name

        if ckpt_path is None:
            ckpt_path = './ckpts/%s/%s.ckpt' % (save_name, save_name)

        try:
            self.saver.restore(sess, ckpt_path)

            if verbose:
                sayd("Model [{name}] loaded from '{ckpt_path}'.".format(name=self.name, ckpt_path=ckpt_path))
        except:
            sayw("Failed loading model [{name}] from '{ckpt_path}'.".format(name=self.name, ckpt_path=ckpt_path))
            if not ignore:
                raise
