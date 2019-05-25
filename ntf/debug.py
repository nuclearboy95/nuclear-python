import tensorflow as tf


__all__ = ['debug']


def debug(tensor):
    print(tensor.name, ':', tensor.shape)
