import tensorflow as tf
from npy import ddict
from npy.ns import sayi
import datetime


__all__ = ['add_summary', 'add_summary_values']


def add_summary(summary, step=None, name=''):
    filewriter = writers[name]
    filewriter.add_summary(summary, global_step=step)


def add_summary_values(d, step=None, name=''):
    filewriter = writers[name]
    summary = tf.summary.Summary(value=[tf.summary.Summary.Value(tag=tag, simple_value=value)
                                        for tag, value in d.items()])
    filewriter.add_summary(summary, global_step=step)


def filewriter_factory(name) -> tf.summary.FileWriter:
    time_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S.%f')

    if name == '':
        exp_name = '{}'.format(time_str)
    else:
        exp_name = '{} ({})'.format(name, time_str)

    sayi('Tensorboard exp name is %s' % exp_name)

    log_name = 'tblog/%s' % exp_name
    return tf.summary.FileWriter(log_name)


writers = ddict(filewriter_factory)
