import tensorflow as tf
from npy import ddict
from npy.ns import sayi
from npy.image import image_to_bytes, shape
import datetime


__all__ = ['add_summary', 'add_summary_values',
           'add_summary_images', 'add_summary_imagess',
           'add_summary_scalars']


def add_summary(summary, step=None, name='', filewriter=None):
    if filewriter is None:
        filewriter = writers[name]
    filewriter.add_summary(summary, global_step=step)


##############################################


def _to_value_scalar(tag, scalar):
    return tf.summary.Summary.Value(tag=tag, simple_value=scalar)


def add_summary_scalars(d, step=None, name='', prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_summary_scalars(d2, step=step, name=name, filewriter=filewriter)
        return

    summary = tf.summary.Summary(
        value=[_to_value_scalar(tag, scalar) for tag, scalar in d.items()]
    )
    add_summary(summary, step=step, name=name, filewriter=filewriter)


##############################################

def _to_value_image(tag, image):
    H, W, C = shape(image)
    summary = tf.summary.Summary.Value(
        tag=tag,
        image=tf.summary.Summary.Image(
            encoded_image_string=image_to_bytes(image).getvalue(),
            height=H, width=W
        )
    )
    return summary


def _to_summary_image(tag, image):
    return tf.summary.Summary(value=[_to_value_image(tag, image)])


def add_summary_images(d, step=None, name='', prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_summary_images(d2, step=step, name=name, filewriter=filewriter)
        return

    for tag, image in d.items():
        summary = _to_summary_image(tag, image)
        add_summary(summary, step=step, name=name, filewriter=filewriter)


def _to_summary_images(tag, images):
    values = [
        _to_value_image('%s/%d' % (tag, i), image)
        for i, image in enumerate(images)
    ]
    summary = tf.summary.Summary(value=values)
    return summary


def add_summary_imagess(d, step=None, name='', prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_summary_imagess(d2, step=step, name=name, filewriter=filewriter)
        return

    for tag, image in d.items():
        summary = _to_summary_images(tag, image)
        add_summary(summary, step=step, name=name, filewriter=filewriter)


##############################################

def filewriter_factory(name):
    time_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S.%f')

    if name == '':
        exp_name = '{}'.format(time_str)
    else:
        exp_name = '{} ({})'.format(name, time_str)

    sayi('Tensorboard exp name is %s' % exp_name)

    log_name = 'tblog/%s' % exp_name
    return tf.summary.FileWriter(log_name, graph=tf.get_default_graph())


writers = ddict(filewriter_factory)
add_summary_values = add_summary_scalars
