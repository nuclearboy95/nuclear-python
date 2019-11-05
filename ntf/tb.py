import tensorflow as tf
from npy import ddict
from npy.ns import sayi
from npy.image import image_to_bytes, shape
import datetime


_DEFAULT_TB_NAME = ''


__all__ = ['add', 'set_name', 'add_images', 'add_imagess', 'add_scalars', 'add_scalar']


def set_name(name):
    global _DEFAULT_TB_NAME
    _DEFAULT_TB_NAME = name


def add(summary, step=None, name=None, filewriter=None):
    if name is None:
        name = _DEFAULT_TB_NAME
    if filewriter is None:
        filewriter = writers[name]
    filewriter.add_summary(summary, global_step=step)


##############################################


def _to_value_scalar(tag, scalar):
    return tf.summary.Summary.Value(tag=tag, simple_value=scalar)


def add_scalars(d, step=None, name=None, prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_scalars(d2, step=step, name=name, filewriter=filewriter)
        return

    if name is None:
        name = _DEFAULT_TB_NAME

    summary = tf.summary.Summary(
        value=[_to_value_scalar(tag, scalar) for tag, scalar in d.items()]
    )
    add(summary, step=step, name=name, filewriter=filewriter)


def add_scalar(key, value, step=None, name=None, prefix='', filewriter=None):
    add_scalars({key: value}, step=step, name=name, prefix=prefix, filewriter=filewriter)


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


def add_images(d, step=None, name=None, prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_images(d2, step=step, name=name, filewriter=filewriter)
        return

    if name is None:
        name = _DEFAULT_TB_NAME

    for tag, image in d.items():
        summary = _to_summary_image(tag, image)
        add(summary, step=step, name=name, filewriter=filewriter)


def _to_summary_images(tag, images):
    values = [
        _to_value_image('%s/%d' % (tag, i), image)
        for i, image in enumerate(images)
    ]
    summary = tf.summary.Summary(value=values)
    return summary


def add_imagess(d, step=None, name=None, prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_imagess(d2, step=step, name=name, filewriter=filewriter)
        return

    if name is None:
        name = _DEFAULT_TB_NAME

    for tag, image in d.items():
        summary = _to_summary_images(tag, image)
        add(summary, step=step, name=name, filewriter=filewriter)


##############################################

def filewriter_factory(name):
    time_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S.%f')

    if name == '':
        exp_name = f'{time_str}'
    else:
        exp_name = f'{name} ({time_str})'

    sayi(f'Tensorboard exp name is {exp_name}')

    log_name = f'tblog/{exp_name}'
    return tf.summary.FileWriter(log_name, graph=tf.get_default_graph())


writers = ddict(filewriter_factory)
