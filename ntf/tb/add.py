import tensorflow as tf
from npy.image import image_to_bytes, shape
from .core import add_summary


__all__ = ['add_images', 'add_imagess', 'add_scalars', 'add_scalar']


##############################################


def _to_value_scalar(tag, scalar):
    return tf.summary.Summary.Value(tag=tag, simple_value=scalar)


def add_scalars(d, step=None, name=None, prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_scalars(d2, step=step, name=name, filewriter=filewriter)
        return

    summary = tf.summary.Summary(
        value=[_to_value_scalar(tag, scalar) for tag, scalar in d.items()]
    )
    add_summary(summary, step=step, name=name, filewriter=filewriter)


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


def add_imagess(d, step=None, name=None, prefix='', filewriter=None):
    if prefix:
        d2 = {prefix + '/' + key: value for key, value in d.items()}
        add_imagess(d2, step=step, name=name, filewriter=filewriter)
        return

    for tag, image in d.items():
        summary = _to_summary_images(tag, image)
        add_summary(summary, step=step, name=name, filewriter=filewriter)
