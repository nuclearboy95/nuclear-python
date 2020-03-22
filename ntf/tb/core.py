import tensorflow as tf
from npy import ddict
from npy.ns import sayi
import datetime

_DEFAULT_TB_NAME = ''
_USE_TB = True

__all__ = ['set_name', 'get_name', 'add_summary', 'off', 'on']


def off():
    global _USE_TB
    _USE_TB = False


def on():
    global _USE_TB
    _USE_TB = True


def get_tb_use():
    global _USE_TB
    return _USE_TB


def set_name(name: str):
    global _DEFAULT_TB_NAME
    _DEFAULT_TB_NAME = name


def get_name() -> str:
    global _DEFAULT_TB_NAME
    return _DEFAULT_TB_NAME


def filewriter_factory(name: str):
    time_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S.%f')

    if name == '':
        exp_name = f'{time_str}'
    else:
        exp_name = f'{name} ({time_str})'

    sayi(f'Tensorboard exp name is {exp_name}')

    log_name = f'tblog/{exp_name}'
    return tf.summary.FileWriter(log_name, graph=tf.get_default_graph())


writers = ddict(filewriter_factory)


def add_summary(summary, step: int = None, name: str = None, filewriter=None):
    if not _USE_TB:
        return
    if name is None:
        name = get_name()
    if filewriter is None:
        filewriter = writers[name]
    filewriter.add_summary(summary, global_step=step)
