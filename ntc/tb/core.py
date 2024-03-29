from npy import ddict
from npy.ns import sayi
import datetime
import os

_DEFAULT_TB_NAME = ''
_USE_TB = True
_ROOT_PATH = './tblog'

__all__ = ['set_name', 'get_name', 'get_writer', 'off', 'on', 'set_root_path']


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


def get_root_path() -> str:
    global _ROOT_PATH
    return _ROOT_PATH


def set_root_path(v: str):
    global _ROOT_PATH
    _ROOT_PATH = v


def get_log_path(name: str):
    time_str = datetime.datetime.now().strftime('%d %b %H:%M:%S')
    if name == '':
        exp_name = f'{time_str}'
    else:
        exp_name = f'{name} ({time_str})'

    sayi(f'Tensorboard exp name is {exp_name}')

    log_path = os.path.join(get_root_path(), exp_name)
    return log_path


def filewriter_factory(name: str):
    from torch.utils.tensorboard import SummaryWriter
    log_path = get_log_path(name)
    return SummaryWriter(log_path)


writers = ddict(filewriter_factory)


def get_writer(name=None):
    if not _USE_TB:
        return
    if name is None:
        name = get_name()
    return writers[name]