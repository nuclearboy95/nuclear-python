from npy.ns import *
import numpy as np
import ntf
from npy import filter_d_of_l_of_num


__all__ = ['on_epoch']


def get_fmt_str(result):
    fmt_tokens = list()
    keys = sorted(result.keys())

    def is_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test']:
            return False
        return True

    for key in list(filter(is_key, keys)):
        key_token = key.split('/')[-1]
        fmt_tokens.append('%s: {%s:.3f}' % (key_token, key_token))

    fmt_str = ', '.join(fmt_tokens)
    fmt_str = 'Epoch #{i_epoch:03d} ' + fmt_str

    return fmt_str


def refine_result(result) -> dict:
    """
    :param dict result:

    :return:
    """
    with task('Filter keys'):
        result = filter_d_of_l_of_num(result)
        keys = sorted(result.keys())

    with task('Take average'):
        if 'batch_size' in keys:
            result = {key: np.average(result[key], weights=result['batch_size'])
                      for key in keys if key != 'batch_size'}

        else:
            result = {key: np.mean(result[key]) for key in keys}

    return result


def rename_result(result) -> dict:
    return {key.split('/')[-1]: value for key, value in result.items()}


#######################################

def on_epoch(result, i_epoch, tbname=None, tb=True):
    with task('Preprocess result'):
        result = refine_result(result)
        # result has only scalar data.

    with task('1. Tensorboard'):
        if tb:
            ntf.add_summary_values(result, step=i_epoch, name=tbname)

    with task('2. Print'):
        fmt_str = get_fmt_str(result)
        renamed_result = rename_result(result)
        log_str = fmt_str.format(i_epoch=i_epoch, **renamed_result)

        print('\r', end='')
        sayi(log_str)
