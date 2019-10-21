from npy.ns import *
import numpy as np
import ntf
from npy import keys_d_of_l_of_num, keys_d_of_num, isarray


__all__ = ['on_epoch']


def get_fmt_str(result):
    fmt_tokens = list()
    keys = sorted(result.keys())

    def is_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test', 'train_loss', 'test_loss']:
            return False
        return True

    for key in list(filter(is_key, keys)):
        key_token = '/'.join(key.split('/')[1:])
        fmt_tokens.append('%s: {%s:.3f}' % (key_token, key_token))

    fmt_str = '  '.join(fmt_tokens)
    prefix = 'Epoch {i_epoch:2d}]  '
    return prefix + fmt_str


def refine_result(result) -> dict:
    """
    :param dict result:

    :return:
    """
    with task('Filter keys'):
        keys = keys_d_of_num(result) + keys_d_of_l_of_num(result)
        keys = list(filter(lambda k: not k.startswith(':'), keys))
        keys.sort()

    with task('Take average'):
        if 'batch_size' in keys:
            ret = {}
            for k in keys:
                if k == 'batch_size':
                    continue
                if isarray(result[k]):
                    ret[k] = np.average(result[k], weights=result['batch_size'])
                else:
                    ret[k] = result[k]

        else:
            ret = {key: np.mean(result[key]) for key in keys}

    return ret


def rename_result(result) -> dict:
    return {key.split('/')[-1]: value for key, value in result.items()}


#######################################

def on_epoch(result, i_epoch, tbname=None, tb=True):
    with task('Preprocess result'):
        result = refine_result(result)
        # result has only scalar data.

    with task('1. Tensorboard'):
        if tb:
            ntf.tb.add_scalars(result, step=i_epoch, name=tbname)

    with task('2. Print'):
        fmt_str = get_fmt_str(result)
        renamed_result = rename_result(result)
        log_str = fmt_str.format(i_epoch=i_epoch, **renamed_result)

        print('\r', end='')
        sayi(log_str)
