from npy.ns import *
import numpy as np
import ntf


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

    if 'i_batch' in result:
        fmt_str = 'Batch #{i_batch:04d} ' + fmt_str

    return fmt_str


def get_keys(result_epoch):
    keys = sorted(result_epoch.keys())
    keys = list(filter(lambda key: not key.startswith(':'), keys))
    keys = list(filter(lambda key: key != 'i_batch', keys))
    return keys


def refine_result(result_epoch) -> dict:
    with task('Filter keys'):
        keys = get_keys(result_epoch)
        result_epoch = {key: np.asarray(result_epoch[key]) for key in keys}
        keys = list(filter(lambda key: np.issubdtype(result_epoch[key].dtype, np.number), keys))

    with task('Take average'):
        if 'batch_size' in keys:
            result_epoch = {key: np.average(result_epoch[key], weights=result_epoch['batch_size'])
                            for key in keys if key != 'batch_size'}

        else:
            result_epoch = {key: np.mean(result_epoch[key]) for key in keys}

    return result_epoch


def rename_result(result_epoch) -> dict:
    keys = sorted(result_epoch.keys())
    with task('Rename keys'):
        ret = {}
        for key in keys:
            ret[key.split('/')[-1]] = result_epoch[key]
    return ret


#######################################

def on_epoch(result_epoch, i_epoch, tbname=None, tb=True):
    with task('Preprocess result'):
        result_epoch = refine_result(result_epoch)

    with task('Print'):
        fmt_str = get_fmt_str(result_epoch)
        renamed_result = rename_result(result_epoch)
        log_str = fmt_str.format(i_epoch=i_epoch, **renamed_result)

        print('\r', end='')
        sayi(log_str)

    with task('Tensorboard'):
        if tb:
            ntf.add_summary_values(result_epoch, step=i_epoch, name=tbname)
