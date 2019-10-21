from abc import ABCMeta
import numpy as np
from npy import isnum, d_of_l, keys_d_of_l_of_num, keys_d_of_num, isarray, sayi
import time


__all__ = ['PrintCallback', 'Callback', 'TensorboardCallback']


def refine_batch_result(result) -> dict:
    """

    :param dict result:
    :return:
    """
    def is_valid_key(key):
        if key.startswith(':'):
            return False
        if not isnum(result[key]):
            return False
        return True

    keys = sorted(list(filter(is_valid_key, result.keys())))
    result = {key: result[key] for key in keys}

    return result


def refine_epoch_result(result) -> dict:
    """
    :param dict result:

    :return:
    """
    keys = keys_d_of_num(result) + keys_d_of_l_of_num(result)
    keys = list(filter(lambda k: not k.startswith(':'), keys))
    keys.sort()

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


def get_batch_fmt_str(result) -> str:
    """

    :param dict result:
    :return:
    """
    fmt_tokens = list()
    keys = sorted(result.keys())

    def is_valid_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test', 'train_loss', 'test_loss']:
            return False
        return True

    for key in list(filter(is_valid_key, keys)):
        key_token = '/'.join(key.split('/')[1:])
        fmt_tokens.append('%s: {%s:.3f}' % (key_token, key_token))

    fmt_str = '  '.join(fmt_tokens)
    if 'i_batch' in result:
        prefix = 'Batch {i_batch:3d}  '
    else:
        prefix = ''

    return prefix + fmt_str


def get_epoch_fmt_str(result):
    fmt_tokens = list()
    keys = sorted(result.keys())

    def is_valid_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test', 'train_loss', 'test_loss']:
            return False
        return True

    for key in list(filter(is_valid_key, keys)):
        key_token = '/'.join(key.split('/')[1:])
        fmt_tokens.append('%s: {%s:.3f}' % (key_token, key_token))

    fmt_str = '  '.join(fmt_tokens)
    prefix = 'Epoch {i_epoch:2d}]  '
    return prefix + fmt_str


def rename_result(result) -> dict:
    return {key.split('/')[-1]: value for key, value in result.items()}


class Callback:
    __metaclass__ = ABCMeta

    def on_train_begin(self):
        pass

    def on_train_end(self, result=None):
        pass

    def on_test_begin(self):
        pass

    def on_test_end(self, result=None):
        pass

    def on_train_batch_begin(self, i_batch, result_batch=None):
        pass

    def on_train_batch_end(self, i_batch, result_batch=None):
        pass

    def on_epoch_begin(self, i_epoch, result_epoch=None):
        pass

    def on_epoch_end(self, i_epoch, result_epoch=None):
        pass

    def on_test_batch_begin(self, i_batch, result_batch=None):
        pass

    def on_test_batch_end(self, i_batch, result_batch=None):
        pass


class HistoryCallback(Callback):
    def __init__(self):
        self.result_epoch = d_of_l()
        self.result_test = d_of_l()

    def on_train_batch_end(self, i_batch, result_batch=None):
        self.result_epoch.appends(result_batch)

    def on_epoch_begin(self, i_epoch, result_epoch=None):
        self.result_epoch.clear()
        if result_epoch is not None:
            self.result_epoch.update(result_epoch)

    def on_epoch_end(self, i_epoch, result_epoch=None):
        if result_epoch is not None:
            self.result_epoch.update(result_epoch)

    def on_test_begin(self):
        self.result_test.clear()

    def on_test_end(self, i_epoch=None, result=None):
        if result is not None:
            self.result_test.update(result)


class PrintCallback(HistoryCallback):
    def __init__(self):
        super(PrintCallback, self).__init__()

    def on_train_batch_end(self, i_batch, result_batch=None):
        if result_batch is None:
            result_batch = {}
        super().on_train_batch_end(i_batch, result_batch)

        result_batch = refine_batch_result(result_batch)
        fmt_str = get_batch_fmt_str(result_batch)
        result_renamed = rename_result(result_batch)
        log_str = fmt_str.format(**result_renamed)
        print('\r' + log_str, end='')

    def on_epoch_begin(self, i_epoch, result_epoch=None):
        super().on_epoch_begin(i_epoch, result_epoch)

        print('\r', end='', flush=True)
        time.sleep(0.01)

    def on_epoch_end(self, i_epoch, result_epoch=None):
        super().on_epoch_end(i_epoch, result_epoch)

        print('\r', end='')
        result_epoch = self.result_epoch
        result_epoch = refine_epoch_result(result_epoch)
        fmt_str = get_epoch_fmt_str(result_epoch)
        renamed_result = rename_result(result_epoch)
        log_str = fmt_str.format(i_epoch=i_epoch, **renamed_result)
        sayi(log_str)

    def on_test_end(self, i_epoch=None, result=None):
        super().on_test_end(i_epoch, result)

        print('\r', end='')
        result_test = self.result_test
        result_test = refine_epoch_result(result_test)
        fmt_str = get_epoch_fmt_str(result_test)
        renamed_result = rename_result(result_test)
        log_str = fmt_str.format(i_epoch=i_epoch, **renamed_result)
        sayi(log_str)


class TensorboardCallback(Callback):
    def on_epoch_end(self, i_epoch, result_epoch=None):
        pass
