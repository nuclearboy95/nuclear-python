import time
import numpy as np
from npy import isnum, d_of_l, keys_d_of_l_of_num, keys_d_of_num, isarray, sayi, task
from .. import tb

__all__ = ['HistoryCallback']


def removed_prefix(key):
    tokens = key.split('/')
    if tokens[0] in ['train', 'train_loss']:
        return '/'.join(tokens[1:])
    if tokens[0] in ['test', 'test_loss']:
        return 'test_' + '/'.join(tokens[1:])
    return key


def sorted_keys(keys):
    keys_test = sorted(list(filter(lambda x: x.startswith('test'), keys)))
    keys_train = sorted(list(filter(lambda x: x not in keys_test, keys)))
    return keys_train + keys_test


def refine_result_batch(i_batch, result) -> dict:
    """
    :param int i_batch:
    :param dict result:
    :return:
    """
    def is_valid_key(key):
        if key.startswith(':'):
            return False
        if not isnum(result[key]):
            return False
        return True

    keys = sorted_keys(list(filter(is_valid_key, result.keys())))
    result = {key: result[key] for key in keys}
    result['i_batch'] = i_batch

    return result


def refine_result_epoch(i_epoch, result) -> dict:
    """
    :param int i_epoch:
    :param dict result:

    :return:
    """
    keys = keys_d_of_num(result) + keys_d_of_l_of_num(result)
    keys = list(filter(lambda k: not k.startswith(':'), keys))
    keys.sort()

    if 'batch_size' in keys:
        weights = result['batchs_size']
        ret = {}
        for k in keys:
            if k == 'batch_size':
                continue

            if isarray(result[k]):
                ret[k] = np.average(result[k], weights=weights)
            else:
                ret[k] = result[k]

    else:
        ret = {key: np.mean(result[key]) for key in keys}

    ret['i_epoch'] = i_epoch

    return ret


def filter_result_epoch_tensorboard(i_epoch, result) -> dict:
    keys = list(result.keys())
    keys = [key for key in keys if key not in ['i_batch', 'i_epoch']]
    ret = {key: result[key] for key in keys}
    return ret


def get_batch_fmt_str(result) -> str:
    """

    :param dict result:
    :return:
    """
    fmt_tokens = list()
    keys = sorted_keys(result.keys())

    def is_valid_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test', 'train_loss', 'test_loss']:
            return False
        return True

    for key in list(filter(is_valid_key, keys)):
        # key_token = '/'.join(key.split('/')[1:])
        key_name = removed_prefix(key)
        key_token = key.replace('/', '_')
        fmt_tokens.append('%s: {%s:.3f}' % (key_name, key_token))

    fmt_str = '  '.join(fmt_tokens)
    if 'i_batch' in result:
        prefix = 'Batch {i_batch:3d}  '
    else:
        prefix = ''

    return prefix + fmt_str


def get_epoch_fmt_str(result):
    fmt_tokens = list()
    keys = sorted_keys(result.keys())

    def is_valid_key(key):
        tokens = key.split('/')
        if len(tokens) != 2:
            return False
        if tokens[0] not in ['train', 'test', 'train_loss', 'test_loss']:
            return False
        return True

    for key in list(filter(is_valid_key, keys)):
        # key_token = '/'.join(key.split('/')[1:])

        key_name = removed_prefix(key)
        key_token = key.replace('/', '_')

        fmt_tokens.append('%s: {%s:.3f}' % (key_name, key_token))

    fmt_str = '  '.join(fmt_tokens)
    prefix = 'Epoch {i_epoch:2d}]  '
    return prefix + fmt_str


def rename_result(result) -> dict:
    ret = {}
    for key, value in result.items():
        new_key = key.replace('/', '_')
        ret[new_key] = value
    return ret


class HistoryCallback:
    def __init__(self):
        self.result_train_epoch = d_of_l()
        self.result_test_epoch = d_of_l()

    def on_epoch_begin(self, i_epoch):
        self.result_train_epoch.clear()
        self.result_test_epoch.clear()

        self.clear_line()

    def on_epoch_end(self, i_epoch):
        self.clear_line()

        with task('Print'):
            result_train_epoch = refine_result_epoch(i_epoch, self.result_train_epoch)
            result_test_epoch = refine_result_epoch(i_epoch, self.result_test_epoch)

            result_epoch = result_test_epoch.copy()
            result_epoch.update(result_train_epoch)

            self.print_every_epoch(result_epoch)

        with task('Tensorboard'):
            result_tb = filter_result_epoch_tensorboard(i_epoch, result_epoch)
            tb.add_scalars(result_tb, step=i_epoch)

    def on_train_batch_begin(self, i_batch):
        pass

    def on_train_batch_end(self, i_batch, result_batch=None):
        if result_batch is None:
            result_batch = {}
        self.result_train_epoch.appends(result_batch)

        result_batch = refine_result_batch(i_batch, result_batch)
        self.print_every_batch(result_batch)

    def on_test_batch_begin(self, i_batch):
        pass

    def on_test_batch_end(self, i_batch, result_batch=None):
        self.result_test_epoch.appends(result_batch)

    ##################

    def report_train(self, result):
        self.result_train_epoch.appends(result)

    def report_test(self, result):
        self.result_test_epoch.appends(result)

    ##################

    @staticmethod
    def clear_line():
        print('\r', end='', flush=True)
        time.sleep(0.01)

    @staticmethod
    def print_every_batch(result):
        fmt_str = get_batch_fmt_str(result)
        result_renamed = rename_result(result)
        log_str = fmt_str.format(**result_renamed)
        print('\r' + log_str, end='')

    @staticmethod
    def print_every_epoch(result):
        fmt_str = get_epoch_fmt_str(result)
        renamed_result = rename_result(result)
        log_str = fmt_str.format(**renamed_result)
        sayi(log_str)
