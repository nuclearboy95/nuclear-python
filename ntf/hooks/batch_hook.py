from npy import task
import numpy as np


__all__ = ['on_batch']


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

    fmt_str = '  '.join(fmt_tokens)

    if 'i_batch' in result:
        fmt_str = 'Batch {i_batch:3d}  ' + fmt_str

    return fmt_str


def refine_result(result) -> dict:
    with task('Filter keys'):
        keys = sorted(result.keys())

        # 1. exclude ":train_op"
        keys = list(filter(lambda key: not key.startswith(':'), keys))

        # 2. only 0-dimensional numerical data.
        keys = list(filter(lambda key: np.issubdtype(type(result[key]), np.number), keys))

        result = {key: result[key] for key in keys}

    return result


def rename_result(result) -> dict:
    return {key.split('/')[-1]: value for key, value in result.items()}


#############################

def on_batch(result):
    with task('Preprocess result'):
        result = refine_result(result)

    with task('1. Tensorboard'):  # For now, do nothings.
        pass

    with task('2. Print'):
        fmt_str = get_fmt_str(result)
        result_renamed = rename_result(result)
        log_str = fmt_str.format(**result_renamed)

        print('\r' + log_str, end='')



