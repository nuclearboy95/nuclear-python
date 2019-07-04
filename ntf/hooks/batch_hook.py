from npy import task


__all__ = ['on_batch']


def get_fmt_str(result_batch) -> str:  # FIXME logic-based
    return 'Batch #{i_batch:04d} Acc:{Acc:0.3f}, Loss:{Loss:0.3f}'


def get_keys(result_batch) -> list:  # FIXME logic-based
    return ['i_batch', 'train/Acc', 'train/Loss']


def refine_result(result_batch) -> dict:
    with task('Filter keys'):
        keys = get_keys(result_batch)
        result_batch = {key: result_batch[key] for key in keys}

    return result_batch


def rename_result(result_batch) -> dict:
    keys = sorted(result_batch.keys())
    with task('Rename keys'):
        ret = {}
        for key in keys:
            ret[key.split('/')[-1]] = result_batch[key]
    return ret


#############################

def on_batch(result_batch):
    with task('Preprocess result'):
        result_batch = refine_result(result_batch)

    with task('Print'):
        fmt_str = get_fmt_str(result_batch)
        result_renamed = rename_result(result_batch)
        log_str = fmt_str.format(**result_renamed)

        print('\r' + log_str, end='')

    with task('Tensorboard'):  # For now, do nothings.
        pass

