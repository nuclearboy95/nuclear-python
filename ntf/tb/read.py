import tensorflow as tf
from typing import List
list_of_dict = List[dict]


__all__ = ['scalars']


def scalars(fpath: str) -> list_of_dict:
    result = list()
    for event in tf.train.summary_iterator(fpath):
        if len(event.summary.accumulate) == 0:
            continue
        d = dict()
        for v in event.summary.accumulate:
            d[v.tag] = v.simple_value
        d['step'] = event.step

        result.append(d)
    return result
