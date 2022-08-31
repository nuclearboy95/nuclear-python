import tensorflow as tf
from typing import List
list_of_dict = List[dict]


__all__ = ['scalars']


def scalars(fpath: str) -> list_of_dict:
    result = list()
    for event in tf.train.summary_iterator(fpath):
        if len(event.summary.value) == 0:
            continue
        d = dict()
        for v in event.summary.value:
            d[v.tag] = v.simple_value
        d['step'] = event.step

        result.append(d)
    return result
