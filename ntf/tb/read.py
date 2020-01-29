import tensorflow as tf
from typing import List
list_of_dict = List[dict]


__all__ = ['scalars']


def scalars(fpath: str) -> list_of_dict:
    result = list()
    for event in tf.train.summary_iterator(fpath):
        d = dict()
        for v in event.summary.value:
            d[v.tag] = d[v.simple_value]
        d['step'] = event.step

        result.append(d)
    return result
