from collections import defaultdict
from abc import ABCMeta
import numpy as np


__all__ = ['d_of_l', 'd_of_f', 'd_of_i',
           'd_of_d_of_l', 'd_of_d_of_f',
           'attrdict']


class BetterDict(dict):
    def filt_keys(self, prefix=''):
        return self.__class__({k: v for k, v in self.items() if k.startswith(prefix)})

    def apply(self, func):
        for k, v in self.items():
            self[k] = func(v)
        return self

    def applyarr(self, func):
        for k, v in self.items():
            if isinstance(v, list) or isinstance(v, np.ndarray):
                self[k] = func(v)
        return self

    def as_dict(self):
        return dict(self)


class attrdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    as_dict = BetterDict.as_dict
    filt_keys = BetterDict.filt_keys
    apply = BetterDict.apply
    applyarr = BetterDict.applyarr


class d_of_sth(defaultdict):
    __metaclass__ = ABCMeta
    __getattr__ = dict.__getitem__

    as_dict = BetterDict.as_dict
    filt_keys = BetterDict.filt_keys
    apply = BetterDict.apply
    applyarr = BetterDict.applyarr


#############################

class d_of_l(d_of_sth):
    def __init__(self, *args, **kwargs):
        super().__init__(list, *args, **kwargs)

    def appends(self, d):
        for key, value in d.items():
            self[key].append(value)


class d_of_f(d_of_sth):
    def __init__(self, *args, **kwargs):
        super().__init__(float, *args, **kwargs)


class d_of_i(d_of_sth):
    def __init__(self, *args, **kwargs):
        super().__init__(int, *args, **kwargs)


class d_of_d_of_l(d_of_sth):
    def __init__(self, *args, **kwargs):
        super().__init__(d_of_l, *args, **kwargs)


class d_of_d_of_f(d_of_sth):
    def __init__(self, *args, **kwargs):
        super().__init__(d_of_f, *args, **kwargs)
