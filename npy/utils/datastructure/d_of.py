from collections import defaultdict
from abc import ABCMeta


__all__ = ['d_of_l', 'd_of_f', 'd_of_i', 'attrdict']


class attrdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def as_dict(self):
        return dict(self)

    def filt_keys(self, prefix=''):
        return attrdict({k: v for k, v in self.items() if k.startswith(prefix)})


class d_of_sth(defaultdict):
    __metaclass__ = ABCMeta
    __getattr__ = dict.__getitem__

    def as_dict(self):
        return dict(self)

    filt_keys = attrdict.filt_keys


#############################

class d_of_l(d_of_sth):
    def __init__(self):
        super().__init__(list)

    def appends(self, d):
        for key, value in d.items():
            self[key].append(value)


class d_of_f(d_of_sth):
    def __init__(self):
        super().__init__(float)


class d_of_i(d_of_sth):
    def __init__(self):
        super().__init__(int)
