from collections import defaultdict
from abc import ABCMeta


__all__ = ['d_of_l', 'attrdict']


class attrdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class d_of_something(defaultdict):
    __metaclass__ = ABCMeta
    __getattr__ = dict.__getitem__

    def as_dict(self):
        return dict(self)


#############################

class d_of_l(d_of_something):
    def __init__(self):
        super().__init__(list)
