import torch
import torch.nn as nn

import numpy as np
from npy import makedirpath
from npy.log import sayd, saye
import os
from functools import reduce


__all__ = ['Module', 'parameters', 'count_params', 'calc_norm']


def parameters(*modules) -> list:
    params = [list(module.parameters()) for module in modules]
    params = reduce(lambda x, y: x + y, params)
    return params


class Module(nn.Module):
    def save(self, fpath):
        makedirpath(fpath)
        torch.save(self.state_dict(), fpath)
        sayd(f'Saved to {fpath}.')

    def load(self, fpath):
        if os.path.exists(fpath):
            self.load_state_dict(torch.load(fpath))
            sayd(f'Loaded from {fpath}.')
        else:
            saye(f'Failed to load: {fpath} does not exist.')


def count_params(module):
    count = 0
    for param in list(module.parameters()):
        count += np.prod(param.size())
    return count


def calc_norm(params):
    ret = 0.
    for p in params:
        norm = p.grad.data.norm(2)
        ret += norm ** 2
    return ret ** 0.5
