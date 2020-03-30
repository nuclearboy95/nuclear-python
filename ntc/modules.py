import torch
import torch.nn as nn

from npy import makedirpath
from npy.log import sayd, saye
import os
from functools import reduce


__all__ = ['Flatten', 'Module', 'parameters']


def parameters(*modules) -> list:
    params = [list(module.parameters()) for module in modules]
    params = reduce(lambda x, y: x + y, params)
    return params


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


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

