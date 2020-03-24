from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os

from npy import makedirpath
from npy.log import sayd, saye
import os
from functools import reduce


__all__ = ['softmax', 'Flatten', 'Module', 'parameters',
           'ArrayDataset', 'ConcatDataset']


def softmax(logits):
    exps = (logits - logits.max()).exp()
    return exps / exps.sum()


def parameters(*modules) -> list:
    params = [list(module.parameters()) for module in modules]
    params = reduce(lambda x, y: x + y, params)
    return params


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class ArrayDataset(Dataset):
    def __init__(self, x, y=None, tfs=None, repeat=1):
        super().__init__()
        self.x = x
        self.y = y
        self.tfs = tfs
        self.repeat = repeat
        self.N = self.x.shape[0]

    def __len__(self):
        return self.N * self.repeat

    def __getitem__(self, idx):
        idx = idx % self.N
        x = self.x[idx]

        if self.tfs is not None:
            x = self.tfs(x)

        if self.y is not None:
            return x, self.y[idx]

        else:
            return x


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        lengths = [len(d) for d in datasets]
        self._length = min(lengths)
        assert min(lengths) == max(lengths), 'Length of the datasets should be the same'

    def __getitem__(self, idx):
        return tuple(d[idx] for d in self.datasets)

    def __len__(self):
        return self._length


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

