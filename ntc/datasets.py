from torch.utils.data import Dataset
import numpy as np
import npy


__all__ = ['ArrayDataset', 'DictionaryConcatDataset', 'PatchDataset']


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


class DictionaryConcatDataset(Dataset):
    def __init__(self, d_of_datasets):
        self.d_of_datasets = d_of_datasets
        lengths = [len(d) for d in d_of_datasets.values()]
        self._length = min(lengths)
        self.keys = self.d_of_datasets.keys()
        assert min(lengths) == max(lengths), 'Length of the datasets should be the same'

    def __getitem__(self, idx):
        return {
            key: self.d_of_datasets[key][idx]
            for key in self.keys
        }

    def __len__(self):
        return self._length


class PatchDataset(Dataset):
    def __init__(self, memmap, memmap_label=None, transform=None, K=32, S=1):
        super(PatchDataset, self).__init__()
        self.arr = memmap
        self.label = memmap_label
        self.transform = transform
        self.S = S
        self.K = K

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.row_num * self.col_num

    @npy.lazy_property
    def row_num(self):
        N, H, W = self.arr.shape[:3]
        K = self.K
        S = self.S
        I = npy.calc.cnn_output_size(H, K=K, S=S)
        return I

    @npy.lazy_property
    def col_num(self):
        N, H, W = self.arr.shape[:3]
        K = self.K
        S = self.S
        J = npy.calc.cnn_output_size(W, K=K, S=S)
        return J

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        n, i, j = np.unravel_index(idx, (N, self.row_num, self.col_num))
        K = self.K
        S = self.S
        image = self.arr[n]
        h, w = S * i, S * j
        patch = image[h: h + K, w: w + K]

        if self.transform:
            patch = self.transform(patch)

        if self.label is not None:
            return patch, n, i, j, self.label[n]

        else:
            return patch, n, i, j
