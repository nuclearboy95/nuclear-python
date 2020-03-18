from torch.utils.data import Dataset


__all__ = ['softmax', 'ArrayDataset']


def softmax(logits):
    exps = (logits - logits.max()).exp()
    return exps / exps.sum()


class ArrayDataset(Dataset):
    def __init__(self, x, y=None, tfs=None):
        super().__init__()
        self.x = x
        self.y = y
        self.tfs = tfs

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]

        if self.tfs is not None:
            x = self.tfs(x)

        if self.y is not None:
            return x, self.y[idx]

        else:
            return x
