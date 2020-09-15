import numpy as np
from ntc import to_numpy

__all__ = ['cost2matchvolume', 'grid2matchvolume']


def cost2matchvolume(costvolume):
    costvolume = to_numpy(costvolume)
    H1, W1, H2, W2 = costvolume.shape

    A = np.full((H1, W1, 2), -10, dtype=np.int32)
    B = np.full((H2, W2, 2), -10, dtype=np.int32)
    AB = np.full((H2, W2, 2), -10, dtype=np.int32)

    for h1 in range(H1):
        for w1 in range(W1):
            h2, w2 = np.unravel_index(np.argmin(costvolume[h1, w1, ...]), (H2, W2))
            print(h1, w1, h2, w2)
            A[h1, w1] = h2, w2

    for h2 in range(H2):
        for w2 in range(W2):
            h1, w1 = np.unravel_index(np.argmin(costvolume[..., h2, w2]), (H1, W1))
            B[h2, w2] = h1, w1

            if (h2, w2) == np.unravel_index(np.argmin(costvolume[h1, w1, ...]), (H2, W2)):
                AB[h2, w2] = h1, w1

    return AB, A, B


def grid2matchvolume(grid, L):
    x, y = grid[..., 0].copy(), grid[..., 1].copy()
    grid[..., 0], grid[..., 1] = y, x

    LH, LW, _ = grid.shape
    gridn = np.zeros((LH - 1, LW - 1, 2), dtype=np.float32)
    for h in range(LH - 1):
        for w in range(LW - 1):
            gridn[h, w, :] = grid[h: h + 2, w: w + 2].mean(axis=(0, 1))
    grid = gridn

    grid -= 1 / L
    grid += 1
    grid *= (L / 2)
    grid = np.round(grid).astype(np.int32)
    m = np.logical_or(grid >= L, grid < 0).any(axis=-1)
    grid[m] = -10

    return grid
