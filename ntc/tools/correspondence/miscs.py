import numpy as np
from ntc import to_numpy

__all__ = ['cost2matchvolume', 'grid2matchlabel']


def cost2matchvolume(costvolume):
    costvolume = to_numpy(costvolume)
    N, H1, W1, H2, W2 = costvolume.shape

    A = np.full((N, H1, W1, 2), -10, dtype=np.int32)
    B = np.full((N, H2, W2, 2), -10, dtype=np.int32)
    AB = np.full((N, H2, W2, 2), -10, dtype=np.int32)

    for n in range(N):
        for h1 in range(H1):
            for w1 in range(W1):
                h2, w2 = np.unravel_index(np.argmin(costvolume[n, h1, w1, ...]), (H2, W2))
                A[n, h1, w1] = h2, w2

        for h2 in range(H2):
            for w2 in range(W2):
                h1, w1 = np.unravel_index(np.argmin(costvolume[n, ..., h2, w2]), (H1, W1))
                B[n, h2, w2] = h1, w1

                if (h2, w2) == np.unravel_index(np.argmin(costvolume[n, h1, w1, ...]), (H2, W2)):
                    AB[n, h2, w2] = h1, w1

    return AB, A, B


def grid2matchlabel(grid, L, oneD=False):
    x, y = grid[..., 0].copy(), grid[..., 1].copy()
    grid[..., 0], grid[..., 1] = y, x

    N, LH, LW, _ = grid.shape
    matchlabel = np.zeros((N, LH - 1, LW - 1, 2), dtype=np.float32)
    for h in range(LH - 1):
        for w in range(LW - 1):
            matchlabel[:, h, w, :] = grid[:, h: h + 2, w: w + 2].mean(axis=(1, 2))

    matchlabel -= 1 / L
    matchlabel += 1
    matchlabel *= (L / 2)
    matchlabel = np.round(matchlabel).astype(np.int32)
    m = np.logical_or(matchlabel >= L, matchlabel < 0).any(axis=-1)
    matchlabel[m] = -10

    if oneD:
        matchlabel2 = np.zeros((N, LH - 1, LW - 1), dtype=np.int32)
        for n in range(N):
            for h in range(LH - 1):
                for w in range(LW - 1):
                    v1, v2 = matchlabel[n, h, w]
                    if v1 == -10:
                        matchlabel2[n, h, w] = -10
                    else:
                        matchlabel2[n, h, w] = v1 * (LH - 1) + v2  # IS THIS RIGHT ORDER?
        matchlabel = matchlabel2

    return matchlabel
