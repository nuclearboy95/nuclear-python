import numpy as np
from .utils_primitive import ranges
from ..calc import cnn_output_size


__all__ = ['score2mask', 'avgpool2d',
           'upsample_scoremask', 'upsample_scoremasks', 'get_contiguous_blocks']


def score2mask(H, W, K, Hs, Ws, scores) -> np.ndarray:
    """

    :param int H:
    :param int W:
    :param int K:
    :param Hs:
    :param Ws:
    :param scores:
    :return:
    """
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)
    for h, w, s in zip(Hs, Ws, scores):
        mask[h: h + K, w: w + K] += s
        cnt[h: h + K, w: w + K] += 1
    cnt[cnt == 0] = 1  # avoid divide by zero
    return mask / cnt


def upsample_scoremask(score_mask, output_shape, K: int, S: int) -> np.ndarray:
    H, W = output_shape
    mask = np.zeros([H, W], dtype=np.float32)
    cnt = np.zeros([H, W], dtype=np.int32)

    I, J = score_mask.shape[:2]
    for i, j in ranges(I, J):
        h, w = i * S, j * S

        mask[h: h + K, w: w + K] += score_mask[i, j]
        cnt[h: h + K, w: w + K] += 1

    cnt[cnt == 0] = 1

    return mask / cnt


def upsample_scoremasks(score_masks, output_shape, K: int, S: int) -> np.ndarray:
    N = score_masks.shape[0]
    results = [upsample_scoremask(score_masks[n], output_shape, K, S) for n in range(N)]
    return np.asarray(results)


def avgpool2d(x, K, S):
    H, W = x.shape[:2]
    I = cnn_output_size(H, K, S)
    J = cnn_output_size(W, K, S)
    ret = np.zeros((I, J), dtype=x.dtype)
    for i, j in ranges(I, J):
        h = i * S
        w = j * S
        p = x[h: h + K, w: w + K]
        ret[i, j] = p.mean()
    return ret


def get_contiguous_blocks(arr) -> list:
    """

    :param np.ndarray arr: boolean list.
    :return:
    """
    N = len(arr)
    blocks = list()

    on = False
    start = -1
    for n in range(N):
        if on:
            if not arr[n]:
                block = (start, n - 1)
                blocks.append(block)
                on = False
        else:
            if arr[n]:
                start = n
                on = True
    else:
        if on:
            block = (start, N - 1)
            blocks.append(block)

    return blocks
