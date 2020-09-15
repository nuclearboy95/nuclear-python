import torch

__all__ = ['calc_cost_euclidean', 'calc_cost_correlation']


def calc_cost_correlation(f_A, f_B):
    B, C, H, W = f_A.size()
    f_A = f_A.view(B, C, H * W).transpose(1, 2)  # size [b,c,h*w]
    f_B = f_B.view(B, C, H * W)  # size [b,c,h*w]
    f_mul = torch.bmm(f_A, f_B)

    corr = f_mul.view(B, H, W, H, W)
    return 1 - corr


def calc_cost_euclidean(f_A, f_B):
    N, C, H, W = f_A.size()

    f_As = f_A.view(N, C, H, W, 1, 1)
    f_Bs = f_B.view(N, C, 1, 1, H, W)
    diff = f_As - f_Bs
    dists = diff.norm(dim=1)

    return dists
