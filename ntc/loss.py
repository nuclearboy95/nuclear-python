import torch
import torch.nn as nn


__all__ = ['AbstainLoss', 'LqLoss']


class AbstainLoss(nn.Module):
    def __init__(self, abstain_coef):
        super().__init__()
        self.abstain_coef = abstain_coef

    def forward(self, probs, targets):
        probs, reserve = probs[..., :-1], probs[..., -1:]
        targets = targets.unsqueeze(-1)
        gain = torch.gather(probs, dim=-1, index=targets).squeeze()
        doubling_rate = (gain + reserve / self.abstain_coef).log()
        return -doubling_rate.mean()


class LqLoss(nn.Module):
    def __init__(self, q):  # q is (0, 1]. q=1: MAE / q->0: CrossEntropy
        super().__init__()
        self.q = q

    def forward(self, probs, targets):
        q = self.q
        probs_y = torch.gather(probs, dim=-1, index=targets.unsqueeze(-1))
        loss = (1 - torch.pow(probs_y, q)) / q
        return loss.mean()
