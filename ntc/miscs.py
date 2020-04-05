import torch


__all__ = ['abstain_loss']


def abstain_loss(inputs, targets, abstain_coef):
    probs = inputs.softmax(dim=-1)
    probs, reserve = probs[..., :-1], probs[..., -1:]
    targets = targets.unsqueeze(-1)
    gain = torch.gather(probs, dim=-1, index=targets).squeeze()
    doubling_rate = (gain + reserve / abstain_coef).log()
    return -doubling_rate.mean()
