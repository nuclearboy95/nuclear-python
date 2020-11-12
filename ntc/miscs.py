import torch
import numpy as np

__all__ = ['abstain_loss', 'calc_correct', 'to_device', 'to_numpy']


def abstain_loss(logits, targets, abstain_coef):
    probs = logits.softmax(dim=-1)
    probs, reserve = probs[..., :-1], probs[..., -1:]
    targets = targets.unsqueeze(-1)
    gain = torch.gather(probs, dim=-1, index=targets).squeeze()
    doubling_rate = (gain + reserve / abstain_coef).log()
    return -doubling_rate.mean()


def calc_correct(outputs, targets):
    preds = outputs.argmax(dim=-1, keepdim=True)
    correct = preds.eq(targets.view_as(preds)).detach().cpu().numpy()
    return correct


def to_device(obj, device, non_blocking=False):
    """

    :param obj:
    :param device:
    :param bool non_blocking:
    :return:
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    elif isinstance(obj, dict):
        return {k: to_device(v, device, non_blocking=non_blocking)
                for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_device(v, device, non_blocking=non_blocking)
                for v in obj]

    elif isinstance(obj, tuple):
        return tuple([to_device(v, device, non_blocking=non_blocking)
                      for v in obj])

    else:
        raise TypeError('Unknown type.')


def to_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    elif isinstance(obj, np.ndarray):
        return obj

    elif isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_numpy(v) for v in obj]

    elif isinstance(obj, tuple):
        return tuple([to_numpy(v) for v in obj])

    else:
        raise TypeError('Unknown type.')
