
__all__ = ['softmax']


def softmax(logits):
    exps = (logits - logits.max()).exp()
    return exps / exps.sum()
