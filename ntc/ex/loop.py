__all__ = ["loop"]


def loop():
    import torch

    x = torch.randn((1024, 1024)).cuda()
    y = torch.randn((1024, 1024)).cuda()
    while True:
        z = x @ y
