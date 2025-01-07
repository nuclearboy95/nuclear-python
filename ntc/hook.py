import torch
import torch.nn as nn
from typing import Callable
from . import hookf


__all__ = ["register", "register_inout"]


def register(module: nn.Module, hook: Callable):
    return module.register_forward_hook(hook, with_kwargs=True)


def register_inout_log(module: nn.Module):
    return register(module, hookf.log_post())
