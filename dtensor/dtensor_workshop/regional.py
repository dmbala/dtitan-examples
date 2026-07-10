import torch


def regional_compile(module):
    return torch.compile(module)
