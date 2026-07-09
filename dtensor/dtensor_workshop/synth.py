import torch


def synthetic_tensor(shape, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)
