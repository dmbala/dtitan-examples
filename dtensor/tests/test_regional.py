import torch

from dtensor_workshop.model import build_block
from dtensor_workshop.regional import regional_compile


def test_compiled_matches_eager():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32)
    eager = block(x)
    compiled = regional_compile(block)(x)
    assert torch.allclose(eager, compiled, atol=1e-4), (eager - compiled).abs().max().item()
