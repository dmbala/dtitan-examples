import torch

from dtensor_workshop.acheckpoint import forward_maybe_checkpointed
from dtensor_workshop.model import build_block


def test_ac_matches_plain_forward():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    plain = forward_maybe_checkpointed(block, x, use_ac=False)
    ac = forward_maybe_checkpointed(block, x, use_ac=True)
    assert torch.allclose(plain, ac, atol=1e-6), (plain - ac).abs().max().item()


def test_ac_backward_runs():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    forward_maybe_checkpointed(block, x, use_ac=True).sum().backward()
    assert block.fc1.weight.grad is not None
