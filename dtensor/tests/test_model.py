import pytest
import torch

from dtensor_workshop.model import TransformerBlock, build_block


def test_forward_shape():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=0)
    out = block(torch.randn(2, 8, 32))
    assert tuple(out.shape) == (2, 8, 32)


def test_deterministic_build():
    a = build_block(dim=32, hidden=64, n_heads=4, seed=7)
    b = build_block(dim=32, hidden=64, n_heads=4, seed=7)
    x = torch.randn(2, 8, 32)
    assert torch.equal(a(x), b(x))


def test_head_divisibility_guard():
    with pytest.raises(AssertionError):
        TransformerBlock(dim=32, hidden=64, n_heads=5)
