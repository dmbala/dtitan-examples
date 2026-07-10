import torch

from dtensor_workshop.moe import MoEFeedForward, routing_imbalance


def test_all_tokens_routed():
    torch.manual_seed(0)
    moe = MoEFeedForward(dim=32, hidden=64, n_experts=4)
    out, counts = moe(torch.randn(64, 32))
    assert tuple(out.shape) == (64, 32)
    assert sum(counts) == 64
    assert len(counts) == 4


def test_routing_imbalance():
    assert routing_imbalance([10, 10, 10, 10]) == 1.0
    assert routing_imbalance([40, 0, 0, 0]) == 4.0
    assert routing_imbalance([0, 0, 0, 0]) == 0.0
