import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _parity_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    x = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=1)(x).detach()
    tp_block = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh["tp"])
    out = tp_block(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    distenv.shutdown()


def test_tp_block_matches_single_device():
    run_distributed(_parity_worker, world_size=4)
