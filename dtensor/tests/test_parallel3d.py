import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_fsdp, apply_hsdp_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _fsdp_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=1)(x).detach()
    model = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh)
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    assert isinstance(next(model.parameters()), DTensor)
    distenv.shutdown()


def _hsdp_tp_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=2)(x).detach()
    model = apply_hsdp_tp(build_block(dim=32, hidden=64, n_heads=4, seed=2), mesh)
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    distenv.shutdown()


def test_fsdp_parity():
    run_distributed(_fsdp_worker, world_size=2)


def test_hsdp_tp_parity():
    run_distributed(_hsdp_tp_worker, world_size=8)
