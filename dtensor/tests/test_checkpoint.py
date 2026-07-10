import torch

from dtensor_workshop import distenv
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _roundtrip_worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    x = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(0))

    m1 = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh["tp"])
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1, momentum=0.9)
    m1(x).pow(2).mean().backward()
    opt1.step()
    dcp_save(m1, opt1, ckpt_dir)

    m2 = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=999), mesh["tp"])  # different init
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1, momentum=0.9)
    dcp_load(m2, opt2, ckpt_dir)

    o1, o2 = m1(x), m2(x)
    o1 = o1.full_tensor() if isinstance(o1, DTensor) else o1
    o2 = o2.full_tensor() if isinstance(o2, DTensor) else o2
    assert torch.allclose(o1, o2, atol=1e-6), (o1 - o2).abs().max().item()
    distenv.shutdown()


def test_dcp_roundtrip(tmp_path):
    run_distributed(_roundtrip_worker, world_size=4, args=(str(tmp_path / "ckpt"),))
