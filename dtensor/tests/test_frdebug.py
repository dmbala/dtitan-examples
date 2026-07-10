import torch

from dtensor_workshop import distenv
from dtensor_workshop.frdebug import shapes_agree
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed


def _agree_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    group = mesh.get_group()
    same = torch.zeros(8)                       # identical on every rank
    assert shapes_agree(same, group) is True
    differ = torch.zeros(rank + 1)              # rank-dependent size -> disagreement
    assert shapes_agree(differ, group) is False
    distenv.shutdown()


def test_shapes_agree_detects_mismatch():
    run_distributed(_agree_worker, world_size=2)
