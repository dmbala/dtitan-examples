import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_nccl_debug


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    assert l3_nccl_debug.diagnose_batch(torch.zeros(8), mesh) is True     # agree
    assert l3_nccl_debug.diagnose_batch(torch.zeros(rank + 1), mesh) is False  # disagree
    distenv.shutdown()


def test_diagnose_batch():
    run_distributed(_worker, world_size=2)
