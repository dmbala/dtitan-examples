from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_hsdp_tp


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    diff = l3_hsdp_tp.hsdp_tp_parity_maxdiff(mesh)
    assert diff < 1e-4, diff
    distenv.shutdown()


def test_hsdp_tp_parity():
    run_distributed(_worker, world_size=8)
