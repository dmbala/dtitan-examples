from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_tp_block


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    diff = l2_tp_block.tp_parity_maxdiff(mesh)
    assert diff < 1e-4, diff
    distenv.shutdown()


def test_tp_parity():
    run_distributed(_worker, world_size=4)
