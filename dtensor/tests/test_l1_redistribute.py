from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_redistribute


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    diff = l1_redistribute.replicate_max_diff(mesh, rows=1024, cols=8)
    assert diff == 0.0, diff
    distenv.shutdown()


def test_replicate_max_diff_is_zero():
    run_distributed(_worker, world_size=4)
