import math

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_capstone


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    dist_sum, ref_sum = l1_capstone.distributed_global_sum(mesh)
    assert math.isclose(dist_sum, ref_sum, rel_tol=1e-4, abs_tol=1e-3), (dist_sum, ref_sum)
    distenv.shutdown()


def test_global_sum_matches_reference():
    run_distributed(_worker, world_size=4)
