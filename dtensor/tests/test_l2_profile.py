import os

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_profile


def _worker(rank, world_size, out_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    out_path = os.path.join(out_dir, f"l2_trace_rank{rank}.json")
    n = l2_profile.profiled_2d_step(mesh, out_path)
    assert n > 0
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    distenv.shutdown()


def test_profiled_2d_step(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path),))
