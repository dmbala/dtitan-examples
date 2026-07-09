import os

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_profile


def _worker(rank, world_size, out_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    out_path = os.path.join(out_dir, f"trace_rank{rank}.json")
    n_events = l1_profile.profiled_run(mesh, out_path)
    assert n_events > 0
    if rank == 0:
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    distenv.shutdown()


def test_profiled_run(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path),))
