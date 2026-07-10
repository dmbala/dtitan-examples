from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_capstone


def _worker(rank, world_size, ckpt_dir, trace_dir):
    import os
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    res = l2_capstone.run_capstone(
        mesh, ckpt_dir, os.path.join(trace_dir, f"trace_rank{rank}.json")
    )
    assert res["parity_maxdiff"] < 1e-4, res
    assert res["resume_maxdiff"] < 1e-6, res
    distenv.shutdown()


def test_capstone(tmp_path):
    run_distributed(
        _worker, world_size=4,
        args=(str(tmp_path / "ckpt"), str(tmp_path)),
    )
