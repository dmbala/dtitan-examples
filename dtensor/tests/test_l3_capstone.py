from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_capstone


def _worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp_replicate", "dp_shard"), device_type="cpu")
    res = l3_capstone.run_capstone(mesh, ckpt_dir)
    assert res["resume_maxdiff"] < 1e-6, res
    assert res["imbalance"] >= 1.0
    assert res["steps"] == 3
    distenv.shutdown()


def test_capstone(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path / "ckpt"),))
