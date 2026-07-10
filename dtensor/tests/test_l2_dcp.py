from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_dcp


def _worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    diff = l2_dcp.save_restore_resume_maxdiff(mesh, ckpt_dir)
    assert diff < 1e-6, diff
    distenv.shutdown()


def test_dcp_resume(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path / "l2ckpt"),))
