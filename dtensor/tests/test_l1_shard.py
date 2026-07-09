from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_shard


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    rep = l1_shard.shard_report(mesh, rows=1024, cols=8)
    assert rep["global_shape"] == (1024, 8)
    assert rep["local_shape"] == (1024 // world_size, 8)
    distenv.shutdown()


def test_shard_report():
    run_distributed(_worker, world_size=4)
