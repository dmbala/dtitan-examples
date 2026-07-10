from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_moe


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    rep = l3_moe.moe_report(mesh, tokens=64, n_experts=4)
    assert rep["routed"] == 64
    assert len(rep["counts"]) == 4
    assert rep["imbalance"] >= 1.0
    distenv.shutdown()


def test_moe_report():
    run_distributed(_worker, world_size=2)
