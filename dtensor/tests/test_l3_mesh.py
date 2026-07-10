from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_mesh


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    roles = l3_mesh.mesh_roles(mesh)
    assert set(roles) == {"dp_replicate", "dp_shard", "tp"}
    assert all(0 <= v <= 1 for v in roles.values())
    four_d = l3_mesh.build_illustrative_4d()
    assert four_d.ndim == 4
    distenv.shutdown()


def test_mesh_roles_and_4d():
    run_distributed(_worker, world_size=8)
