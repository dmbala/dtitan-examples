from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_mesh


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    coords = l2_mesh.mesh_coords(mesh)
    assert coords["dp"] in (0, 1) and coords["tp"] in (0, 1)
    # rank r -> (dp=r//2, tp=r%2) for a row-major (dp, tp) mesh
    assert coords["dp"] == rank // 2
    assert coords["tp"] == rank % 2
    distenv.shutdown()


def test_mesh_coords():
    run_distributed(_worker, world_size=4)
