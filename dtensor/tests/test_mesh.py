from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed


def _mesh_1d_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    assert mesh.ndim == 1
    assert mesh.size() == world_size
    assert mesh.mesh_dim_names == ("dp",)
    distenv.shutdown()


def _mesh_2d_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    assert mesh.ndim == 2
    assert mesh["dp"].size() == 2
    assert mesh["tp"].size() == 2
    distenv.shutdown()


def test_build_1d_mesh():
    run_distributed(_mesh_1d_worker, world_size=4)


def test_build_2d_mesh():
    run_distributed(_mesh_2d_worker, world_size=4)
