import pytest

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_shape_bug


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    with pytest.raises(RuntimeError):
        l1_shape_bug.buggy_matmul(mesh)
    out = l1_shape_bug.fixed_matmul(mesh)
    assert out == (256, 256)
    distenv.shutdown()


def test_bug_raises_and_fix_works():
    run_distributed(_worker, world_size=4)
