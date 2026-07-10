from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_train


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    par, base = l2_train.parallel_and_baseline_losses(mesh, steps=4)
    assert len(par) == len(base) == 4
    maxdiff = max(abs(a - b) for a, b in zip(par, base))
    assert maxdiff < 1e-4, (maxdiff, par, base)
    distenv.shutdown()


def test_2d_loss_parity():
    run_distributed(_worker, world_size=4)
