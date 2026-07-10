import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from dtensor_workshop.train import average_gradients, run_training


def _avg_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    dp = mesh["dp"]
    lin = torch.nn.Linear(4, 1)
    # give each dp replica a distinct grad; tp-peers share a value
    for p in lin.parameters():
        p.grad = torch.full_like(p, float(dp.get_local_rank() + 1))  # 1.0 or 2.0
    average_gradients(lin, dp)
    for p in lin.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 1.5)), p.grad  # mean(1,2)
    distenv.shutdown()


def test_average_gradients_means_over_dp():
    run_distributed(_avg_worker, world_size=4)


def _run_training_worker(rank, world_size):
    distenv.init_process_group("gloo")
    lin = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    batches = [torch.randn(3, 4, generator=torch.Generator().manual_seed(s)) for s in range(3)]
    losses = run_training(lin, batches, opt, dp_mesh=None)
    assert len(losses) == 3
    assert losses[1] != losses[0]  # loss changes as params update
    distenv.shutdown()


def test_run_training_returns_losses():
    run_distributed(_run_training_worker, world_size=4)
