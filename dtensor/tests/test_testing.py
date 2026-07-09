import pytest

from dtensor_workshop import distenv
from dtensor_workshop.testing import run_distributed


def _allreduce_worker(rank, world_size):
    import torch
    import torch.distributed as dist
    distenv.init_process_group("gloo")
    t = torch.tensor([float(rank)])
    dist.all_reduce(t)
    expected = float(sum(range(world_size)))
    assert t.item() == expected, (t.item(), expected)
    distenv.shutdown()


def _failing_worker(rank, world_size):
    distenv.init_process_group("gloo")
    try:
        assert rank != world_size - 1, "boom on last rank"
    finally:
        distenv.shutdown()


def test_run_distributed_allreduce():
    run_distributed(_allreduce_worker, world_size=4)


def test_run_distributed_propagates_failure():
    with pytest.raises(Exception):
        run_distributed(_failing_worker, world_size=2)
