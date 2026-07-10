import torch

from dtensor_workshop.faultdata import ResumableLoader


def test_deterministic_and_resumable():
    a = ResumableLoader((4, 8), base_seed=7)
    first = [a.next() for _ in range(5)]
    saved = a.state_dict()          # after 5 steps

    # a fresh loader restored to the saved step resumes identically
    b = ResumableLoader((4, 8), base_seed=7)
    b.load_state_dict(saved)
    resumed = [b.next() for _ in range(3)]

    a_more = [a.next() for _ in range(3)]
    for r, e in zip(resumed, a_more):
        assert torch.equal(r, e)


def test_step_advances():
    loader = ResumableLoader((2, 2))
    assert loader.state_dict()["step"] == 0
    loader.next()
    assert loader.state_dict()["step"] == 1
