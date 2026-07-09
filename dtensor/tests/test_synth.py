import torch

from dtensor_workshop import synth


def test_shape_and_dtype():
    t = synth.synthetic_tensor((8, 4))
    assert tuple(t.shape) == (8, 4)
    assert t.dtype == torch.float32


def test_deterministic_same_seed():
    a = synth.synthetic_tensor((16, 3), seed=7)
    b = synth.synthetic_tensor((16, 3), seed=7)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = synth.synthetic_tensor((16, 3), seed=1)
    b = synth.synthetic_tensor((16, 3), seed=2)
    assert not torch.equal(a, b)
