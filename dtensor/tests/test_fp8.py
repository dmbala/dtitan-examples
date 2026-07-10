from dtensor_workshop.fp8 import hopper_available, maybe_convert_fp8
from dtensor_workshop.model import build_block


def test_no_op_off_hopper():
    # On CPU CI, hopper_available() is False and the model is returned unchanged.
    if hopper_available():
        return  # on a Hopper GPU this path is exercised by the smoke run, not here
    block = build_block(dim=32, hidden=64, n_heads=4, seed=0)
    assert maybe_convert_fp8(block) is block


def test_hopper_available_is_bool():
    assert isinstance(hopper_available(), bool)
