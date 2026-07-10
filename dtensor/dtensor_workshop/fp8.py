import torch


def hopper_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def maybe_convert_fp8(model):
    if not hopper_available():
        return model
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    return convert_to_float8_training(model, config=Float8LinearConfig())
