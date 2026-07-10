import torch


def hopper_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def _fp8_convertible(module, fqn: str) -> bool:
    return (
        isinstance(module, torch.nn.Linear)
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


def maybe_convert_fp8(model):
    if not hopper_available():
        return model
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    return convert_to_float8_training(model, module_filter_fn=_fp8_convertible, config=Float8LinearConfig())
