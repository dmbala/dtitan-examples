from torch.distributed.fsdp import fully_shard

from .tp import apply_tp


def apply_fsdp(model, mesh, reshard_after_forward: bool = True):
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


def apply_hsdp_tp(model, mesh):
    apply_tp(model, mesh["tp"])
    fully_shard(model, mesh=mesh["dp_replicate", "dp_shard"])
    return model
