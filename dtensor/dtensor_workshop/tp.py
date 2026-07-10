from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

TP_PLAN = {
    "q": ColwiseParallel(),
    "k": ColwiseParallel(),
    "v": ColwiseParallel(),
    "o": RowwiseParallel(),
    "fc1": ColwiseParallel(),
    "fc2": RowwiseParallel(),
}


def apply_tp(block, tp_mesh):
    tp_size = tp_mesh.size()
    if block.n_heads % tp_size != 0:
        raise ValueError(f"n_heads {block.n_heads} not divisible by tp size {tp_size}")
    parallelize_module(block, tp_mesh, TP_PLAN)
    return block
