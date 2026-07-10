import torch
import torch.distributed as dist


def shapes_agree(tensor, group) -> bool:
    world = dist.get_world_size(group)
    local = torch.tensor([float(tensor.numel())])
    gathered = [torch.zeros(1) for _ in range(world)]
    dist.all_gather(gathered, local, group=group)
    values = [g.item() for g in gathered]
    return all(v == values[0] for v in values)


def dump_flight_recorder(path) -> bool:
    if not torch.cuda.is_available():
        return False
    from torch.distributed import _dump_nccl_trace
    with open(path, "wb") as fh:
        fh.write(_dump_nccl_trace())
    return True
