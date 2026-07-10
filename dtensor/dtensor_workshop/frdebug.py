import torch
import torch.distributed as dist


def shapes_agree(tensor, group) -> bool:
    """Return whether all ranks in `group` agree on `tensor`'s element count.

    Uses a fixed-size scalar all-gather (never itself mismatches), so it is a
    hang-free preflight for the collective-size disagreements that would
    otherwise hang an NCCL collective. Compares element counts, not shapes.
    """
    world = dist.get_world_size(group)
    device = tensor.device
    local = torch.tensor([float(tensor.numel())], device=device)
    gathered = [torch.zeros(1, device=device) for _ in range(world)]
    dist.all_gather(gathered, local, group=group)
    values = [g.item() for g in gathered]
    return all(v == values[0] for v in values)


def dump_flight_recorder(path) -> bool:
    if not torch.cuda.is_available():
        return False
    from torch._C._distributed_c10d import _dump_nccl_trace
    with open(path, "wb") as fh:
        fh.write(_dump_nccl_trace())
    return True
