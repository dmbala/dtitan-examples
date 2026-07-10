import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from dtensor_workshop.acheckpoint import forward_maybe_checkpointed


def average_gradients(model, dp_mesh) -> None:
    group = dp_mesh.get_group()
    size = dp_mesh.size()
    for p in model.parameters():
        if p.grad is None:
            continue
        local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        local /= size


def run_training(model, batches, optimizer, dp_mesh=None, use_ac=False):
    losses = []
    for batch in batches:
        optimizer.zero_grad()
        loss = forward_maybe_checkpointed(model, batch, use_ac).pow(2).mean()
        loss.backward()
        if dp_mesh is not None:
            average_gradients(model, dp_mesh)
            lt = loss.detach().reshape(1).clone()
            dist.all_reduce(lt, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
            losses.append((lt / dp_mesh.size()).item())
        else:
            losses.append(loss.item())
        optimizer.step()
    return losses
