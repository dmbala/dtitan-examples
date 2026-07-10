import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_fsdp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def fsdp_resume_maxdiff(mesh, checkpoint_id, steps=2, reshard_after_forward=True):
    device = mesh.device_type
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0)).to(device)
    batches = [x for _ in range(steps)]

    orig = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1).to(device),
                      mesh, reshard_after_forward=reshard_after_forward)
    orig_opt = torch.optim.SGD(orig.parameters(), lr=0.1, momentum=0.9)
    run_training(orig, batches, orig_opt)
    dcp_save(orig, orig_opt, checkpoint_id)

    restored = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1001).to(device),
                          mesh, reshard_after_forward=reshard_after_forward)
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    dcp_load(restored, restored_opt, checkpoint_id)

    run_training(orig, [x], orig_opt)
    run_training(restored, [x], restored_opt)
    return (_full(orig(x)) - _full(restored(x))).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp_replicate", "dp_shard"))
    diff = fsdp_resume_maxdiff(mesh, "checkpoints/l3_fsdp")
    rlog.info(f"FSDP2 resume-after-restore max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
