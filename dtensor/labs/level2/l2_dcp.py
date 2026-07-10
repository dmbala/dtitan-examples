import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def save_restore_resume_maxdiff(mesh, checkpoint_id, steps=2, dim=32, hidden=64, n_heads=4, seed=6):
    x = torch.randn(2, 8, dim, generator=torch.Generator().manual_seed(0))
    batches = [x for _ in range(steps)]

    orig = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"])
    orig_opt = torch.optim.SGD(orig.parameters(), lr=0.1)
    run_training(orig, batches, orig_opt, dp_mesh=mesh["dp"])
    dcp_save(orig, orig_opt, checkpoint_id)

    restored = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=123), mesh["tp"])
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1)
    dcp_load(restored, restored_opt, checkpoint_id)

    # one more identical step on both; outputs should stay identical
    run_training(orig, [x], orig_opt, dp_mesh=mesh["dp"])
    run_training(restored, [x], restored_opt, dp_mesh=mesh["dp"])
    return (_full(orig(x)) - _full(restored(x))).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    diff = save_restore_resume_maxdiff(mesh, "checkpoints/l2_dcp")
    rlog.info(f"resume-after-restore output max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
