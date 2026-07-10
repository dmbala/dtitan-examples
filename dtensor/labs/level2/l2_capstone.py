import torch
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def run_capstone(mesh, checkpoint_id, trace_path, steps=4, dim=32, hidden=64, n_heads=4, seed=8):
    device = mesh.device_type
    dp = mesh["dp"]
    gen = torch.Generator().manual_seed(2024)
    global_batch = torch.randn(4, 8, dim, generator=gen).to(device)
    batches = [global_batch for _ in range(steps)]

    # single-device baseline
    base = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device)
    base_opt = torch.optim.SGD(base.parameters(), lr=0.1)
    baseline = run_training(base, batches, base_opt, dp_mesh=None)

    # 2D-parallel training with a profiler trace
    lo = dp.get_local_rank() * 2
    par_batches = [b[lo:lo + 2] for b in batches]
    model = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device), mesh["tp"])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        # activation checkpointing = the capstone's memory optimization
        parallel = run_training(model, par_batches, opt, dp_mesh=dp, use_ac=True)
    prof.export_chrome_trace(trace_path)

    dcp_save(model, opt, checkpoint_id)
    restored = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed + 1000).to(device), mesh["tp"])
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1)
    dcp_load(restored, restored_opt, checkpoint_id)

    parity = max(abs(a - b) for a, b in zip(parallel, baseline))
    resume = (_full(model(global_batch[lo:lo + 2])) - _full(restored(global_batch[lo:lo + 2]))).abs().max().item()
    return {"parity_maxdiff": parity, "resume_maxdiff": resume}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    res = run_capstone(
        mesh, "checkpoints/l2_capstone",
        f"artifacts/l2_capstone_trace_rank{distenv.rank()}.json",
    )
    rlog.info(f"parity_maxdiff={res['parity_maxdiff']:.2e} resume_maxdiff={res['resume_maxdiff']:.2e}")
    if distenv.rank() == 0:
        rlog.info(
            "DIAGNOSIS (fill in): dominant bottleneck? "
            "which collective (all-gather/reduce-scatter/all-reduce)? "
            "peak memory before vs after activation checkpointing?"
        )
    distenv.shutdown()


if __name__ == "__main__":
    main()
