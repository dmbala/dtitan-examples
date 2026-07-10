import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _global_batch(dim, generator):
    return torch.randn(4, 8, dim, generator=generator)


def parallel_and_baseline_losses(mesh, steps=4, dim=32, hidden=64, n_heads=4, seed=5):
    device = mesh.device_type
    gen = torch.Generator().manual_seed(2024)
    global_batch = _global_batch(dim, gen).to(device)
    batches = [global_batch for _ in range(steps)]

    # single-device baseline on the full batch
    base_model = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device)
    base_opt = torch.optim.SGD(base_model.parameters(), lr=0.1)
    baseline = run_training(base_model, batches, base_opt, dp_mesh=None)

    # 2D parallel: TP block, dp replica takes its half of each global batch
    dp = mesh["dp"]
    lo = dp.get_local_rank() * 2
    par_batches = [b[lo:lo + 2] for b in batches]
    par_model = apply_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device), mesh["tp"]
    )
    par_opt = torch.optim.SGD(par_model.parameters(), lr=0.1)
    parallel = run_training(par_model, par_batches, par_opt, dp_mesh=dp)
    return parallel, baseline


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    par, base = parallel_and_baseline_losses(mesh)
    maxdiff = max(abs(a - b) for a, b in zip(par, base))
    rlog.info(f"2D-parallel vs single-device loss max abs diff = {maxdiff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
