import torch
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def profiled_2d_step(mesh, out_path) -> int:
    device = mesh.device_type
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    block = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1).to(device), mesh["tp"])
    opt = torch.optim.SGD(block.parameters(), lr=0.1)
    lo = mesh["dp"].get_local_rank() * 2
    batch = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0)).to(device)[lo:lo + 2]
    with profile(activities=activities) as prof:
        run_training(block, [batch], opt, dp_mesh=mesh["dp"])
    prof.export_chrome_trace(out_path)
    return len(prof.key_averages())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    out_path = f"artifacts/l2_trace_rank{distenv.rank()}.json"
    n = profiled_2d_step(mesh, out_path)
    rlog.info(f"exported {n} profiled event rows to {out_path}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
