import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def distributed_global_sum(mesh, rows: int = 2048, cols: int = 16, seed: int = 0):
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    result = (sharded * 2.0 + 1.0).sum()          # Partial placement
    replicated = result.redistribute(mesh, [Replicate()])  # all-reduce
    dist_sum = replicated.to_local().item()
    ref_sum = (full * 2.0 + 1.0).sum().item()
    return dist_sum, ref_sum


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        dist_sum, ref_sum = distributed_global_sum(mesh)
    if distenv.rank() == 0:
        prof.export_chrome_trace("artifacts/l1_capstone_trace.json")
    ok = abs(dist_sum - ref_sum) <= 1e-3 + 1e-4 * abs(ref_sum)
    rlog.info(f"distributed_sum={dist_sum:.4f} reference={ref_sum:.4f} parity_ok={ok}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
