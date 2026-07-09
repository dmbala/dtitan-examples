import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def profiled_run(mesh, out_path: str) -> int:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    sharded = distribute_tensor(synth.synthetic_tensor((4096, 32)), mesh, [Shard(0)])
    with profile(activities=activities) as prof:
        scaled = sharded * 2.0
        gathered = scaled.redistribute(mesh, [Replicate()])  # all-gather = comm op
        _ = gathered.to_local().sum()
    prof.export_chrome_trace(out_path)
    return len(prof.key_averages())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    n = profiled_run(mesh, "artifacts/l1_trace.json")
    rlog.info(f"exported trace with {n} profiled event rows to artifacts/l1_trace.json")
    distenv.shutdown()


if __name__ == "__main__":
    main()
