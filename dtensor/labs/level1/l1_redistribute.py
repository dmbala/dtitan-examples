from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def replicate_max_diff(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> float:
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    replicated = sharded.redistribute(mesh, [Replicate()])
    return (replicated.to_local() - full).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    diff = replicate_max_diff(mesh)
    rlog.info(f"replicate max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
