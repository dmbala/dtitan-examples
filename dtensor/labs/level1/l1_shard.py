from torch.distributed.tensor import Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def shard_report(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> dict:
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    dtensor = distribute_tensor(full, mesh, [Shard(0)])
    return {
        "local_shape": tuple(dtensor.to_local().shape),
        "global_shape": tuple(dtensor.shape),
    }


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    rep = shard_report(mesh)
    rlog.info(f"local={rep['local_shape']} global={rep['global_shape']}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
