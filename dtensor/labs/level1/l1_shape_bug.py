import sys

from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def buggy_matmul(mesh):
    # BUG: left operand is (256, 128), right operand is (64, 256) -> the
    # contracting dims (128 vs 64) do not match, so matmul raises.
    left = distribute_tensor(synth.synthetic_tensor((256, 128), seed=1), mesh, [Shard(0)])
    right = distribute_tensor(synth.synthetic_tensor((64, 256), seed=2), mesh, [Replicate()])
    return (left @ right).to_local()


def fixed_matmul(mesh):
    # FIX: make the contracting dims agree (128 == 128).
    left = distribute_tensor(synth.synthetic_tensor((256, 128), seed=1), mesh, [Shard(0)])
    right = distribute_tensor(synth.synthetic_tensor((128, 256), seed=2), mesh, [Replicate()])
    out = left @ right
    return tuple(out.shape)


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    if "--bug" in sys.argv:
        buggy_matmul(mesh)
    else:
        rlog.info(f"fixed matmul global shape = {fixed_matmul(mesh)}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
