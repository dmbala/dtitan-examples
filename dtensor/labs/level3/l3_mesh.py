from dtensor_workshop import distenv, mesh as mesh_mod, rlog


def mesh_roles(mesh) -> dict:
    return {
        "dp_replicate": mesh["dp_replicate"].get_local_rank(),
        "dp_shard": mesh["dp_shard"].get_local_rank(),
        "tp": mesh["tp"].get_local_rank(),
    }


def build_illustrative_4d():
    # pp is size 1: 8 GPUs cannot exercise a real 4D layout — code-that-would-scale only.
    return mesh_mod.build_mesh((2, 2, 2, 1), ("dp_replicate", "dp_shard", "tp", "pp"))


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"))
    roles = mesh_roles(mesh)
    rlog.info(
        f"roles={roles} | mapping: dp_replicate across nodes (inter-node link), "
        f"dp_shard+tp within a node (NVLink)"
    )
    distenv.shutdown()


if __name__ == "__main__":
    main()
