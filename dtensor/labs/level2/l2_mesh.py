from dtensor_workshop import distenv, mesh as mesh_mod, rlog


def mesh_coords(mesh) -> dict:
    return {"dp": mesh["dp"].get_local_rank(), "tp": mesh["tp"].get_local_rank()}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    coords = mesh_coords(mesh)
    rlog.info(f"mesh coordinate dp={coords['dp']} tp={coords['tp']}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
