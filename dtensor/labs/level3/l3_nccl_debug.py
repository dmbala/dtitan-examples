import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.frdebug import dump_flight_recorder, shapes_agree


def diagnose_batch(local_batch, mesh) -> bool:
    return shapes_agree(local_batch, mesh.get_group())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    # intentional per-rank shape disagreement (the collective-mismatch failure mode)
    agree = diagnose_batch(torch.zeros(distenv.rank() + 1).to(mesh.device_type), mesh)
    rlog.info(f"batch shapes agree across ranks = {agree}")
    dumped = dump_flight_recorder(f"artifacts/l3_flight_recorder_rank{distenv.rank()}.dump")
    rlog.info(f"flight recorder dumped = {dumped}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
