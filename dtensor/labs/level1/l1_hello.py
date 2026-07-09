import torch
import torch.distributed as dist

from dtensor_workshop import distenv, rlog


def main():
    distenv.init_process_group()
    rlog.info(
        f"world_size={distenv.world_size()} local_rank={distenv.local_rank()} "
        f"cuda={torch.cuda.is_available()}"
    )
    dist.barrier()
    if distenv.rank() == 0:
        rlog.info("all ranks reached the barrier")
    distenv.shutdown()


if __name__ == "__main__":
    main()
