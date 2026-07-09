import os
import socket

import torch.multiprocessing as mp


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank, world_size, port, worker, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    worker(rank, world_size, *args)


def run_distributed(worker, world_size: int = 4, args: tuple = ()) -> None:
    port = _free_port()
    mp.spawn(_entry, args=(world_size, port, worker, args),
             nprocs=world_size, join=True)
