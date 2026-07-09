import os

import torch
import torch.distributed as dist


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_process_group(backend: str | None = None) -> str:
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return backend


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
