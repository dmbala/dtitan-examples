from . import distenv


def prefix() -> str:
    return f"[rank {distenv.rank()}/{distenv.world_size()}]"


def info(msg: str) -> None:
    print(f"{prefix()} {msg}", flush=True)
