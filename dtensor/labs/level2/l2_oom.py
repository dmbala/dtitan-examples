import sys

import torch

from dtensor_workshop import distenv, rlog
from dtensor_workshop.acheckpoint import forward_maybe_checkpointed
from dtensor_workshop.model import build_block

SIZES = {
    "small": dict(dim=512, hidden=2048, n_heads=8, batch=8, seq=512),
    # Tune upward on H200 (141 GB) if OOM does not trigger.
    "big": dict(dim=2048, hidden=8192, n_heads=16, batch=16, seq=2048),
}


def ac_equivalence_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float:
    block = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    x = torch.randn(2, 8, dim, requires_grad=True)
    plain = forward_maybe_checkpointed(block, x, use_ac=False)
    ac = forward_maybe_checkpointed(block, x, use_ac=True)
    return (plain - ac).abs().max().item()


def start_memory_record() -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)


def dump_memory_snapshot(path) -> None:
    torch.cuda.memory._dump_snapshot(path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    size = "big" if "--size" in argv and "big" in argv else "small"
    use_ac = "--ac" in argv
    cfg = SIZES[size]

    if not torch.cuda.is_available():
        rlog.info(f"CPU: AC-equivalence max diff = {ac_equivalence_maxdiff()}")
        return

    distenv.init_process_group()
    device = torch.device("cuda", distenv.local_rank())
    start_memory_record()
    block = build_block(cfg["dim"], cfg["hidden"], cfg["n_heads"]).to(device)
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["dim"], device=device, requires_grad=True)
    forward_maybe_checkpointed(block, x, use_ac=use_ac).pow(2).mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    dump_memory_snapshot(f"artifacts/l2_mem_rank{distenv.rank()}.pickle")
    rlog.info(f"size={size} ac={use_ac} peak_gb={peak:.2f}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
