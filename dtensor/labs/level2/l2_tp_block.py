import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp


def tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=1) -> float:
    x = torch.randn(2, 8, dim, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)(x).detach()
    tp_block = apply_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"]
    )
    out = tp_block(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    return (out - ref).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    diff = tp_parity_maxdiff(mesh)
    rlog.info(f"TP block parity max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
