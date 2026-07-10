import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_hsdp_tp


def hsdp_tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=2) -> float:
    device = mesh.device_type
    x = torch.randn(4, 8, dim, generator=torch.Generator().manual_seed(0)).to(device)
    ref = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device)(x).detach()
    model = apply_hsdp_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device), mesh
    )
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    return (out - ref).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"))
    diff = hsdp_tp_parity_maxdiff(mesh)
    rlog.info(f"HSDP+TP parity max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
