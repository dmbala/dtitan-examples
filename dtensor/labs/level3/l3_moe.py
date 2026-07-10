import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.moe import MoEFeedForward, routing_imbalance
from dtensor_workshop.parallel3d import apply_fsdp


def moe_report(mesh, tokens=64, dim=32, hidden=64, n_experts=4, seed=5) -> dict:
    device = mesh.device_type
    torch.manual_seed(seed)
    moe = apply_fsdp(MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device), mesh)
    x = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(0)).to(device)
    _out, counts = moe(x)
    return {"counts": counts, "imbalance": routing_imbalance(counts), "routed": sum(counts)}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    rep = moe_report(mesh)
    rlog.info(f"expert counts={rep['counts']} imbalance={rep['imbalance']:.3f}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
