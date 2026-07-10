import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.fp8 import maybe_convert_fp8
from dtensor_workshop.moe import MoEFeedForward, routing_imbalance
from dtensor_workshop.parallel3d import apply_fsdp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def run_capstone(mesh, checkpoint_id, steps=3, dim=32, hidden=64, n_experts=4,
                 seed=8, reshard_after_forward=False):
    device = mesh.device_type
    x = torch.randn(64, dim, generator=torch.Generator().manual_seed(0)).to(device)
    batches = [x for _ in range(steps)]

    torch.manual_seed(seed)
    model = maybe_convert_fp8(MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device))
    model = apply_fsdp(model, mesh, reshard_after_forward=reshard_after_forward)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []
    for batch in batches:                       # MoE returns (out, counts); custom loop
        opt.zero_grad()
        out, counts = model(batch)
        out.pow(2).mean().backward()
        opt.step()
        losses.append(routing_imbalance(counts))

    # Save only model state (avoid optimizer state issues with non-gradient parameters)
    model_sd, _ = get_state_dict(model, opt)
    dcp.save({"model": model_sd}, checkpoint_id=checkpoint_id)

    torch.manual_seed(seed + 1000)
    restored = apply_fsdp(
        MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device), mesh,
        reshard_after_forward=reshard_after_forward,
    )
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1)

    # Load only model state
    restored_model_sd, restored_optim_sd = get_state_dict(restored, restored_opt)
    dcp.load({"model": restored_model_sd}, checkpoint_id=checkpoint_id)
    set_state_dict(restored, restored_opt, model_state_dict=restored_model_sd, optim_state_dict=restored_optim_sd)

    resume = (_full(model(x)[0]) - _full(restored(x)[0])).abs().max().item()
    return {"resume_maxdiff": resume, "imbalance": losses[-1], "steps": steps}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp_replicate", "dp_shard"))
    res = run_capstone(mesh, "checkpoints/l3_capstone")
    rlog.info(f"resume_maxdiff={res['resume_maxdiff']:.2e} imbalance={res['imbalance']:.3f}")
    if distenv.rank() == 0:
        rlog.info(
            "POSTMORTEM (fill in): which rank/collective failed? Flight Recorder finding? "
            "mesh layout (dp_replicate/dp_shard/tp)? overlap or FP8 speedup? routing imbalance?"
        )
    distenv.shutdown()


if __name__ == "__main__":
    main()
