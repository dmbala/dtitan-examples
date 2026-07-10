import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def dcp_save(model, optimizer, checkpoint_id) -> None:
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=checkpoint_id)


def dcp_load(model, optimizer, checkpoint_id) -> None:
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=checkpoint_id)
    set_state_dict(
        model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd
    )
