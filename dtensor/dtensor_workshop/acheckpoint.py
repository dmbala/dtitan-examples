from torch.utils.checkpoint import checkpoint


def forward_maybe_checkpointed(module, x, use_ac: bool):
    if use_ac:
        return checkpoint(module, x, use_reentrant=False)
    return module(x)
