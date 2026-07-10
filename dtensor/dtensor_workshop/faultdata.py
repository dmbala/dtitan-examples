import torch


class ResumableLoader:
    def __init__(self, shape, base_seed: int = 0):
        self.shape = tuple(shape)
        self.base_seed = base_seed
        self.step = 0

    def next(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.base_seed * 1_000_003 + self.step)
        batch = torch.randn(*self.shape, generator=gen)
        self.step += 1
        return batch

    def state_dict(self) -> dict:
        return {"step": self.step}

    def load_state_dict(self, state) -> None:
        self.step = state["step"]
