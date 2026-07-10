import torch
import torch.nn as nn


class _Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 1024, n_experts: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([_Expert(dim, hidden) for _ in range(n_experts)])

    def forward(self, x):
        assignment = self.router(x).argmax(dim=-1)      # top-1
        out = torch.zeros_like(x)
        counts = []
        for e in range(self.n_experts):
            mask = assignment == e
            counts.append(int(mask.sum()))
            if mask.any():
                out[mask] = self.experts[e](x[mask])
        return out, counts


def routing_imbalance(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    mean = total / len(counts)
    return max(counts) / mean
