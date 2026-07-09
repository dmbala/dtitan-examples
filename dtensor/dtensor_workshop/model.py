import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 1024, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def _to_heads(self, t, batch, seq):
        n_local = t.shape[-1] // self.head_dim
        return t.view(batch, seq, n_local, self.head_dim).transpose(1, 2)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self._to_heads(self.q(x), batch, seq)
        k = self._to_heads(self.k(x), batch, seq)
        v = self._to_heads(self.v(x), batch, seq)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq, -1)
        x = x + self.o(attn)
        x = x + self.fc2(self.act(self.fc1(x)))
        return x


def build_block(dim: int = 256, hidden: int = 1024, n_heads: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    return TransformerBlock(dim=dim, hidden=hidden, n_heads=n_heads)
