import torch

from dtensor_workshop import distenv, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.regional import regional_compile


def compile_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float:
    block = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    x = torch.randn(2, 8, dim)
    eager = block(x)
    compiled = regional_compile(block)(x)
    return (eager - compiled).abs().max().item()


def main():
    rlog.info(f"regional compile eager-vs-compiled max diff = {compile_maxdiff()} "
              f"(compare step time on GPU: torch.compile amortizes after warm-up)")


if __name__ == "__main__":
    main()
