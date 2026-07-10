import torch

from dtensor_workshop import rlog
from dtensor_workshop.faultdata import ResumableLoader


def simulate_crash_and_resume(shape=(4, 8), base_seed=7, total=6, crash_at=3) -> bool:
    reference = ResumableLoader(shape, base_seed=base_seed)
    all_batches = [reference.next() for _ in range(total)]

    crashing = ResumableLoader(shape, base_seed=base_seed)
    for _ in range(crash_at):
        crashing.next()
    saved = crashing.state_dict()               # checkpoint on a shared filesystem

    resumed = ResumableLoader(shape, base_seed=base_seed)
    resumed.load_state_dict(saved)              # restart from the checkpoint
    tail = [resumed.next() for _ in range(total - crash_at)]
    return all(torch.equal(t, ref) for t, ref in zip(tail, all_batches[crash_at:]))


def main():
    rlog.info(f"deterministic restart succeeded = {simulate_crash_and_resume()}")


if __name__ == "__main__":
    main()
