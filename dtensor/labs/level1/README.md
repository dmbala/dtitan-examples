# Level 1 — Foundations of Distributed PyTorch

**`DeviceMesh`, `DTensor`, and Diagnostics**

The goal of Level 1 is a **correct mental model**: how the same Python program
runs across many ranks (SPMD), and how DTensor *placements* describe where a
tensor's data actually lives. You leave able to explain what each rank owns,
verify a distributed operation is correct, and collect a trace when something
looks off.

- **Audience:** comfortable with single-GPU PyTorch; new to multi-process / SPMD.
- **Hardware:** 1 node × 4 GPUs (`kempner_h100`). Launcher: `slurm/launch_1node.sbatch`.
- **Mesh:** a 1D `DeviceMesh` of size 4, dimension named `dp`.
- **Milestone order:** correct → observable.

## Core ideas

- **SPMD execution** — ranks, world size, local rank, process groups, launchers.
- **`DeviceMesh`** — mapping logical mesh dimensions onto devices.
- **DTensor placements** — `Shard(dim)`, `Replicate()`, and `Partial()`.
- **Correctness checks** — local shards, global shape, redistribution, rank-aware logging.
- **Introductory profiling** — capturing a `torch.profiler` chrome trace.
- **Debugging** — reading a multi-rank stack trace from a shape mismatch.

Library modules introduced here: `distenv`, `rlog`, `synth`, `mesh`, and the test
harness `testing`.

## The labs

Each lab exposes a small pure function (tested on CPU/gloo) and a `main()` for
`torchrun`. Run interactively on CPU with, e.g.:

```bash
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif
singularity exec "$IMAGE" bash -lc 'cd dtensor && torchrun --standalone --nproc_per_node=4 -m labs.level1.l1_hello'
```

| Lab | Teaches | Success criterion |
|-----|---------|-------------------|
| `l1_hello` | rank-aware SPMD state (world size, local rank) | 4 ranks report `world_size=4` and the right `local_rank` |
| `l1_shard` | shard a large tensor over a 1D mesh with `Shard(0)` | local shard shape = global / 4; global shape reconstructs |
| `l1_redistribute` | move between `Shard` and `Replicate` (all-gather) | replicated copy identical on every rank (max abs diff = 0) |
| `l1_shape_bug` | **failure-driven**: a cross-rank shape mismatch and its fix | the buggy matmul raises; the fixed one yields the right global shape |
| `l1_profile` | capture a profiler chrome trace of compute + a collective | trace file created; a communication op appears |
| `l1_capstone` | end-to-end: mesh → shard → math → validate → trace | distributed global sum matches the single-device reference within tolerance |

### Notes on specific labs

- **`l1_redistribute`** shows the `Shard → Replicate` transition; because every
  rank ends up with the full tensor, the difference against the reference is
  *exactly* zero.
- **`l1_shape_bug`** is the level's intentional failure. `buggy_matmul` contracts
  mismatched dimensions and raises a `RuntimeError`; `fixed_matmul` aligns them.
  Reading the multi-rank traceback is the point.
- **`l1_capstone`** shards a tensor, computes `y = x*2 + 1`, reduces `y.sum()`
  (which lands in a **`Partial`** placement) to `Replicate` via all-reduce, and
  checks the result against a single-device computation — then exports a trace.

## Capstone

`l1_capstone` is the deliverable: build a 1D mesh, shard a synthetic tensor, run
several operations, **validate the global result against a single-device
reference**, and export a profiler trace showing compute and communication.

```bash
sbatch dtensor/slurm/launch_1node.sbatch -m labs.level1.l1_capstone
```

## What you can do after Level 1

- Explain what data each rank holds for a `Shard`, `Replicate`, or `Partial` DTensor.
- Verify a distributed tensor operation is correct against a single-device baseline.
- Capture and open a profiler trace to see compute vs. communication.
- Read and fix a multi-rank shape-mismatch error.

## Verification

- **CPU tests:** `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_shard.py tests/test_l1_redistribute.py tests/test_l1_shape_bug.py tests/test_l1_profile.py tests/test_l1_capstone.py -v'`
- **GPU smoke (validated):** `l1_capstone` on H100 → `parity_ok=True` on all 4 ranks (distributed sum = single-device reference).
