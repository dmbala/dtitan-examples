# Workshop Labs — Overview

This directory holds the hands-on labs, organized into three levels that build on
one another. Each level is a folder of small, focused scripts plus a capstone,
and its own `README.md` with per-lab instructions and expected results.

- **[`level1/`](level1/README.md)** — Foundations of distributed PyTorch (1D mesh, DTensor, diagnostics)
- **[`level2/`](level2/README.md)** — 2D parallelism, checkpointing, memory & bottleneck profiling
- **[`level3/`](level3/README.md)** — Scaling & production debugging (FSDP2, MoE, FP8, NCCL)

## The learning arc

The same compact **Transformer block** runs through all three levels, so each
level feels like an extension of the last rather than a new codebase:

```
Level 1   raw tensors + a linear      → what each rank owns, is it correct, collect a trace
Level 2   a 2D-parallel block         → tensor + data parallel, save/restore, find the bottleneck
Level 3   an MoE model under FSDP2    → 3D layout, FSDP2, FP8, and debugging failures at scale
```

Within each level the labs follow the order real distributed systems are built:

> **correct → restartable → fast → debuggable**

You make it produce the right answer first, make it survive a restart second,
make it fast third, and (Level 3) make failures diagnosable fourth. Each level
also includes one **intentional failure** you learn to diagnose — a shape
mismatch (L1), an OOM (L2), and a collective mismatch (L3).

## How to run a lab

Labs are Python modules, launched with `torchrun`. The working directory must be
`dtensor/` so `import labs.levelN.xxx` resolves — the Slurm launchers handle this
automatically.

**On CPU (gloo, no GPUs needed — ideal for reading behavior):**

```bash
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif
singularity exec "$IMAGE" bash -lc 'cd dtensor && torchrun --standalone --nproc_per_node=4 -m labs.level1.l1_hello'
```

**On GPUs via Slurm** (submit from the repo root; launchers are submit-dir robust):

```bash
sbatch dtensor/slurm/launch_1node.sbatch -m labs.level2.l2_capstone      # 1 node × 4 GPUs
sbatch dtensor/slurm/launch_2node.sbatch -m labs.level3.l3_hsdp_tp       # 2 nodes × 4 GPUs
```

Anything after the launcher path is passed straight to `torchrun -m …`, so you
can add lab flags (e.g. `-m labs.level2.l2_oom --size small --ac`). Job logs and
artifacts (profiler traces, memory snapshots, Flight-Recorder dumps) are written
under `dtensor/artifacts/`.

## Hardware per level

| Level | Launcher | Layout | Notes |
|-------|----------|--------|-------|
| 1 | `launch_1node.sbatch` | 1 node × 4 GPUs, 1D mesh | — |
| 2 | `launch_1node.sbatch` | 1 node × 4 GPUs, `dp2 × tp2` | — |
| 3 | `launch_2node.sbatch` | 2 nodes × 4 GPUs, `dp_replicate2 × dp_shard2 × tp2` | on `kempner_h100` the per-user cap is 8 GPUs, so 2-node jobs run one at a time |

## Correctness vs. GPU-only behavior

Every lab exposes a small **pure function** (e.g. `shard_report`, `tp_parity_maxdiff`,
`fsdp_resume_maxdiff`) that the test suite exercises GPU-free on CPU/gloo, plus a
`main()` for `torchrun`. So the parallelism *correctness* is tested without GPUs;
the *GPU-only* aspects (FP8 numerics, Flight-Recorder traces, real multi-node
timing) are validated by the documented smoke runs in each level's README.
