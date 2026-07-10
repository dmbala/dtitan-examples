# Level 2 — Two-Dimensional Parallelism and Checkpointing

**2D Parallelism, Bottleneck Profiling, and Distributed Checkpointing**

Level 2 turns the foundations into a **realistic training loop** and makes it
resilient enough to save, load, profile, and debug. You combine data and tensor
parallelism, checkpoint and resume, diagnose an OOM, and use profiler evidence
instead of guesswork.

- **Audience:** understands the Level 1 basics; wants a real 2D-parallel loop.
- **Hardware:** 1 node × 4 GPUs (`kempner_h100`). Launcher: `slurm/launch_1node.sbatch`.
- **Mesh:** 2D `DeviceMesh` `dp=2 × tp=2`, dimensions named `dp` and `tp`.
- **Milestone order:** correct → restartable → fast.

## Core ideas

- **2D `DeviceMesh`** with named `dp` / `tp` dimensions and per-rank coordinates.
- **Tensor parallelism (Megatron-style)** — column-parallel then row-parallel
  linears (`ColwiseParallel` / `RowwiseParallel`). Attention uses **separate
  q/k/v projections** (a fused qkv is wrong under column sharding).
- **Data parallelism** — gradient averaging across the `dp` dimension.
- **Distributed Checkpointing (DCP)** — sharded save/load of model + optimizer,
  and resuming at the same step.
- **Memory** — allocation peaks, OOM diagnosis via CUDA memory snapshots, and
  **activation checkpointing** to trade compute for memory.
- **Bottleneck profiling** — locating expensive collectives and idle gaps.

The running example — the `TransformerBlock` in `dtensor_workshop/model.py` —
is introduced here. Library modules added: `model`, `tp`, `train`, `checkpoint`,
`acheckpoint`.

## The labs

| Lab | Teaches | Success criterion |
|-----|---------|-------------------|
| `l2_mesh` | build the 2D mesh; read each rank's `(dp, tp)` coordinate | rank *r* → `(dp=r//2, tp=r%2)` |
| `l2_tp_block` | tensor-parallelize the block (Megatron col/row) | TP output matches the single-device block (< 1e-4) |
| `l2_train` | full 2D training; loss parity vs a single-device baseline | 2D-parallel loss curve matches single-device (spiked exact) |
| `l2_dcp` | DCP save → restore into a fresh model → resume | reloaded model resumes identically (< 1e-6) |
| `l2_oom` | **failure-driven**: memory & activation checkpointing | AC forward matches plain forward; peak memory drops with `--ac` |
| `l2_profile` | profile one 2D step; find the dominant collective | per-rank trace produced with communication ops |
| `l2_capstone` | 2D loop + DCP + activation checkpointing + trace + diagnosis | loss parity < 1e-4 **and** DCP resume < 1e-6 |

### Notes on specific labs

- **`l2_train`** is the correctness heart of the level: a same-seed TP block,
  each `dp` replica taking its half of a shared global batch and averaging
  gradients, reproduces the single-device full-batch loss curve — because
  averaging equal-size per-replica mean-gradients equals the full-batch gradient.
- **`l2_dcp`** builds a *differently-seeded* fresh model before loading, so a
  near-zero post-restore diff proves the checkpoint actually transferred state
  (not shared initialization). It uses momentum SGD so optimizer state is real.
- **`l2_oom`** is the level's intentional failure. On CPU the test checks that
  activation checkpointing is numerically equivalent to a plain forward; on GPU
  the lab records a CUDA memory snapshot and reports peak memory. Flags:

  ```bash
  sbatch dtensor/slurm/launch_1node.sbatch -m labs.level2.l2_oom --size small
  sbatch dtensor/slurm/launch_1node.sbatch -m labs.level2.l2_oom --size small --ac   # lower peak
  ```

  Use `--size big` to force an OOM and inspect the snapshot; the snapshot is
  written to `artifacts/l2_mem_rank{N}.pickle` (load it in <https://pytorch.org/memory_viz>).
  `--size big` is calibrated for H100 (80 GB); on H200 (141 GB) raise the size to
  trigger OOM.
- **`l2_profile`** and the capstone write **per-rank** trace files
  (`artifacts/l2_trace_rank{N}.json`, `artifacts/l2_capstone_trace_rank{N}.json`)
  — never a single shared file — so ranks don't clobber each other's output.

## Capstone

`l2_capstone` trains the 2D-parallel block with activation checkpointing, saves
and reloads via DCP, exports a profiler trace, checks loss parity and resume
correctness, and prints a short diagnosis template.

```bash
sbatch dtensor/slurm/launch_1node.sbatch -m labs.level2.l2_capstone
```

## What you can do after Level 2

- Construct a 2D `dp × tp` mesh and reason about each rank's role.
- Apply Megatron tensor parallelism to attention and MLP correctly.
- Train data-parallel and verify loss parity against a single-device baseline.
- Save and restore model + optimizer with DCP and resume at the same step.
- Diagnose an OOM from a memory snapshot and cut peak memory with activation checkpointing.
- Read a profiler trace to name the dominant communication cost.

## Verification

- **CPU tests:** `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_mesh.py tests/test_l2_tp_block.py tests/test_l2_train.py tests/test_l2_dcp.py tests/test_l2_oom.py tests/test_l2_profile.py tests/test_l2_capstone.py -v'`
- **GPU smoke (validated on H100):** `l2_capstone` → `parity_maxdiff=1.19e-07`, `resume_maxdiff=0.0`; `l2_oom` → peak `0.26 GB` (no AC) vs `0.25 GB` (with AC).
