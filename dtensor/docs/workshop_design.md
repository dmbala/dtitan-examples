# DTensor Workshop — Design Spec

> Infrastructure-grounded revision of `initial_outline.md`.
> Target environment: Kempner cluster on FASRC, H100/H200 nodes (4 GPUs/node),
> up to 2 nodes, Slurm, `dtitan.sif` Apptainer/Singularity container.
> Date: 2026-07-09.

## Purpose

A three-level workshop teaching modern PyTorch distributed training through
`DeviceMesh`, `DTensor`, Distributed Checkpointing (DCP), profiling, and
production-oriented debugging. Each level moves participants from a working
script to an observable, restartable, and diagnosable distributed training
workflow.

The workshop uses **standalone DTensor** primitives (not torchtitan). torchtitan
is installed in the container and is the subject of a separate future workshop;
here the code is self-contained so participants see the primitives directly.

## Audience Assumptions

- Participants can write and train a single-GPU PyTorch model.
- Participants are familiar with tensors, autograd, optimizers, and training loops.
- No prior experience with multi-node training, NCCL, or distributed checkpoint
  formats is required.
- Level 1 and Level 2 run on a single 4-GPU node; Level 3 uses 2 nodes (8 GPUs).

## The Running Example

One model family is used across all three levels so participants focus on
distributed-systems concepts rather than model details. It grows with each level:

| Level | Model surface | What is new |
| --- | --- | --- |
| 1 | Raw tensors + a single `nn.Linear` | Distributed tensor layout and correctness |
| 2 | A compact Transformer block (attention + MLP) | 2D parallelism, checkpointing, memory |
| 3 | The same block, MoE-augmented (expert MLP) | FSDP2, 3D mesh, FP8, overlap, NCCL debugging |

The block is intentionally small (single-digit-million parameters, short sequence
length) so exercises are fast and deterministic on synthetic data.

## Infrastructure & Runbook

This section is shared by all three levels. Levels reference it rather than
repeating launch details.

### Container

```
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif
```

Base: NGC `pytorch:25.11-py3` → torch 2.10.0a0 (2.11 preview), recent NCCL,
flash-attn, torchao (FP8), torchtitan 0.2.0 (present but unused here).

The container **already exports** the NCCL and Flight Recorder variables the
workshop relies on — do not re-set them:

```
TORCH_NCCL_ASYNC_ERROR_HANDLING=1
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_TRACE_BUFFER_SIZE=20971520   # 20 MB Flight Recorder ring buffer
HF_HOME=/data/hf_cache
HF_HUB_OFFLINE=1
```

Standard invocation from the working directory:

```bash
singularity exec --nv \
  --bind $(pwd)/data:/data \
  --bind $(pwd)/checkpoints:/checkpoints \
  --bind $(pwd)/artifacts:/artifacts \
  "$IMAGE" \
  torchrun --nproc_per_node=4 script.py --flags
```

### Directory Layout

Created once before the workshop; bound into the container as shown above.

```
data/          synthetic datasets (and optional HF-offline cache under hf_cache/)
checkpoints/   DCP sharded checkpoints
artifacts/     profiler traces, memory snapshots, Flight Recorder dumps
```

### Slurm Launchers

Two templates. `<account>` is a placeholder to confirm before the workshop
(likely `kempner_dev`); add a `#SBATCH --qos=<qos>` line if your account requires
one. Partitions are confirmed: `kempner_h100`, `kempner_h200` for GPU work;
`kempner_eng` for build/interactive use.

**1-node / 4-GPU (Level 1 and Level 2):**

```bash
#!/bin/bash
#SBATCH --job-name=dtensor-l12
#SBATCH --partition=kempner_h100        # or kempner_h200
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

singularity exec --nv \
  --bind $(pwd)/data:/data --bind $(pwd)/checkpoints:/checkpoints --bind $(pwd)/artifacts:/artifacts \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 "$@"
```

**2-node / 8-GPU (Level 3):** uses `torchrun` c10d rendezvous across the two
nodes. `srun` launches one `torchrun` per node; each spawns 4 local ranks.

```bash
#!/bin/bash
#SBATCH --job-name=dtensor-l3
#SBATCH --partition=kempner_h100
#SBATCH --account=<account>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500

srun singularity exec --nv \
  --bind $(pwd)/data:/data --bind $(pwd)/checkpoints:/checkpoints --bind $(pwd)/artifacts:/artifacts \
  "$IMAGE" \
  torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    "$@"
```

### Preflight Script (spec — built during implementation)

A single script participants run before the workshop to catch environment
problems early. It verifies, in order, and prints a pass/fail line for each:

1. GPU visibility (`nvidia-smi`; `torch.cuda.device_count() == 4`).
2. NCCL init and a 2-rank `all_reduce` returns the expected sum.
3. torch version and `torch.distributed.tensor` (DTensor) import.
4. DCP **write** access: save and re-load a tiny sharded state dict under `checkpoints/`.
5. Profiler output path under `artifacts/` is writable.
6. Container bind mounts (`/data`, `/checkpoints`, `/artifacts`) resolve inside the container.

### Reference Artifacts

Instructors provide known-good outputs so participants can compare when local
hardware behaves differently: a Level 1 profiler trace, a Level 2 memory snapshot
and DCP checkpoint, and a Level 3 Flight Recorder dump. Stored under
`artifacts/reference/`.

### Environment Notes

- **H100 (80 GB) vs H200 (141 GB HBM3e):** the Level 2 OOM lab exposes a size knob
  (batch × sequence × hidden) so OOM triggers on either card. FP8 works on both.
- For Level 3 debugging labs, participants additionally set `NCCL_DEBUG=INFO` and,
  for async tensor parallelism, `CUDA_DEVICE_MAX_CONNECTIONS=1` at launch.

---

## Level 1 — Foundations of Distributed PyTorch

**Title:** Foundations of Distributed PyTorch: DeviceMesh, DTensor, and Diagnostics

**Environment / launch:** 1 node, 4 GPUs, `kempner_h100` (or `kempner_h200`).
1-node launcher, `torchrun --standalone --nproc_per_node=4`.

**Target audience:** ML engineers and researchers comfortable with single-GPU
PyTorch and new to distributed execution and the SPMD model.

### Goals

Leave with a correct mental model for how the same Python program runs across
ranks and how DTensor placements describe distributed tensor layout.

### Mesh Sizing

1D `DeviceMesh` of size 4 (one dimension, all four local GPUs).

### Core Topics

- SPMD execution: ranks, world size, local rank, process groups, launchers.
- `DeviceMesh`: mapping logical mesh dimensions onto devices.
- DTensor placements: `Shard(dim)`, `Replicate()`, `Partial()`.
- Correctness checks: local shards, global shape, redistribution, rank-aware logging.
- Introductory profiling with `torch.profiler`.
- Beginner troubleshooting: mismatched shapes, missing collectives, launcher errors,
  multi-rank stack traces.

### Labs

Milestone order for the level: **correct → observable**.

| # | Lab | Command | Expected artifact | Success criterion |
| --- | --- | --- | --- | --- |
| 1 | Launch across ranks, print rank-aware state | `torchrun --standalone --nproc_per_node=4 l1_hello.py` | stdout with per-rank lines | 4 distinct ranks report world_size=4, correct local_rank |
| 2 | Build 1D mesh, shard a large tensor | `... l1_shard.py` | stdout of local shard shapes | local shard shape = global/4; `full_tensor()` reconstructs global shape |
| 3 | Redistribute Shard ↔ Replicate | `... l1_redistribute.py` | stdout before/after placements | replicated copy identical on all ranks (max abs diff = 0) |
| 4 | **Failure-driven:** trigger + debug a cross-rank shape mismatch | `... l1_shape_bug.py` | captured error + fixed run | participant explains the mismatch and the fix resumes cleanly |
| 5 | Capture a profiler trace | `... l1_profile.py` | `artifacts/l1_trace.json` | trace shows both compute and a communication op (e.g. all-gather) |

### Capstone

A script that creates a 1D `DeviceMesh`, shards a large synthetic tensor, runs
several math operations, validates the global result against a single-device
reference, and exports a profiler trace showing compute and communication.

### Expected Outcome

Participants can explain what each rank owns, verify a distributed tensor
operation is correct, and collect a basic trace when something looks suspicious.

### Instructor Notes

- Estimated time: ~half day.
- Common errors: forgetting `torchrun` (script runs as world_size=1); binding the
  wrong device (all ranks land on GPU 0 → set device from `LOCAL_RANK`).
- Recovery: emphasize `LOCAL_RANK`-based device selection early; it prevents most
  Level 1 confusion.

---

## Level 2 — Two-Dimensional Parallelism and Checkpointing

**Title:** 2D Parallelism, Bottleneck Profiling, and Distributed Checkpointing

**Environment / launch:** 1 node, 4 GPUs. Same 1-node launcher as Level 1.

**Target audience:** practitioners who understand distributed basics and want to
combine data + tensor parallelism, memory profiling, and restartable checkpointing.

### Goals

Structure a realistic 2D parallel training loop and make it resilient enough to
save, load, profile, and debug.

### Mesh Sizing

2D `DeviceMesh` `dp=2 × tp=2` = 4 GPUs, dimensions named `dp` and `tp`.

### Core Topics

- 2D `DeviceMesh` with named dimensions (`dp`, `tp`).
- Tensor-parallel linear layers: Megatron-style column and row sharding.
- Data-parallel training semantics across the `dp` dimension.
- Distributed Checkpointing with sharded state dicts; async saves; restart validation.
- Memory profiling: allocation timelines, peak memory, fragmentation, OOM diagnosis.
- Activation checkpointing tradeoffs and selective activation checkpointing.
- Communication bottlenecks: all-reduce, all-gather, reduce-scatter, idle GPU time.

### Labs

Milestone order: **correct → restartable → fast**.

| # | Lab | Command | Expected artifact | Success criterion |
| --- | --- | --- | --- | --- |
| 1 | Build 2D mesh, assign `dp`/`tp` roles | `... l2_mesh.py` | stdout of mesh + per-rank coords | each rank prints correct `(dp, tp)` coordinate |
| 2 | Implement TP MLP / Transformer block (col + row) | `... l2_tp_block.py` | stdout of output shape | TP output matches single-device block within tolerance |
| 3 | Train on synthetic data; validate loss parity | `... l2_train.py` | `artifacts/l2_loss.csv` | loss matches single-device baseline within tolerance |
| 4 | DCP save/load of model+optimizer+scheduler | `... l2_dcp.py` | `checkpoints/l2/` | reload resumes at the same step; async save completes without blocking |
| 5 | **Failure-driven:** trigger OOM, snapshot, fix with activation checkpointing | `... l2_oom.py --size big` then `--ac` | `artifacts/l2_mem_snapshot.pickle` | OOM reproduced; snapshot identifies peak; AC lowers peak below limit |
| 6 | Profile to locate expensive collectives / idle gaps | `... l2_profile.py` | `artifacts/l2_trace.json` | participant names the dominant collective and any idle gap |

### Capstone

A 2D-parallel training loop for the compact Transformer block, including a working
DCP save/load path, a profiler trace, a memory-optimization change, and a short
written diagnosis of the dominant communication or memory bottleneck.

### Expected Outcome

Participants can build a small realistic distributed loop, recover from
checkpoints, reason about memory pressure, and use profiler evidence instead of
guesswork.

### Instructor Notes

- Estimated time: ~full day.
- Common errors: TP sharding dim mismatch (column vs row confusion) → wrong output
  shape; forgetting to reshard optimizer state before DCP save.
- OOM lab: `--size big` is calibrated for H100 (80 GB); on H200 (141 GB) participants
  increase the size knob further to force OOM. Provide the exact knob values per card.
- Recovery: keep a single-device reference handy for loss-parity debugging.

---

## Level 3 — Advanced Scaling and Production Debugging

**Title:** Supercomputer Scaling: FSDP2, MoE, Async Optimizations, and NCCL Debugging

**Environment / launch:** 2 nodes, 8 GPUs, `kempner_h100`. 2-node launcher with
c10d rendezvous. `NCCL_DEBUG=INFO`; `CUDA_DEVICE_MAX_CONNECTIONS=1` for async TP labs.

**Target audience:** infrastructure, performance, and research engineers scaling
training across multi-node GPU clusters.

### Goals

Reason about complex parallel layouts, overlap communication with computation, and
debug failures that appear only at larger scale.

### Mesh Sizing

**Primary mesh — real 3D, mapped to the physical topology (8 GPUs):**

```
dp_replicate = 2   # across the 2 nodes  (slower inter-node link)
dp_shard     = 2   # within a node       (FSDP2 shards here)
tp           = 2   # within a node       (fast NVLink)
                   # 2 × 2 × 2 = 8 GPUs → HSDP + TP
```

This maps the slow dimension (replication) to the inter-node link and the
communication-heavy dimensions (shard + TP) to intra-node NVLink — a genuinely
meaningful layout at 8 GPUs, not a degenerate one.

**MoE variant (focused exercise):** swap in an `ep` (expert-parallel) dimension at
small size (e.g. `dp=2 × ep=2 × tp=2`) so participants see expert routing without
needing more hardware.

**4D — conceptual only:** adding a `pp` (pipeline) dimension is taught as
code-that-would-scale. The construction code is real and correct; it is run at
degenerate size (some dims = 1) purely to show it constructs and is **honestly
labeled illustrative**, since 8 GPUs cannot exercise a meaningful 4D layout.

### Core Topics

- 3D meshes combining data, tensor, and (variant) expert or fully-sharded dims;
  4D discussed conceptually.
- FSDP2 (`fully_shard`) for parameter, gradient, and optimizer-state sharding.
- Mixture-of-Experts routing, expert parallelism, token dispatch overhead.
- Communication/compute overlap: backward prefetch, async tensor parallelism,
  reduce-scatter/all-gather scheduling.
- **FP8 communication and compute (torchao) on Hopper — core content** (both H100
  and H200 are Hopper).
- Regional `torch.compile` on stable subgraphs.
- Advanced NCCL debugging: hangs, timeouts, mismatched collectives, environment
  variables, Flight Recorder traces (buffer pre-enabled in the container).
- Fault-aware data loading and deterministic restart on the shared filesystem.

### Labs

Milestone order: **correct → restartable → fast → debuggable**.

| # | Lab | Command | Expected artifact | Success criterion |
| --- | --- | --- | --- | --- |
| 1 | Build the 3D mesh; document each dim's role | `sbatch l3.sbatch l3_mesh.py` | stdout of mesh + physical mapping | each dim's size and node/NVLink placement is correct and explained |
| 2 | Wrap regions with FSDP2; validate DCP compatibility | `sbatch l3.sbatch l3_fsdp2_dcp.py` | `checkpoints/l3/` | FSDP2 state dict saves and reloads; resumes at same step |
| 3 | Add a small MoE layer; inspect routing imbalance | `sbatch l3.sbatch l3_moe.py` | `artifacts/l3_routing.csv` | per-expert token counts reported; imbalance quantified |
| 4 | Compare baseline vs communication-overlap | `sbatch l3.sbatch l3_overlap.py` | `artifacts/l3_trace_{base,overlap}.json` | overlap run shows reduced exposed communication time |
| 5 | Enable regional `torch.compile`; measure impact | `sbatch l3.sbatch l3_compile.py` | `artifacts/l3_compile.csv` | step time change reported; region compiles without graph breaks in the target block |
| 6 | Enable FP8 (torchao) on the block; verify | `sbatch l3.sbatch l3_fp8.py` | `artifacts/l3_fp8.csv` | FP8 run trains stably; throughput vs bf16 reported |
| 7 | **Failure-driven:** inject collective mismatch / delay, isolate failing rank | `sbatch l3.sbatch l3_hang.py --inject mismatch` | `artifacts/l3_flight_recorder/` dump | Flight Recorder dump + `NCCL_DEBUG` pinpoint the failing rank and collective |

### Capstone

An optimized training script for a small MoE model on the 2-node cluster,
demonstrating: a documented mesh layout, FSDP2 sharding, DCP recovery, at least one
overlap **or** precision (FP8) optimization, and a postmortem-style debugging note
for an injected NCCL issue.

### Expected Outcome

Participants can connect model architecture, parallel layout, checkpointing,
profiling, and cluster diagnostics into a production-style distributed workflow.

### Instructor Notes

- Estimated time: ~full day.
- Common errors: rendezvous failures (wrong `MASTER_ADDR`, port in use); mismatched
  collectives across ranks producing hangs rather than errors; FP8 numerics
  diverging without proper scaling.
- The injected-hang lab depends on the container's pre-set Flight Recorder buffer —
  confirm `TORCH_NCCL_TRACE_BUFFER_SIZE` is non-zero at job start.
- Recovery: keep the 2-node launcher's rendezvous settings fixed and documented; most
  Level 3 setup failures are rendezvous, not model, problems.

---

## Cross-Cutting Workshop Design

### Format

- Each level runs as a half-day (L1) or full-day (L2, L3) module.
- Every level starts from a minimal runnable baseline before optimization or failure modes.
- Lectures short; each concept paired with a concrete lab.
- The single running example (above) is used throughout.
- Each capstone includes a correctness check, a profiling artifact, and a short written diagnosis.

### Milestone Ordering

Within each level, participants make the code **correct first, restartable second,
fast third** (and **debuggable** in Level 3). This mirrors how distributed systems
are built in practice and is reflected in the lab ordering.

### Failure-Driven Exercises

Each level includes at least one intentional failure, introduced once the baseline
works: **shape mismatch (L1)**, **OOM (L2)**, **collective mismatch/hang (L3)**.

### Environment Requirements

- PyTorch and NCCL pinned by the `dtitan.sif` container (documented in the runbook).
- Reproducible environment via the container image (`dtitan.def` recipe is versioned).
- Single-node and 2-node Slurm launchers provided.
- Small synthetic datasets for deterministic exercises; optional HF-offline dataset
  only after distributed mechanics are stable.
- Pre-created `checkpoints/`, `artifacts/`, and `data/` directories, bound into the container.

## Assessment Rubric

Small, gradable items per level:

| Dimension | What is checked |
| --- | --- |
| Correctness validation | Loss/output parity against a single-device reference within tolerance |
| Checkpoint recovery | DCP reload resumes at the same step |
| Profiler interpretation | Participant names the dominant bottleneck from a trace |
| Debugging explanation | Written diagnosis of the level's injected failure and its fix |

## Notes on Revisions from `initial_outline.md`

- **FP8/Hopper promoted to core** (was "optional advanced"): both target cards are Hopper.
- **4D mesh reframed as conceptual**: an 8-GPU ceiling cannot exercise a meaningful 4D layout;
  the real 3D HSDP+TP mesh replaces it as the runnable primary.
- **Old recommendations #2, #5, #8** folded into per-level lab tables and milestone ordering.
- **Recommendation #7** revised: only 2nd-node multi-node NCCL work remains modular (in case a
  second node is unavailable); FP8 is no longer optional.
- **Environment specifics** now concrete: container path, bind mounts, Slurm partitions
  (`kempner_h100`, `kempner_h200`, `kempner_eng`), pre-set NCCL/Flight Recorder vars.
