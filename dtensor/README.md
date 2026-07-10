# DTensor Distributed-Training Workshop

A hands-on, three-level course that teaches **modern PyTorch distributed training**
from first principles using `DeviceMesh`, `DTensor`, FSDP2, Distributed
Checkpointing (DCP), profiling, and production-style debugging. Each level moves
you from a working script to an **observable, restartable, and diagnosable**
distributed training workflow.

The code is **standalone DTensor** — it uses PyTorch's distributed primitives
directly rather than a higher-level framework, so you see exactly what each
collective, placement, and shard does. (A separate torchtitan-focused workshop
builds on these fundamentals.)

---

## The three levels

| Level | Theme | Parallelism | Hardware |
|------|-------|-------------|----------|
| **1** | Foundations: `DeviceMesh`, `DTensor`, diagnostics | 1D (`dp`), 4 GPUs | 1 node |
| **2** | 2D parallelism, checkpointing, memory & bottleneck profiling | `dp × tp`, 4 GPUs | 1 node |
| **3** | Scaling & production debugging: FSDP2, MoE, FP8, NCCL | real 3D `dp_replicate × dp_shard × tp`, 8 GPUs | 2 nodes |

One compact **Transformer block** is the running example throughout: raw tensor
ops in Level 1 → a 2D-parallel block in Level 2 → an MoE-augmented model under
FSDP2 in Level 3. Full curriculum design is in
[`docs/workshop_design.md`](docs/workshop_design.md).

---

## What's in this directory

```
dtensor/
├── dtensor_workshop/     Reusable library the labs import (see table below)
├── labs/                 The workshop itself — level1/, level2/, level3/
│   └── README.md         Labs overview + how to run
├── tests/                62 GPU-free correctness tests (pytest, CPU/gloo)
├── slurm/                Ready-to-run Slurm launchers (1-node, 2-node)
├── preflight.py          Environment self-check to run before the workshop
├── docs/                 Design spec + per-level implementation plans
├── data/  checkpoints/  artifacts/   Runtime outputs (git-ignored)
├── pyproject.toml        Package + pytest config
└── conftest.py           Puts the package on sys.path for pytest
```

### `dtensor_workshop/` — the shared library

| Module | Purpose | Introduced in |
|--------|---------|---------------|
| `distenv.py` | rank / world / local-rank, device type, process-group init & shutdown | L1 |
| `rlog.py` | rank-aware logging (`[rank i/N] …`) | L1 |
| `synth.py` | deterministic, rank-independent synthetic tensors | L1 |
| `mesh.py` | `build_mesh()` around `init_device_mesh` | L1 |
| `testing.py` | `run_distributed()` — the `mp.spawn` gloo harness the tests use | L1 |
| `model.py` | `TransformerBlock` (attention + MLP) + `build_block()` | L2 |
| `tp.py` | Megatron tensor-parallel plan + `apply_tp()` | L2 |
| `train.py` | `run_training()` + `average_gradients()` (data-parallel) | L2 |
| `checkpoint.py` | `dcp_save()` / `dcp_load()` (Distributed Checkpoint) | L2 |
| `acheckpoint.py` | `forward_maybe_checkpointed()` (activation checkpointing) | L2 |
| `parallel3d.py` | `apply_fsdp()` + `apply_hsdp_tp()` (FSDP2, HSDP+TP) | L3 |
| `moe.py` | `MoEFeedForward` + `routing_imbalance()` | L3 |
| `regional.py` | `regional_compile()` (regional `torch.compile`) | L3 |
| `faultdata.py` | `ResumableLoader` (deterministic restart) | L3 |
| `fp8.py` | `hopper_available()` + `maybe_convert_fp8()` (torchao FP8) | L3 |
| `frdebug.py` | `shapes_agree()` + `dump_flight_recorder()` (NCCL debugging) | L3 |

---

## Environment

Everything runs inside the workshop container (NGC PyTorch 25.11 → torch 2.10,
NCCL, flash-attn, torchao):

```bash
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif
```

The container already exports the NCCL / Flight-Recorder environment the labs
rely on — you don't set those yourself.

### Preflight

Before the workshop, verify your environment:

```bash
singularity exec --nv "$IMAGE" bash -lc 'cd dtensor && python preflight.py'
```

It checks GPU visibility, a 2-rank NCCL all-reduce, torch/DTensor import, DCP
write access, and that the bind-mount directories resolve.

---

## Quick start

**Run the correctness tests (GPU-free, on any node):**

```bash
singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests -q'
# 62 passed
```

**Run a lab interactively on CPU (4 gloo ranks — great for reading output):**

```bash
singularity exec "$IMAGE" bash -lc 'cd dtensor && torchrun --standalone --nproc_per_node=4 -m labs.level1.l1_hello'
```

**Run a lab on GPUs via Slurm** (from the repo root; the launchers are
submit-directory robust):

```bash
# 1-node / 4-GPU labs (Levels 1 & 2)
sbatch dtensor/slurm/launch_1node.sbatch -m labs.level2.l2_capstone

# 2-node / 8-GPU labs (Level 3)
sbatch dtensor/slurm/launch_2node.sbatch -m labs.level3.l3_hsdp_tp
```

The launchers are preconfigured for the Kempner cluster (`--account=kempner_dev`,
`--partition=kempner_h100`, `--mem`, bind mounts, container invocation). Logs
land in `artifacts/`.

---

## How the labs are tested

The suite is deliberately **GPU-free**: correctness is verified on CPU with the
`gloo` backend via a `torch.multiprocessing.spawn` harness
(`dtensor_workshop.testing.run_distributed`). This makes every parallelism
claim — sharding, redistribution, tensor/data parallel, FSDP2, DCP round-trips,
MoE routing — reproducible on any node with no GPUs, and fast enough for a tight
edit-test loop.

Genuinely **GPU-only** behavior — FP8 execution, real NCCL Flight-Recorder
traces, communication/compute overlap timing, and multi-node runs — is validated
by the documented `sbatch` smoke runs in each level's README, not by the CPU
suite.

**Status:** all 62 CPU tests pass, and every lab has been smoke-run on real
**H100** hardware (1- and 2-node) on `kempner_h100`.

---

## Where to go next

- **[`labs/README.md`](labs/README.md)** — labs overview and the learning arc.
- **[`labs/level1/README.md`](labs/level1/README.md)** — Foundations.
- **[`labs/level2/README.md`](labs/level2/README.md)** — 2D parallelism & checkpointing.
- **[`labs/level3/README.md`](labs/level3/README.md)** — Scaling & production debugging.
- **[`docs/workshop_design.md`](docs/workshop_design.md)** — full curriculum design.
