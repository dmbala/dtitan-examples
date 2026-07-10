# Level 3 — Advanced Scaling and Production Debugging

**FSDP2, MoE, Async Optimizations, and NCCL Debugging**

Level 3 scales across **two nodes** and adds the tools you need when training
gets big and failures only appear at scale: fully-sharded data parallel (FSDP2)
composed with tensor parallelism on a real 3D mesh, a Mixture-of-Experts layer,
regional compilation, FP8, fault-aware restart, and NCCL/Flight-Recorder
debugging.

- **Audience:** infrastructure / performance / research engineers scaling training.
- **Hardware:** 2 nodes × 4 GPUs = 8 (`kempner_h100`). Launcher: `slurm/launch_2node.sbatch`.
- **Primary mesh:** a real 3D mesh mapped to the physical topology —
  `dp_replicate=2` across the two nodes (slower inter-node link) ×
  `dp_shard=2 × tp=2` within a node (fast NVLink) = 8.
- **Milestone order:** correct → restartable → fast → debuggable.

> **Cluster note:** on `kempner_h100` the per-user GPU cap is **8 GPUs**, so each
> 2-node job uses your entire budget and they run **one at a time**. Submit them
> sequentially (or let them queue).

## Core ideas

- **3D parallelism** — `DeviceMesh` combining data-replicate, data-shard, and
  tensor dims; 4D (adding pipeline) is taught as an honestly-labeled *conceptual*
  walkthrough since 8 GPUs can't exercise a real 4D layout.
- **FSDP2 (`fully_shard`)** — parameter/gradient/optimizer sharding, composed
  **TP-first-then-FSDP** for HSDP+TP; DCP-compatible.
- **Mixture-of-Experts** — top-1 routing, per-expert dispatch, and **routing
  imbalance** measurement.
- **Communication/compute overlap** — FSDP2 `reshard_after_forward` as the knob.
- **FP8 (torchao) on Hopper** — with an important limitation (below).
- **Regional `torch.compile`** on a stable subgraph.
- **NCCL debugging** — a hang-free collective-shape preflight and Flight-Recorder
  trace dumps to isolate a failing rank/collective.
- **Fault-aware data loading** — deterministic restart on a shared filesystem.

Library modules added: `parallel3d`, `moe`, `regional`, `faultdata`, `fp8`, `frdebug`.

## The labs

Most Level 3 labs need the full 8 GPUs (world size 8). Run them one at a time:

```bash
sbatch dtensor/slurm/launch_2node.sbatch -m labs.level3.l3_hsdp_tp
```

| Lab | Teaches | Success criterion |
|-----|---------|-------------------|
| `l3_mesh` | build the 3D mesh; each rank's role + physical mapping; illustrative 4D | roles reported; 4D mesh constructs (labeled illustrative) |
| `l3_fsdp` | FSDP2 over a 2D HSDP mesh + DCP save/restore/resume | FSDP2 model resumes identically (< 1e-6) |
| `l3_hsdp_tp` | HSDP + TP composed on the real 3D mesh | parity vs single-device < 1e-4 (≈ 2.4e-7) |
| `l3_moe` | MoE feed-forward under FSDP2; routing imbalance | all tokens routed; imbalance ≥ 1.0 |
| `l3_compile` | regional `torch.compile` of the block | compiled output matches eager (< 1e-4) |
| `l3_faultdata` | deterministic restart across a simulated crash | resumed batches exactly match an uninterrupted run |
| `l3_nccl_debug` | **failure-driven**: collective-shape check + Flight Recorder | disagreement detected; Flight-Recorder dump written (GPU) |
| `l3_capstone` | MoE under HSDP + model-only DCP + overlap + postmortem | resume < 1e-6; routing imbalance reported |

### Notes on specific labs

- **`l3_hsdp_tp`** is the flagship: tensor-parallelize the block on the `tp` dim,
  then `fully_shard` over the `(dp_replicate, dp_shard)` sub-mesh, and confirm the
  output still matches a single-device reference — verified across all 8 ranks on
  two nodes.
- **`l3_nccl_debug`** is the intentional failure. `shapes_agree` all-gathers a
  fixed-size scalar (never itself mismatching) to catch the rank-to-rank shape
  disagreement that would otherwise **hang** a NCCL collective; on GPU it then
  dumps a Flight-Recorder trace per rank. Because the container's
  `TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated in torch 2.10, set the current
  variable for a reliable dump:

  ```bash
  TORCH_FR_BUFFER_SIZE=20971520 sbatch dtensor/slurm/launch_2node.sbatch -m labs.level3.l3_nccl_debug
  ```

  Dumps land at `artifacts/l3_flight_recorder_rank{N}.dump`.
- **`l3_capstone`** trains a small **MoE** model under FSDP2/HSDP with the overlap
  knob, records routing imbalance, and checkpoints **model state only** — a
  deliberate choice: the MoE's argmax router never receives gradients, so it has
  no optimizer state and full model+optimizer DCP raises a missing-momentum-buffer
  mismatch. (Full optimizer-state DCP recovery is demonstrated in `l3_fsdp`.)

### FP8 — what works and what doesn't (validated on H100)

`dtensor_workshop.fp8.maybe_convert_fp8` (torchao) is a **no-op off Hopper** and
converts eligible linears on Hopper. Confirmed on H100:

- ✅ **Dense, 16-aligned layers** (the Transformer block) train in FP8 cleanly.
- ❌ **Sparse MoE** does **not** work out of the box: Hopper FP8 (`_scaled_mm`)
  requires the token dimension divisible by 16, but MoE routing dispatches a
  variable, non-aligned token count per expert. FP8 + MoE needs token padding —
  an advanced technique beyond this workshop — so the capstone does **not**
  FP8-convert the MoE (its overlap knob is the optimization instead).

## Capstone

`l3_capstone` is the culmination: a documented 3D/HSDP mesh, FSDP2 sharding, DCP
recovery, a communication/compute-overlap optimization, MoE routing analysis, and
a postmortem-style debugging note.

```bash
sbatch dtensor/slurm/launch_2node.sbatch -m labs.level3.l3_capstone
```

## What you can do after Level 3

- Build and reason about a 3D `dp_replicate × dp_shard × tp` mesh mapped to a
  real node/NVLink topology.
- Shard a model with FSDP2 and compose it with tensor parallelism (HSDP+TP).
- Add an MoE layer and quantify routing imbalance.
- Apply regional `torch.compile` and FP8 where they're valid — and recognize
  where FP8 isn't (sparse MoE token alignment).
- Make training restartable with a deterministic, fault-aware data loader.
- Catch the shape disagreement that hangs a collective, and pull a Flight-Recorder
  trace to isolate a failing rank.

## Verification

- **CPU tests:** `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_parallel3d.py tests/test_moe.py tests/test_l3_hsdp_tp.py tests/test_l3_fsdp.py tests/test_l3_capstone.py tests/test_l3_nccl_debug.py tests/test_l3_faultdata.py tests/test_l3_compile.py tests/test_fp8.py -v'`
- **GPU smoke (validated on 2× H100 nodes):** `l3_hsdp_tp` parity `2.38e-07` across 8 ranks; `l3_nccl_debug` Flight-Recorder dumps on all 8 ranks; `l3_capstone` resume `0.0`, imbalance `1.438`; dense-block FP8 forward/backward runs cleanly on Hopper.
