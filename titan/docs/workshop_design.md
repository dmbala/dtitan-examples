# TorchTitan Workshop — Design Spec

> Infrastructure-grounded revision of `initial_plan.md`.
> Target environment: Kempner cluster on FASRC, H100 (80 GB) / H200 nodes,
> 4 GPUs/node, up to 2 nodes, Slurm, a **rebuilt** `dtitan.sif` Apptainer image.
> Companion to the standalone-DTensor workshop in `../dtensor/`.
> Date: 2026-07-10.

## Purpose

A three-level workshop teaching **TorchTitan** as a PyTorch-native platform for
large-scale generative-model training. The arc moves from a small observable run
→ a restartable, profile-driven multi-parallel run → a production-style cluster
deployment plan. Emphasis is on practical operating skills: reading TorchTitan's
training flow and extension points, choosing and overriding configs safely,
composing PyTorch-native parallelism with minimal model changes, and
saving/loading/profiling/debugging distributed jobs — **validating correctness
before optimizing throughput**.

### Relationship to the DTensor workshop

This is the **framework** counterpart to the **primitives** workshop in
`../dtensor/`. There, participants build `DeviceMesh`/`DTensor`/FSDP2/DCP by
hand; here they operate the same mechanics through TorchTitan's configs and
launchers. The two cross-reference: DTensor explains what TorchTitan does under
the hood; TorchTitan shows how it's run in production. Both share the same
container and cluster.

## Audience

- Can train and debug a single-GPU PyTorch model; understands tensors, autograd,
  optimizers, data loaders, and the command line.
- May be new to multi-node launchers, `DeviceMesh`, Distributed Checkpointing,
  NCCL debugging, or profiler traces.
- Level 1 runs on one node (or TorchTitan's fake-backend dry-run); Levels 2–3
  scale up.

---

## Environment & Runbook

Shared by all levels; each level references it rather than repeating launch
details.

### Container (rebuild prerequisite — read first)

**The current `dtitan.sif` cannot run this workshop.** It ships **torchtitan
0.2.2** on **torch 2.10.0a0** (NGC 25.11), and `import torchtitan.train` fails:

```
ImportError: cannot import name '_context_parallel_shard' from
             torch.distributed.tensor.experimental._attention
```

torchtitan 0.2.2 imports `_context_parallel_shard`, a **torch 2.11** symbol
absent from the 2.10 preview. **Prerequisite:** rebuild the image on an **NGC
base with torch ≥ 2.11 stable** (a later monthly NGC tag), keeping
**torchtitan 0.2.2**, plus torchao and flash-attn:

```
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif   # rebuilt: torch>=2.11
```

Choosing torch ≥ 2.11 (rather than pinning torchtitan back to 0.2.0) unlocks
**context parallelism**, **varlen attention**, and the newest model specs.
After rebuilding, **re-validate the DTensor workshop's 62 tests** on the new
torch (DTensor APIs are stable; expected to pass, but confirm).

The container keeps the offline/NCCL environment the labs rely on
(`HF_HUB_OFFLINE=1`, `HF_HOME=/data/hf_cache`, `TORCH_NCCL_*`); do not re-set
these. For Flight Recorder on torch ≥ 2.11 use **`TORCH_FR_BUFFER_SIZE`** (the
`TORCH_NCCL_TRACE_BUFFER_SIZE` the image exports is deprecated — a lesson carried
from the DTensor workshop).

### Models & data (offline testbed)

All assets are pre-staged and used **offline** from:

```
MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models
```

| Use | Asset | How |
|-----|-------|-----|
| Tokenizer (everywhere) | `$MODELS/Llama-3.1-8B-Instruct` | `--model.tokenizer_path=$MODELS/Llama-3.1-8B-Instruct` (real Llama-3.1 tokenizer) |
| Fast path (all levels) | torchtitan `llama3` **debug config** (tiny, random init) | quick, self-contained runs with the real tokenizer + synthetic data |
| Trainable real target (L1–L2) | **Llama-3.1-8B** (`llama3` spec) | short FSDP2 / FSDP2+TP runs; optional **HF→DCP conversion** of the 8B safetensors as a training init |
| Scaling / architecture demos (L3) | **Llama-3.1-70B**, **Llama-3.1-405B**, **DeepSeek-R1** (`deepseek_v3` MoE) | mesh/topology planning and architecture walkthroughs — **not** full training runs (see caveat) |

**Caveats:** (1) **8B is the largest *trainable* target on 8×H100** — 70B/405B
don't fit for training on this budget and are used for planning/inference-format
demos only. (2) Loading real Llama weights requires TorchTitan's **HF→DCP
conversion** step; the from-init path needs no weights. (3) The testbed holds
weights + tokenizers, **not datasets** — training data uses TorchTitan's offline
synthetic / HF-offline datasets.

### Config strategy

The installed torchtitan wheel ships **no** `train_configs/*.toml`, so the
workshop **provides its own** under `titan/configs/`:

- `debug.toml` — tiny `llama3`, the fast path.
- `l1_fsdp.toml`, `l2_fsdp_tp.toml`, `l3_hsdp_tp.toml`, `l3_moe.toml` — per-level baselines.

Runs select a config with `--job.config_file` and layer **dotted CLI overrides**
on top. Every lab begins by printing the resolved config
(`python -m torchtitan.config.manager …`) before launching. Representative
override paths (exact names confirmed against `torchtitan.train --help` on the
rebuilt image): `--training.steps`, `--training.local_batch_size`,
`--training.seq_len`, `--parallelism.data_parallel_shard_degree`,
`--parallelism.data_parallel_replicate_degree`,
`--parallelism.tensor_parallel_degree`,
`--parallelism.context_parallel_degree`,
`--parallelism.pipeline_parallel_degree`,
`--parallelism.expert_parallel_degree`, `--checkpoint.enable_checkpoint`,
`--checkpoint.interval`, `--activation_checkpoint.mode`,
`--profiling.enable_profiling`, `--profiling.enable_memory_snapshot`.

### Slurm launchers

Adapt the **already-validated** DTensor launchers (`../dtensor/slurm/`) — same
scaffold that was GPU-tested on `kempner_h100`: `--account=kempner_dev`,
`--partition=kempner_h100`, `--nv`, bind mounts, **`--mem=128G`**, and (2-node)
**`srun --cpu-bind=none`**. Only the entrypoint changes — they invoke TorchTitan:

```bash
# 1 node / 4 GPUs (Levels 1 & 2)
singularity exec --nv --bind $(pwd)/outputs:/outputs "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
    --job.config_file=configs/l1_fsdp.toml "$@"

# 2 nodes / 8 GPUs (Level 3) — c10d rendezvous, srun --cpu-bind=none
srun --cpu-bind=none singularity exec --nv --bind $(pwd)/outputs:/outputs "$IMAGE" \
  torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" -m torchtitan.train \
    --job.config_file=configs/l3_hsdp_tp.toml "$@"
```

**Cluster note:** the per-user GPU cap on `kempner_h100` is **8 GPUs**, so
2-node/8-GPU jobs use the entire budget and **run one at a time**.

### Preflight (script — built later)

Run before the workshop; one pass/fail line per check:

1. `import torchtitan` **and** `import torchtitan.train` (catches the torch/torchtitan mismatch above).
2. Config resolves (`torchtitan.config.manager` on `debug.toml`).
3. GPU visibility (`torch.cuda.device_count() == 4`) and a 2-rank NCCL all-reduce.
4. Tokenizer path readable (`$MODELS/Llama-3.1-8B-Instruct`).
5. Write access to `outputs/` (logs, checkpoints, snapshots, traces).
6. A `COMM_MODE=fake_backend` dry-run of a larger intended config **without** real GPUs.

### Output & retention

All artifacts under `outputs/` (bound into the container): logs, DCP
checkpoints, memory snapshots, profiler traces. **Retention:** keep the last *N*
full DCP checkpoints + one model-only final checkpoint; prune profiler traces and
old snapshots after each level. Metrics are **local** — TorchTitan's built-in
metrics logging (loss, memory, tokens/sec, TFLOPs, MFU) to stdout/log files, plus
profiler/memory/DCP artifacts. No live W&B (the container is offline).

---

## Resolved Decisions

The plan's "Open Decisions", settled:

| Decision | Value |
| --- | --- |
| TorchTitan release | **0.2.2** |
| PyTorch / TorchAO | **torch ≥ 2.11** (rebuilt container) / torchao (build-latest) |
| GPU type / count / multi-node | H100 80 GB (`kempner_h100`; H200 also available), 4/node, up to 2 nodes; **8-GPU/user cap** → 2-node jobs serialize |
| Model & data | offline **testbed**: real Llama-3.1 tokenizer + trainable **Llama-3.1-8B**; `llama3` debug config fast path; larger Llamas + DeepSeek for planning; TorchTitan offline datasets |
| Metrics | **local logs** + TorchTitan metrics (no live W&B) |
| Checkpoint retention | last *N* full DCP + model-only final; prune traces |
| Level 3 features | **core:** HSDP+TP, DCP, FP8 (torchao on Hopper), regional `torch.compile`, async TP, NCCL/Flight-Recorder debugging. **optional at 8 GPUs:** pipeline & expert parallelism / MoE (small configs), context parallelism. **out:** MXFP8 (Blackwell-only) |

---

## Level 1 — Running and Observing TorchTitan

**Title:** TorchTitan Foundations: Configs, FSDP2, Metrics, and First Debugging

**Environment / launch:** 1 node, 4 GPUs, `kempner_h100`; 1-node launcher with
`--job.config_file=configs/l1_fsdp.toml`.

### Goals

A concrete mental model of how TorchTitan starts a job, applies the selected
model spec and parallelism, emits logs/metrics, and saves enough to debug a small
run.

### Core topics

Repo tour (`torchtitan/train.py`, model specs under `torchtitan/models/`,
components for checkpoint/metrics/profiling); config flow (`--job.config_file`,
dotted CLI overrides, resolving with `torchtitan.config.manager`); launch basics
(rank/world/local-rank, GPU assignment, `COMM_MODE=fake_backend` dry-run); **1D
FSDP2** baseline; observability (rank-aware logs, loss/memory/throughput/MFU);
first profiler trace; beginner troubleshooting (bad config names/overrides, GPU
count mismatch, missing tokenizer, multi-rank stack traces).

### Labs

Milestone order: **correct → observable**.

| # | Lab | Command / action | Expected artifact | Success criterion |
|---|-----|------------------|-------------------|-------------------|
| 1 | Preflight | `python preflight.py` | pass/fail lines | all checks pass (incl. `torchtitan.train` import) |
| 2 | Inspect a config + override | `torchtitan.config.manager` on `debug.toml`, apply 2 overrides | resolved-config dump | overridden values appear as expected |
| 3 | Fake-backend dry-run | `COMM_MODE=fake_backend … --job.config_file=configs/l1_fsdp.toml` | dry-run log | config launches without real GPUs |
| 4 | 1D FSDP2 run (debug) | `sbatch launch_1node.sbatch --training.steps=20` | training log | loss decreases; run completes |
| 5 | Metrics + profiler | add `--profiling.enable_profiling` | trace + metrics log | loss/memory/tokens-per-sec/MFU located in the log |
| 6 | **Failure-driven:** break a config value | set an invalid override, read the failure | captured error + fix | participant states the root cause and fixes it |
| — | Real-model taste | `--model.tokenizer_path=$MODELS/Llama-3.1-8B-Instruct` + `llama3` 8B, `--training.steps=5` | short 8B log | an 8B step runs under 1D FSDP2 |

### Capstone

A small workshop config variant / override set for a short 1D FSDP2 run,
submitting: launch command, resolved config, training log, a profiler artifact,
and a note on whether loss and throughput look plausible.

### Instructor notes

~half day. Common errors: invalid config path, GPU-count mismatch, tokenizer path
typo. Always resolve-and-print the config before an expensive launch.

---

## Level 2 — Composable Parallelism and Restartable Training

**Title:** FSDP2 + Tensor Parallelism: Checkpointing, Memory, and Bottleneck Profiling

**Environment / launch:** 1 node, 4 GPUs. 1-node launcher,
`--job.config_file=configs/l2_fsdp_tp.toml`.

### Goals

Combine parallelism dimensions, validate correctness against a 1D baseline, save
and resume distributed state, and use profiling to cut memory/communication cost.

### Core topics

Parallelism config (`data_parallel_shard_degree`, `tensor_parallel_degree`,
`context_parallel_degree` where supported; how mesh dims map to ranks); FSDP2
behavior (per-parameter DTensor sharding, sharded model+optimizer state,
reshard tradeoffs); tensor parallelism (column/row sharding, sequence-parallel
implications; spotting all-reduce/all-gather/reduce-scatter in traces);
Distributed Checkpointing (full state, model-only final, **async save**, **seed
checkpoints**, **DCP resharding** across layouts); memory (snapshots, OOM,
full/selective activation checkpointing); correctness (loss-parity vs 1D FSDP,
consistent global batch size and data order, documented numeric tolerance).

### Labs

Milestone order: **correct → restartable → fast**.

| # | Lab | Command / action | Expected artifact | Success criterion |
|---|-----|------------------|-------------------|-------------------|
| 1 | Convert L1 → 2D | add `--parallelism.tensor_parallel_degree=2 --parallelism.data_parallel_shard_degree=2` | training log | job runs as `dp2 × tp2` |
| 2 | Rank-layout diagram | annotate the mesh from logs | labeled diagram | `dp`/`tp` groups correctly identified |
| 3 | DCP save → resume | `--checkpoint.enable_checkpoint --checkpoint.interval=10`, then resume | `outputs/checkpoint/` | resumed job continues from the expected step |
| 4 | Seed checkpoint reuse | create a seed checkpoint; reuse for two layouts | seed checkpoint | both layouts start from identical init |
| 5 | **Failure-driven:** OOM | grow `--training.local_batch_size` / `seq_len` until OOM; snapshot | memory snapshot | largest contributors identified |
| 6 | Activation checkpointing | `--activation_checkpoint.mode=selective` (then `full`) | before/after metrics | peak memory drops; loss behavior unchanged |
| 7 | Bottleneck profiling | profile a 2D step | trace | most expensive communication region named |
| — | Real 8B, 2D | `llama3` 8B with `dp × tp`, short run | 8B log | 8B trains under FSDP2+TP |

### Capstone

A restartable 2D run: parallelism settings, rank-layout diagram, DCP save+resume
evidence, one memory optimization, one profiler-based bottleneck diagnosis, and a
short loss comparison against the Level 1 baseline.

### Instructor notes

~full day. Common errors: TP degree not dividing heads/hidden; forgetting to
match global batch size when comparing to the 1D baseline; OOM lab needs a size
knob calibrated for 80 GB (H200 needs larger). Context parallelism is a labeled
module (now available on torch ≥ 2.11).

---

## Level 3 — Production Scaling and Extensibility

**Title:** Multi-Node TorchTitan: Pipeline, Context, Expert Parallelism, Precision, and NCCL Debugging

**Environment / launch:** 2 nodes, 8 GPUs, `kempner_h100`. 2-node launcher
(`srun --cpu-bind=none`, c10d rendezvous), `--job.config_file=configs/l3_hsdp_tp.toml`.
`NCCL_DEBUG=INFO`; `TORCH_FR_BUFFER_SIZE=20971520`; `CUDA_DEVICE_MAX_CONNECTIONS=1`
for async TP.

### Goals

Plan a larger TorchTitan deployment, make model/data changes through supported
extension points, and debug failures that only appear at multi-node scale.

### Core topics

Cluster execution (Slurm, rendezvous, shared-FS behavior, artifact placement,
cleanup); higher-dimensional parallelism (HSDP+TP plus **pipeline** and/or
**context** parallelism; **expert parallelism / MoE** via `deepseek_v3`/`llama4`;
documenting each mesh dim); pipeline-friendly model structure (clean top-level
forwards, preserved FQNs for checkpoint compatibility, seed checkpoints);
performance (regional `torch.compile`, async TP, **Float8 (torchao) — core on
Hopper**, overlap tuning; interpreting tokens/sec, TFLOPs, MFU); extensibility
(register a model via `ModelSpec`, add config entries, swap data loaders);
advanced debugging (NCCL hangs/timeouts/mismatched collectives, **Flight
Recorder** dumps, deterministic debug mode, comparing failing ranks).

**Feasibility at 8 GPUs (honest scoping):** a real 3D layout (e.g.
`dp_replicate2 × dp_shard2 × tp2`) is runnable; **pipeline and expert
parallelism** are demonstrated at **small/debug configs** (they'd shine at larger
scale); **4D** is taught as code-that-would-scale. **MXFP8 is out** (Blackwell
only). **Llama-3.1-70B/405B and DeepSeek-R1** are used for **mesh/topology
planning** — "here's the layout you'd need" — not full runs.

### Labs

Milestone order: **correct → restartable → fast → debuggable**.

| # | Lab | Command / action | Expected artifact | Success criterion |
|---|-----|------------------|-------------------|-------------------|
| 1 | Adapt the multi-node launcher | tiny 2-node validation run | job record | 8-rank job starts and steps |
| 2 | 2D → 3D layout | add a mesh dim; document each dim's role | topology diagram | every mesh dim's purpose stated |
| 3 | Pipeline or context parallelism | small `pp`/`cp` config vs a known-good baseline | loss comparison | loss tracks the baseline |
| 4 | A performance feature | enable FP8 (`torchao`) **or** `torch.compile` **or** async TP; measure | before/after metrics | throughput/memory change reported with evidence |
| 5 | Extensibility | register a small config / dataset / model variant | the added entry, used in a run | run uses the extension without forking the loop |
| 6 | **Failure-driven:** injected NCCL fault | inject a timeout / collective mismatch | Flight-Recorder dump + rank logs | failing rank/collective identified |
| — | Scaling plan | mesh plan for **Llama-3.1-70B/405B** or DeepSeek MoE | written topology plan | a defensible layout + memory budget |

### Capstone

A production-style deployment package for a larger scenario: Slurm launch script,
resolved config, topology diagram, checkpoint policy, metrics/profiling plan, one
**measured** performance optimization, and a postmortem-style note for an injected
distributed failure.

### Instructor notes

~full day. Common errors: rendezvous (wrong `MASTER_ADDR`, port in use);
collective mismatches hang rather than error (Flight Recorder is the tool); FP8
numerics diverge without proper config. Confirm `TORCH_FR_BUFFER_SIZE` is set for
the debugging lab. Only one 8-GPU job runs at a time (per-user cap).

---

## Cross-Cutting Workshop Design

- **One model family throughout** — `llama3` (debug config → real Llama-3.1-8B);
  MoE demos via `deepseek_v3`.
- **Milestone ordering** per level: correct → restartable → fast → debuggable.
- **Start from a known-good baseline** each level; **one intentional failure**
  per level (bad config → OOM → injected NCCL fault).
- **Config-inspection habit:** every lab prints the resolved config before an
  expensive launch.
- **Required artifacts per level:** launch command/script, resolved config, logs
  from rank 0 **and** a non-zero rank, metrics output, a profiler/memory/checkpoint
  artifact, and a short written diagnosis.
- **Known-good reference artifacts** provided under `outputs/reference/`:
  resolved configs, logs, a profiler trace, a memory snapshot, and a seed
  checkpoint, so participants can compare when hardware differs.
- **Buildable deliverables** (what the implementation plan produces): `titan/configs/*.toml`,
  the adapted Slurm launchers, `preflight.py`, per-lab guide docs, and the
  reference artifacts.

## Assessment Rubric

| Dimension | What is checked |
| --- | --- |
| Correctness | run starts, trains, preserves expected loss behavior |
| Configuration | participant explains the effective config and overrides |
| Reliability | DCP save and resume work as expected |
| Observability | logs/metrics/traces captured and interpreted |
| Debugging | root cause of the injected failure identified |
| Performance | optimization claims backed by measured tokens/sec, memory, or MFU |

## Notes on revisions from `initial_plan.md`

- **Container reality surfaced:** the current image can't import `torchtitan.train`
  (0.2.2 on torch 2.10); the spec makes a **torch ≥ 2.11 rebuild** a hard prerequisite.
- **Open Decisions resolved** into concrete values (see table), grounded in the
  container, cluster, and the offline model testbed.
- **Assets grounded** in `…/testbed/models/`: real Llama-3.1 tokenizer +
  trainable Llama-3.1-8B; larger Llamas / DeepSeek for planning.
- **Config interface grounded** to torchtitan 0.2.2 (`--job.config_file` + dotted
  overrides; workshop ships its own TOMLs since none are in the wheel).
- **Launchers reuse** the DTensor workshop's GPU-validated Slurm scaffold
  (`--account=kempner_dev`, `--mem`, `srun --cpu-bind=none`, 8-GPU cap).
- **Level 3 scoped honestly** to 8 GPUs: FP8 core; pipeline/expert/context
  parallelism as small-config modules; MXFP8 out; 70B/405B for planning.
- **Cross-linked** to the DTensor workshop (primitives ↔ framework).
