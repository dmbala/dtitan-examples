# TorchTitan Workshop — Design Spec

> Infrastructure-grounded revision of `initial_plan.md`.
> Target environment: Kempner cluster on FASRC, `kempner_rtx` (RTX PRO 6000
> Blackwell) nodes, 8 GPUs/node, up to 2 nodes, Slurm, the built
> `dtitan-torch211.sif` Apptainer image (torch 2.11.0a0 / CUDA 13.2).
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

### Container (built — `dtitan-torch211.sif`)

The workshop runs on **`dtitan-torch211.sif`**:

```
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif
```

torchtitan 0.2.2 imports `_context_parallel_shard`
(`torch.distributed.tensor.experimental._attention`) and
`activate_flash_attention_impl`, both **torch-2.11** symbols absent from torch
2.10. The plain **`dtitan.sif`** (torchtitan 0.2.2 on torch 2.10.0a0, NGC 25.11)
is the **dtensor** workshop's image — on it, only the bare `torchtitan.config`
machinery imports; `torchtitan.train` and any model module do not. Do not use
`dtitan.sif` for this workshop. `dtitan-torch211.sif` is built on an NGC base
with **torch 2.11.0a0** (NGC 26.03, CUDA 13.2), keeping **torchtitan 0.2.2**,
plus torchao and flash-attn. Choosing torch ≥ 2.11 (rather than pinning
torchtitan back to 0.2.0) unlocks **context parallelism**, **varlen attention**,
and the newest model specs.

**Reference note — why `kempner_rtx`, not `kempner_h100`:** the torch-2.11
image is **CUDA 13.2**. `kempner_h100` nodes run driver **575.57.08** (CUDA
12.9); its forward-compat window reaches CUDA 13.0 but not 13.2, so
`torch.cuda.init()` fails there with `RuntimeError: The NVIDIA driver on your
system is too old (found version 12090)`. `kempner_rtx` (RTX PRO 6000
Blackwell) nodes run driver **595.71.05** (CUDA 13.2), so the image runs there.
Conversely, the torch-2.10 `dtitan.sif` image has no `sm_120` kernels and does
not run on Blackwell at all — so Blackwell specifically needs the torch-2.11
image. **Gotcha:** `torch.cuda.device_count()` can return `1` with only a
*warning* even when the driver is broken — always confirm with a real tensor
op or `torch.cuda.init()`, not just `device_count()`.

The container keeps the offline/NCCL environment the labs rely on
(`HF_HUB_OFFLINE=1`, `HF_HOME=/data/hf_cache`, `TORCH_NCCL_*`); do not re-set
these. `HF_HOME` is **read-only** at runtime, so the launchers export
`SINGULARITYENV_HF_HOME` / `SINGULARITYENV_HF_DATASETS_CACHE` pointing at the
writable `outputs/hf_cache/` — participants don't set anything. For Flight
Recorder on torch ≥ 2.11 use **`TORCH_FR_BUFFER_SIZE`** (the
`TORCH_NCCL_TRACE_BUFFER_SIZE` the image exports is deprecated — a lesson carried
from the DTensor workshop).

### Models & data (offline testbed)

All assets are pre-staged and used **offline**:

```
MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models
```

| Use | Asset | How |
|-----|-------|-----|
| Fast path (all levels, debug flavor) | small vendored tokenizer, `titan/assets/test_tokenizer/` (vocab 2016) | `--model.hf_assets_path=assets/test_tokenizer` |
| Trainable real target (L1–L2) | **Llama-3.1-8B** tokenizer + weights, `$MODELS/Llama-3.1-8B-Instruct` (`llama3` spec, `--model.flavor 8B`) | short FSDP2 / FSDP2+TP runs; optional **HF→DCP conversion** of the 8B safetensors as a training init |
| Scaling / architecture demos (L3) | **Llama-3.1-70B**, **Llama-3.1-405B**, **DeepSeek-R1** (`deepseek_v3` MoE) | mesh/topology planning and architecture walkthroughs — **not** full training runs (see caveat) |
| Training data (all levels) | generated **C4 subset**, `titan/assets/c4_subset/train.jsonl` (3000 real C4 docs, git-ignored) | one-time `python scripts/prepare_c4_subset.py`, then `--training.dataset=c4_test --training.dataset_path=assets/c4_subset` |

**Why not the real Llama-3.1 tokenizer everywhere:** the `debugmodel` flavor
hardcodes `vocab_size=2048`, and torchtitan 0.2.2 does **not** resize the
embedding from the tokenizer. The real Llama-3.1 tokenizer (vocab 128256)
overflows the debug model's embedding (`vectorized_gather_kernel: index out of
bounds`) — it is only valid paired with `--model.flavor 8B`. The workshop's
fast path therefore uses the small vendored tokenizer for the debug flavor, and
reserves the real tokenizer for the 8B "real-model taste" (Level 2).

**Why the generated C4 subset:** the pip wheel does not ship
`tests/assets/c4_test`, and the built-in `c4` config streams `allenai/c4` from
the Hub, which `HF_HUB_OFFLINE=1` blocks. `scripts/prepare_c4_subset.py`
extracts a local subset from the testbed's real C4 shards
(`.../testbed/text/c4_original/raw/c4_en_train`) that the `c4_test` loader
reads via `load_dataset(dir, split="train")`.

**Caveats:** (1) **8B is the largest *trainable* target** — 70B/405B don't fit
for training on this budget and are used for planning/inference-format demos
only. (2) Loading real Llama weights requires TorchTitan's **HF→DCP
conversion** step; the from-init path needs no weights.

### Config strategy

TorchTitan 0.2.2 selects the model and flavor via CLI flags, not TOML files:
**`--model.name <model> --model.flavor <flavor>`** — there is no `--module`/
`--config` form; passing those is rejected (`Unrecognized options: --module,
--config`). The workshop's fast path is the built-in **`--model.name llama3
--model.flavor debugmodel`**; workshop-specific variants would be added by
**registering entries in the model's `config_registry`** (a small
`titan/configs/` Python module), not by shipping TOMLs — not yet exercised in
Level 1. Every lab layers **dotted CLI overrides** on top and prints the
resolved config before launching. Verified override paths (from the 0.2.2
config dataclasses): `--training.steps`, `--training.local_batch_size`,
`--training.seq_len`, `--training.dataset`, `--training.dataset_path`,
`--model.hf_assets_path`, `--parallelism.data_parallel_shard_degree`,
`--parallelism.data_parallel_replicate_degree`,
`--parallelism.tensor_parallel_degree`,
`--parallelism.context_parallel_degree`,
`--parallelism.pipeline_parallel_degree`,
`--parallelism.expert_parallel_degree`, plus `--checkpoint.*`,
`--activation_checkpoint.mode`, and **`--profiling.enable_profiling`** /
`--profiling.profile_freq` / `--profiling.save_traces_folder` (the section is
`profiling`, not `profiler`).

### Slurm launchers

Adapt the **already-validated** DTensor launchers (`../dtensor/slurm/`) — same
scaffold, GPU-tested on `kempner_rtx`: `--account=kempner_dev`,
`--partition=kempner_rtx`, `--nv`, bind mounts, **`--mem=128G`**, and (2-node)
**`srun --cpu-bind=none`**. Only the entrypoint changes — they invoke TorchTitan:

```bash
# 1 node / 4 GPUs (Levels 1 & 2)
singularity exec --nv --bind $(pwd)/outputs:/outputs "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
    --model.name llama3 --model.flavor debugmodel "$@"

# 2 nodes / 8 GPUs (Level 3) — c10d rendezvous, srun --cpu-bind=none
srun --cpu-bind=none singularity exec --nv --bind $(pwd)/outputs:/outputs "$IMAGE" \
  torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" -m torchtitan.train \
    --model.name llama3 --model.flavor <workshop-registered-flavor> "$@"
```

**Cluster note:** the per-user GPU cap on `kempner_rtx` is **8 GPUs**, so
2-node/8-GPU jobs use the entire budget and **run one at a time**.
`kempner_rtx` nodes have 8 GPUs/node, so an 8-GPU job actually fits on **one**
node — the 2-node launcher above targets Level 3's larger multi-node
topologies, not a hardware requirement at 8 GPUs.

### Preflight (`preflight.py`)

Run before the workshop; one pass/fail line per check, 6 total:

1. `import torchtitan` (config machinery).
2. `import torchtitan.train` (catches the torch/torchtitan mismatch above — the key gate).
3. Tokenizer path readable (`assets/test_tokenizer`, the vendored debug tokenizer).
4. C4 subset present (`assets/c4_subset/train.jsonl`) — fails with a "run
   `python scripts/prepare_c4_subset.py`" hint until generated.
5. Write access to `outputs/` (logs, checkpoints, snapshots, traces).
6. GPU visibility (`torch.cuda.device_count() == 4`; needs a `kempner_rtx` GPU
   allocation with `--gpus-per-node=4`).

Config resolution (Lab 2) and the fake-backend dry-run (Lab 3) are exercised as
separate labs, not part of preflight.

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
| GPU type / count / multi-node | `kempner_rtx` (RTX PRO 6000 Blackwell), 8/node, up to 2 nodes; Level 1–2 jobs use 4 of 8 GPUs/node; **8-GPU/user cap** → 2-node jobs serialize. (`kempner_h100`'s CUDA-12.9 driver cannot run the torch-2.11 image — see the container reference note.) |
| Model & data | offline **testbed**: small vendored debug tokenizer + generated **C4 subset** fast path (`llama3` debug flavor); real Llama-3.1 tokenizer + trainable **Llama-3.1-8B** (flavor `8B` only); larger Llamas + DeepSeek for planning |
| Metrics | **local logs** + TorchTitan metrics (no live W&B) |
| Checkpoint retention | last *N* full DCP + model-only final; prune traces |
| Level 3 features | **core:** HSDP+TP, DCP, FP8 (torchao on Hopper), regional `torch.compile`, async TP, NCCL/Flight-Recorder debugging. **optional at 8 GPUs:** pipeline & expert parallelism / MoE (small configs), context parallelism. **out:** MXFP8 (Blackwell-only) |

---

## Level 1 — Running and Observing TorchTitan

**Title:** TorchTitan Foundations: Configs, FSDP2, Metrics, and First Debugging

**Environment / launch:** 1 node, 4 GPUs, `kempner_rtx`; 1-node launcher with
the built-in `--model.name llama3 --model.flavor debugmodel`.

### Goals

A concrete mental model of how TorchTitan starts a job, applies the selected
model spec and parallelism, emits logs/metrics, and saves enough to debug a small
run.

### Core topics

Repo tour (`torchtitan/train.py`, model specs under `torchtitan/models/`,
components for checkpoint/metrics/profiling); config flow (`--model.name`/
`--model.flavor`, dotted CLI overrides, resolving with
`torchtitan.config.manager`); launch basics (rank/world/local-rank, GPU
assignment, `NGPU=4 torchrun --standalone --nproc_per_node=1 -m
torchtitan.train … --comm.mode=fake_backend` dry-run); **1D FSDP2** baseline;
observability (rank-aware logs, loss/memory/throughput/MFU); first profiler
trace; beginner troubleshooting (bad model name/flavor or overrides, GPU count
or partition/driver mismatch, missing tokenizer or C4 subset, multi-rank stack
traces).

### Labs

Milestone order: **correct → observable**.

| # | Lab | Command / action | Expected artifact | Success criterion |
|---|-----|------------------|-------------------|-------------------|
| 1 | Preflight | `python preflight.py` | pass/fail lines (6 checks) | all checks pass (incl. `torchtitan.train` import, C4 subset, `gpu visible: device_count=4`) |
| 2 | Inspect a config + override | `ConfigManager().parse_args(["--model.name","llama3","--model.flavor","debugmodel", …])`, apply 2 overrides | resolved-config dump | overridden values appear as expected |
| 3 | Fake-backend dry-run | `NGPU=4 torchrun --standalone --nproc_per_node=1 -m torchtitan.train --model.name llama3 --model.flavor debugmodel --comm.mode=fake_backend` | dry-run log | model builds, "Applied FSDP to the model", ends `Training completed` — no real GPUs used |
| 4 | 1D FSDP2 run (debug) | `sbatch launch_1node.sbatch --model.hf_assets_path=assets/test_tokenizer --training.dataset=c4_test --training.dataset_path=assets/c4_subset --training.steps=20 --parallelism.data_parallel_shard_degree=4` | training log | loss decreases (step 1 ≈ 8.12 → step 20 ≈ 3.55); run completes |
| 5 | Metrics + profiler | add `--profiling.enable_profiling` | trace + metrics log | loss/memory/tokens-per-sec/MFU located in the log; trace files under `outputs/profile_traces/` |
| 6 | **Failure-driven:** break a config value | set an invalid override (e.g. `--parallelism.tensor_parallel_degree=3` on 4 GPUs), read the failure | captured error + fix | participant states the root cause (parallel-dims product ≠ `WORLD_SIZE`) and fixes it |

The real Llama-3.1 tokenizer + trainable Llama-3.1-8B ("real-model taste")
first appears in **Level 2** (`--model.flavor 8B`) — it is not paired with the
debug flavor (see the Models & data caveat above).

### Capstone

A small workshop config variant / override set for a short 1D FSDP2 run,
submitting: launch command, resolved config, training log, a profiler artifact,
and a note on whether loss and throughput look plausible.

### Instructor notes

~half day. Common errors: invalid model name/flavor, GPU-count mismatch,
wrong partition (`kempner_h100` instead of `kempner_rtx`), tokenizer/C4-subset
path typo. Always resolve-and-print the config before an expensive launch.

---

## Level 2 — Composable Parallelism and Restartable Training

**Title:** FSDP2 + Tensor Parallelism: Checkpointing, Memory, and Bottleneck Profiling

**Environment / launch:** 1 node, 4 GPUs (`kempner_rtx`). 1-node launcher,
`--model.name llama3` with the Level 2 registered flavor (or `--model.flavor
debugmodel` + parallelism overrides).

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

**Environment / launch:** 2 nodes, 8 GPUs, `kempner_rtx`. 2-node launcher
(`srun --cpu-bind=none`, c10d rendezvous), `--model.name llama3` with the Level 3
registered flavor. `NCCL_DEBUG=INFO`; `TORCH_FR_BUFFER_SIZE=20971520`;
`CUDA_DEVICE_MAX_CONNECTIONS=1` for async TP.

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

- **Container built & GPU-validated:** the original image (`dtitan.sif`)
  couldn't import `torchtitan.train` (0.2.2 on torch 2.10); the rebuilt
  **`dtitan-torch211.sif`** (torch 2.11.0a0 / CUDA 13.2) is built and runs on
  `kempner_rtx` — `kempner_h100`'s CUDA-12.9 driver cannot run this image (see
  the container reference note).
- **Open Decisions resolved** into concrete values (see table), grounded in the
  container, cluster, and the offline model testbed.
- **Assets grounded**: a small vendored debug tokenizer + generated C4 subset
  (`scripts/prepare_c4_subset.py`) for the fast path; the real Llama-3.1
  tokenizer + trainable Llama-3.1-8B from `…/testbed/models/` only for
  `--model.flavor 8B`; larger Llamas / DeepSeek for planning.
- **Config interface grounded** to torchtitan 0.2.2 (`--model.name <model>
  --model.flavor <flavor>` + dotted CLI overrides — there is no `--module`/
  `--config` form; the workshop would register its own config-registry
  variants rather than shipping TOMLs, not yet exercised in Level 1).
- **Launchers reuse** the DTensor workshop's GPU-validated Slurm scaffold
  (`--account=kempner_dev`, `--mem`, `srun --cpu-bind=none`, 8-GPU cap), now on
  `kempner_rtx`.
- **Level 3 scoped honestly** to 8 GPUs: FP8 core; pipeline/expert/context
  parallelism as small-config modules; MXFP8 out; 70B/405B for planning.
- **Cross-linked** to the DTensor workshop (primitives ↔ framework).
