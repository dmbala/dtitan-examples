# TorchTitan Workshop Initial Plan


## Workshop Purpose

This three-level workshop teaches participants how to use TorchTitan as a
PyTorch-native platform for large-scale generative model training. The sequence
should move from a small observable run, to a restartable and profile-driven
multi-parallel run, to a production-style cluster deployment plan.

The workshop should emphasize practical operating skills:

- Reading TorchTitan's training flow and extension points.
- Choosing and overriding registered configs safely.
- Combining PyTorch-native parallelism techniques with minimal model changes.
- Saving, loading, profiling, and debugging distributed jobs.
- Validating correctness before optimizing throughput.

## Current TorchTitan Assumptions

- Pin the exact TorchTitan, PyTorch, TorchAO, CUDA/ROCm, and torchdata versions
  used for the workshop. TorchTitan moves quickly, and advanced features often
  depend on recent PyTorch builds.
- Treat the Python config registry and CLI overrides as the primary
  configuration interface. Avoid describing the workshop as TOML-driven unless
  the local training stack explicitly adds TOML wrappers.
- Use the standard launcher path first, for example
  `MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh`, then introduce
  overrides such as `--training.steps`, `--checkpoint.interval`, and
  `--profiler.enable_memory_snapshot`.
- Use gated assets, such as Llama tokenizers or weights, only after confirming
  access. Beginner labs should have a fallback path that uses a debug model,
  synthetic data, or pre-provisioned assets.
- Keep hardware-specific content modular. FP8, MXFP8, async tensor parallelism,
  and high-MFU tuning should be optional unless the workshop environment has the
  required GPUs and software stack.

## Audience Assumptions

- Participants can train and debug a single-GPU PyTorch model.
- Participants understand tensors, autograd, optimizers, data loaders, and basic
  command-line workflows.
- Participants may not have prior experience with multi-node launchers,
  `DeviceMesh`, Distributed Checkpointing, NCCL debugging, or profiler traces.
- Exercises should run first on one node or through TorchTitan's debug
  communication modes, then scale to multi-node environments in later levels.

## Level 1: Running and Observing TorchTitan

**Title:** TorchTitan Foundations: Configs, FSDP2, Metrics, and First Debugging

**Target audience:** machine learning engineers and researchers who are
comfortable with PyTorch and want a guided path into TorchTitan's training
workflow.

### Goals

Participants should leave with a concrete mental model of how TorchTitan starts
a job, applies the selected model spec and parallelism settings, emits logs and
metrics, and saves enough artifacts to debug a small run.

### Core Topics

- Repository tour:
  - `torchtitan/train.py` as the main training flow.
  - `torchtitan/models/<model>/config_registry.py` for registered configs.
  - model definitions and `parallelize.py` files for model-specific scaling.
  - checkpoint, metrics, profiling, and observability components.
- Config flow:
  - `MODULE` and `CONFIG` selection.
  - CLI overrides with dotted config paths.
  - inspecting resolved configs with `python -m torchtitan.config.manager`.
  - boolean override behavior, including `--no_*` style flags.
- Launch basics:
  - single-node launcher scripts.
  - rank, world size, local rank, and GPU assignment.
  - debug validation with `NGPU=4 … --comm.mode=fake_backend`.
- First parallelism model:
  - 1D FSDP2 as the baseline.
  - what is sharded, what is replicated, and what appears in logs.
- Observability:
  - rank-aware logs and structured logs.
  - loss, memory, throughput, TFLOPs, and MFU metrics.
  - basic CPU/GPU profiler traces.
- Beginner troubleshooting:
  - invalid config names or overrides.
  - GPU count mismatch.
  - missing tokenizer or dataset assets.
  - confusing multi-rank stack traces.

### Hands-On Labs

1. Run a preflight script that checks Python environment, TorchTitan import,
   GPU visibility, write access to `outputs/`, and distributed launcher basics.
2. Inspect a registered config with `torchtitan.config.manager`, then apply two
   CLI overrides and verify the resolved values.
3. Validate a larger intended configuration with `NGPU=4 … --comm.mode=fake_backend`
   before launching a real distributed job.
4. Launch a short 1D FSDP2 run on the available local GPUs.
5. Enable metrics and a profiler trace, then identify where loss, memory, and
   throughput are recorded.
6. Break one config value intentionally, read the failing logs, and document the
   root cause.

### Capstone

Create a small workshop config variant or CLI override set for a short 1D FSDP2
run. The submission must include the launch command, resolved config output,
training logs, a profiler artifact, and a short note explaining whether loss and
throughput look plausible.

### Expected Outcome

Participants can launch a small TorchTitan job, explain the selected config,
read the main observability outputs, and use config inspection plus logs to
debug common startup failures.

## Level 2: Composable Parallelism and Restartable Training

**Title:** FSDP2 + Tensor Parallelism: Checkpointing, Memory, and Bottleneck
Profiling

**Target audience:** practitioners who understand Level 1 concepts and need to
scale beyond a simple 1D run while keeping jobs restartable and diagnosable.

### Goals

Participants should learn to combine parallelism dimensions, validate
correctness against a baseline, save and resume distributed state, and use
profiling evidence to reduce memory or communication bottlenecks.

### Core Topics

- Parallelism configuration:
  - data parallel replicate and shard degrees.
  - tensor parallel degree.
  - context parallelism where the hardware and model path support it.
  - how mesh dimensions map to ranks and GPUs.
- FSDP2 behavior:
  - per-parameter sharding through DTensor.
  - sharded model and optimizer state.
  - memory tradeoffs around resharding.
- Tensor parallelism:
  - column-wise and row-wise linear sharding.
  - sequence-parallel implications where applicable.
  - identifying all-reduce, all-gather, and reduce-scatter in traces.
- Distributed Checkpointing:
  - full training-state checkpoints.
  - model-only final checkpoints.
  - async checkpoint saves.
  - seed checkpoints for reproducible comparisons.
  - DCP resharding across different parallel layouts.
- Memory work:
  - memory snapshots.
  - OOM reproduction.
  - full and selective activation checkpointing.
  - peak memory versus throughput tradeoffs.
- Correctness work:
  - comparing loss curves against a 1D FSDP baseline.
  - keeping global batch size and data-loader behavior consistent.
  - documenting acceptable numeric differences.

### Hands-On Labs

1. Start from the Level 1 run and convert it to a 2D FSDP2 + tensor-parallel
   run using config registry changes or CLI overrides.
2. Draw the resulting rank layout and label data-parallel and tensor-parallel
   groups.
3. Save a DCP checkpoint, resume from it, and verify that the resumed job
   continues from the expected step.
4. Create a seed checkpoint and reuse it for two parallelism configurations.
5. Push the micro-batch or sequence length until the job OOMs, capture a memory
   snapshot, and identify the largest contributors.
6. Enable activation checkpointing and compare peak memory, throughput, and loss
   behavior before and after the change.
7. Inspect a profiler trace and identify the most expensive communication
   region.

### Capstone

Build a restartable 2D TorchTitan run. The final artifact must include the
parallelism settings, rank-layout diagram, DCP save and resume evidence, one
memory optimization, one profiler-based bottleneck diagnosis, and a short loss
comparison against the Level 1 baseline.

### Expected Outcome

Participants can compose FSDP2 with tensor parallelism, recover from a
checkpoint, reason about memory pressure, and use profiler traces instead of
guesswork when choosing the next optimization.

## Level 3: Production Scaling and Extensibility

**Title:** Multi-Node TorchTitan: Pipeline, Context, Expert Parallelism,
Precision, and NCCL Debugging

**Target audience:** senior infrastructure engineers, performance engineers,
and researchers preparing TorchTitan jobs for production GPU clusters.

### Goals

Participants should learn how to plan a larger TorchTitan deployment, make
model and data changes through supported extension points, and debug failures
that only appear at multi-node scale.

### Core Topics

- Cluster execution:
  - Slurm launch scripts.
  - rendezvous configuration.
  - shared filesystem behavior.
  - checkpoint and profiler output placement.
  - cleanup policies for large artifacts.
- Higher-dimensional parallelism:
  - combining FSDP/HSDP, tensor parallelism, pipeline parallelism, and context
    parallelism.
  - expert parallelism and MoE routing where the selected TorchTitan branch and
    hardware stack support it.
  - documenting each mesh dimension's purpose.
- Pipeline-friendly model structure:
  - top-level forwards that partition cleanly.
  - preserving fully qualified names for checkpoint compatibility.
  - seed checkpoints for consistent initialization across layouts.
- Performance features:
  - regional `torch.compile`.
  - async tensor parallelism.
  - Float8 or MXFP8 training when supported by hardware.
  - symmetric memory and overlap-oriented tuning where available.
  - interpreting tokens/sec, TFLOPs, and MFU.
- Extensibility:
  - registering a model through `ModelSpec`.
  - adding config registry entries.
  - adding or swapping Hugging Face, SFT, interleaved, or multimodal data
    loaders.
- Advanced debugging:
  - NCCL hangs, timeouts, and mismatched collectives.
  - Flight Recorder dumps.
  - deterministic debug mode and seeds.
  - comparing failing ranks and reconstructing the last successful training
    phase.

### Hands-On Labs

1. Adapt `multinode_trainer.slurm` or an equivalent launcher for the workshop
   cluster and validate the job with a tiny run.
2. Extend a 2D layout into a 3D or 4D layout and document the role of every mesh
   dimension.
3. Add pipeline or context parallelism, then compare loss behavior with a
   simpler known-good baseline.
4. Enable one performance feature, such as `torch.compile`, async tensor
   parallelism, or Float8, and measure the impact on throughput and memory.
5. Register a small custom config, dataset, or model extension without forking
   the main training loop.
6. Inject a controlled timeout or collective mismatch, then use Flight Recorder
   and rank logs to identify the failing stage or collective.

### Capstone

Produce a production-style TorchTitan deployment package for a larger training
scenario. It should include a Slurm launch script, resolved config, topology
diagram, checkpoint policy, metrics/profiling plan, one measured performance
optimization, and a postmortem-style note for an injected distributed failure.

### Expected Outcome

Participants can connect model architecture, parallel layout, checkpointing,
profiling, cluster scheduling, and NCCL diagnostics into a coherent
production-style training workflow.

## Cross-Cutting Workshop Design

### Format

- Run each level as a separate half-day or full-day module.
- Start every level from a known-good runnable baseline.
- Pair every concept with a concrete lab and a visible artifact.
- Keep one running model family across the workshop where possible.
- Separate correctness, reliability, and performance milestones.
- Include one intentional failure in every level.

### Required Artifacts Per Level

- Launch command or script.
- Resolved config output.
- Logs from at least rank 0 and one non-zero rank.
- Metrics output or dashboard screenshot reference.
- Profiler, memory, or checkpoint artifact as appropriate.
- A short diagnosis written by the participant.

### Environment Requirements

- Pinned container or environment file.
- Preflight validation script.
- Local output directories for logs, checkpoints, memory snapshots, and profiler
  traces.
- Optional TensorBoard and Weights & Biases setup.
- NCCL and launcher environment variables documented for the cluster.
- Pre-downloaded tokenizer or an approved fallback asset path.
- Storage budget for DCP checkpoints and profiler traces.
- Reference commands for single-node and multi-node execution.

### Assessment Rubric

- Correctness: the run starts, trains, and preserves expected loss behavior.
- Configuration: the participant can explain the effective config and overrides.
- Reliability: checkpoint save and resume work as expected.
- Observability: logs, metrics, and traces are captured and interpreted.
- Debugging: the participant identifies the cause of an injected failure.
- Performance: optimization claims are backed by measured throughput, memory, or
  MFU evidence.

## Recommendations

1. Keep Level 1 independent of gated model access.
   Use a debug model or pre-provisioned assets so the first workshop hour is not
   spent resolving Hugging Face permissions.

2. Make config inspection a habit.
   Every lab should start by printing the resolved config before launching an
   expensive distributed job.

3. Provide known-good artifacts.
   Include reference logs, resolved configs, profiler traces, memory snapshots,
   and DCP checkpoints so participants can compare outputs when hardware differs.

4. Teach failure modes deliberately.
   Use config mistakes, OOMs, checkpoint resume issues, and controlled collective
   failures as first-class exercises.

5. Treat advanced hardware features as optional modules.
   Float8, MXFP8, async TP, symmetric memory, and high-MFU tuning are valuable,
   but they should not block the core workshop if hardware support is missing.

6. Include instructor notes.
   For each lab, document expected runtime, common error messages, likely root
   causes, and the fastest recovery path.

7. End every level with an operational checklist.
   The checklist should cover config, launch arguments, rank layout, checkpoint
   path, profiler settings, expected logs, and cleanup.

## Open Decisions

- Exact TorchTitan commit or release.
- PyTorch and TorchAO versions.
- GPU type, GPU count, and whether multi-node hardware is guaranteed.
- Whether the workshop uses Llama assets, debug models, synthetic data, or a
  custom model family.
- Whether W&B is allowed, or TensorBoard/local artifacts are required instead.
- Checkpoint retention policy and storage limits.
- Which Level 3 features are required versus optional for the target cluster.

## Reference Anchors

- TorchTitan README: https://github.com/pytorch/torchtitan
- Debugging and profiling: https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md
- Checkpointing: https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md
- FSDP2 notes: https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md
- Metrics: https://github.com/pytorch/torchtitan/blob/main/docs/metrics.md
- Datasets: https://github.com/pytorch/torchtitan/blob/main/docs/datasets.md
- Extensibility: https://github.com/pytorch/torchtitan/blob/main/docs/extension.md
- Loss-convergence guidance: https://github.com/pytorch/torchtitan/blob/main/docs/converging.md
