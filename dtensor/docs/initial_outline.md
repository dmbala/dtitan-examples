# DTensor Workshop Initial Outline

## Workshop Purpose

This three-level workshop teaches modern PyTorch distributed training through
`DeviceMesh`, `DTensor`, Distributed Checkpointing (DCP), profiling, and
production-oriented debugging. Each level should move participants from a
working script to an observable, restartable, and diagnosable distributed
training workflow.

## Audience Assumptions

- Participants can write and train a single-GPU PyTorch model.
- Participants have basic familiarity with tensors, autograd, optimizers, and
  training loops.
- Participants do not need prior experience with multi-node training, NCCL, or
  distributed checkpoint formats.
- Exercises should run first on a local multi-GPU node or simulated process
  group, then scale to a cluster environment in later levels.

## Level 1: Foundations of Distributed PyTorch

**Title:** Foundations of Distributed PyTorch: DeviceMesh, DTensor, and
Diagnostics

**Target audience:** Machine learning engineers and researchers who are
comfortable with single-GPU PyTorch and new to distributed training,
multi-process execution, and the SPMD programming model.

### Goals

Participants should leave with a correct mental model for how the same Python
program runs across ranks and how DTensor placements describe distributed tensor
layout.

### Core Topics

- SPMD execution: ranks, world size, local rank, process groups, and launchers.
- `DeviceMesh`: mapping logical mesh dimensions onto available devices.
- DTensor placements: `Shard(dim)`, `Replicate()`, and `Partial()`.
- Basic correctness checks: local shards, global shape, redistribution, and
  rank-aware logging.
- Introductory profiling with `torch.profiler`.
- Beginner troubleshooting: mismatched tensor shapes, missing collectives,
  launcher errors, and confusing multi-rank stack traces.

### Hands-On Labs

1. Launch the same script across multiple ranks and print rank-aware state.
2. Build a 1D `DeviceMesh` and shard a large tensor with DTensor.
3. Redistribute between sharded and replicated layouts.
4. Trigger and debug a shape mismatch across ranks.
5. Capture a profiler trace and compare compute time with communication time.

### Capstone

Write a script that creates a 1D `DeviceMesh`, shards a large synthetic tensor,
executes several mathematical operations, validates the global result, and
exports a profiler trace showing compute and communication activity.

### Expected Outcome

Participants can explain what each rank owns, verify that a distributed tensor
operation is correct, and collect a basic trace when performance or correctness
looks suspicious.

## Level 2: Two-Dimensional Parallelism and Checkpointing

**Title:** 2D Parallelism, Bottleneck Profiling, and Distributed Checkpointing

**Target audience:** practitioners who understand the basics of distributed
execution and want to combine data parallelism, tensor parallelism, memory
profiling, and restartable checkpointing.

### Goals

Participants should learn how to structure a realistic 2D parallel training
loop and make it resilient enough to save, load, profile, and debug.

### Core Topics

- 2D `DeviceMesh` construction with named mesh dimensions such as `dp` and
  `tp`.
- Tensor-parallel linear layers and Megatron-style column and row sharding.
- Data-parallel training semantics across the data-parallel mesh dimension.
- Distributed Checkpointing with sharded state dictionaries.
- Asynchronous checkpoint saves and restart validation.
- Memory profiling: allocation timelines, peak memory, fragmentation symptoms,
  and OOM diagnosis.
- Activation checkpointing tradeoffs and selective activation checkpointing.
- Communication bottlenecks: all-reduce, all-gather, reduce-scatter, and idle
  GPU time in profiler traces.

### Hands-On Labs

1. Build a 2D mesh and assign separate data-parallel and tensor-parallel roles.
2. Implement tensor-parallel MLP or Transformer block components.
3. Train with synthetic data and validate loss parity against a single-device
   baseline at small scale.
4. Save and restore model, optimizer, and scheduler state with DCP.
5. Intentionally trigger an OOM, inspect a memory snapshot, and reduce peak
   memory with activation checkpointing.
6. Use profiler traces to locate expensive collectives and idle gaps.

### Capstone

Build a 2D-parallel training loop for a small Transformer block. The final
submission should include a working DCP save/load path, a profiler trace, a
memory optimization change, and a short diagnosis explaining the dominant
communication or memory bottleneck.

### Expected Outcome

Participants can build a small but realistic distributed training loop, recover
from checkpoints, reason about memory pressure, and use profiler evidence
instead of guesswork when optimizing.

## Level 3: Advanced Scaling and Production Debugging

**Title:** Supercomputer Scaling: FSDP2, MoE, Async Optimizations, and NCCL
Debugging

**Target audience:** senior infrastructure engineers, performance engineers,
and AI researchers who need to scale training across high-end multi-node GPU
clusters.

### Goals

Participants should learn how to reason about complex parallel layouts,
overlap communication with computation, and debug failures that only appear at
larger scale.

### Core Topics

- 3D and 4D meshes combining data, tensor, expert, and fully sharded data
  parallel dimensions.
- FSDP2 integration for parameter, gradient, and optimizer state sharding.
- Mixture-of-Experts routing, expert parallelism, and token dispatch overhead.
- Communication and compute overlap: backward prefetching, asynchronous tensor
  parallelism, and reduce-scatter/all-gather scheduling.
- FP8 communication and compute considerations on Hopper-class hardware.
- Regional `torch.compile` usage for stable subgraphs.
- Advanced NCCL debugging: hangs, timeouts, mismatched collectives, environment
  variables, and Flight Recorder traces.
- Fault-aware data loading and deterministic restart on shared filesystems.

### Hands-On Labs

1. Extend a 2D mesh into a 3D or 4D topology and document each dimension's role.
2. Wrap model regions with FSDP2 and validate state dict compatibility with DCP.
3. Add a small MoE layer and inspect token routing imbalance.
4. Compare baseline execution with communication-overlap optimizations.
5. Enable regional compilation on a stable model region and measure the impact.
6. Inject a controlled collective mismatch or artificial network delay, then
   isolate the failing rank with NCCL and Flight Recorder diagnostics.

### Capstone

Architect an optimized training script for a small MoE model on a simulated or
real multi-node cluster. The final result should demonstrate a documented mesh
layout, FSDP2 sharding, DCP recovery, at least one overlap or precision
optimization, and a postmortem-style debugging note for an injected NCCL issue.

### Expected Outcome

Participants can connect model architecture, parallel layout, checkpointing,
profiling, and cluster diagnostics into a production-style distributed training
workflow.

## Cross-Cutting Workshop Design

### Suggested Format

- Run each level as a separate half-day or full-day module.
- Start every level with a minimal runnable baseline before introducing
  optimization or failure modes.
- Keep lectures short and pair each concept with a concrete lab.
- Use the same toy model family across all levels so participants focus on
  distributed systems concepts rather than model-specific details.
- Require each capstone to include a correctness check, a profiling artifact,
  and a short written diagnosis.

### Environment Requirements

- PyTorch version pinned and tested before the workshop.
- Reproducible environment file or container image.
- Launcher scripts for single-node and multi-node execution.
- Small synthetic datasets for deterministic exercises.
- Optional real dataset exercise only after the distributed mechanics are
  stable.
- Pre-created directories for checkpoints, profiler traces, and memory
  snapshots.

## Recommendations to Improve the Workshop

1. Define a single running example.
   Use one compact Transformer or MLP-based model throughout all three levels.
   This reduces cognitive load and makes each level feel like an extension of
   the previous one.

2. Add explicit success criteria for every lab.
   Each exercise should state the expected command, expected artifact, and
   expected validation result. For example: "loss matches the single-GPU
   baseline within tolerance" or "checkpoint reload resumes at the same step."

3. Include failure-driven exercises earlier.
   Distributed training is difficult to debug, so each level should include at
   least one intentional failure: shape mismatch in Level 1, OOM in Level 2,
   and collective hang in Level 3.

4. Provide reference traces and checkpoints.
   Give participants known-good profiler traces, memory snapshots, and DCP
   checkpoints so they can compare their output against a baseline when local
   hardware behaves differently.

5. Separate correctness, reliability, and performance milestones.
   Ask participants to make the code correct first, restartable second, and
   fast third. This mirrors how distributed training systems are usually built
   in practice.

6. Add a preflight validation script.
   Before the workshop starts, participants should run a script that verifies
   GPU visibility, NCCL setup, PyTorch version, distributed launch, write access
   to checkpoint storage, and profiler output paths.

7. Keep advanced hardware-specific content modular.
   FP8, Hopper-specific behavior, and multi-node NCCL debugging are valuable
   but hardware dependent. Treat them as optional advanced sections when the
   workshop environment cannot guarantee the right GPUs.

8. End each level with a short operational checklist.
   Summarize what participants should check before running distributed jobs:
   launch arguments, mesh shape, placement choices, checkpoint location,
   profiler settings, environment variables, and expected logs.

9. Add instructor notes for timing and common pitfalls.
   Include estimated time per lab, likely error messages, and recovery steps so
   instructors can keep the room moving.

10. Create an assessment path.
    Use small rubric items such as correctness validation, checkpoint recovery,
    profiler interpretation, and debugging explanation. This makes the workshop
    easier to evaluate and improve after each run.
