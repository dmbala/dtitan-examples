# Level 1: TorchTitan Foundations — Configs, FSDP2, Metrics, and First Debugging

**Duration:** ~half day  
**Hardware:** 1 node, 4 GPUs (`kempner_h100`)  
**Launch:** `slurm/launch_1node.sbatch` and CLI commands via Singularity

**Goals:** Develop a concrete mental model of how TorchTitan starts a job, applies the selected model spec and parallelism, emits logs/metrics, and saves enough to debug a small run.

**Assets:** The built-in `llama3_debugmodel` config needs both a tokenizer and a dataset. The workshop passes `--hf_assets_path=$MODELS/Llama-3.1-8B-Instruct` (the real Llama-3.1 tokenizer, used offline) plus a small local dataset — the debug config's default `dataset=c4_test` is not included in the pip wheel, so the exact dataset wiring is confirmed on the rebuilt container. `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`.

---

## Lab 1: Preflight Checks

**Goal:** Verify that the environment is ready for TorchTitan runs.

**Command:**
```bash
python preflight.py
```

**Expected artifact:** A pass/fail line for each check printed to stdout.

**Success criterion:** All 5 checks pass:
- `import torchtitan` succeeds
- `import torchtitan.train` succeeds (verifies torch/torchtitan compatibility)
- Tokenizer path readable (`$MODELS/Llama-3.1-8B-Instruct`)
- `outputs/` is writable
- At least one GPU visible (`torch.cuda.device_count() > 0`)

Config resolution, NCCL collective checks, and the fake-backend dry-run are **not** part of preflight — they're exercised in Labs 2 and 3 below.

---

## Lab 2: Inspect a Config and Apply Overrides

**Goal:** Understand how TorchTitan's config system works and verify that dotted CLI overrides apply correctly.

**Runs require the rebuilt container (see container/dtitan.def).**

**Command:**
```bash
python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--module','llama3','--config','llama3_debugmodel','--training.steps','7']).training.steps)"
```

**Expected artifact:** A single integer printed to stdout: `7`.

**Success criterion:** The command prints `7`, demonstrating that:
- The config system resolves the `llama3_debugmodel` config
- The dotted override `--training.steps 7` applies to the parsed config
- The resolved value matches the override

**Note:** On the current container (torch 2.10), this command will error with `ImportError: Cannot import config_registry for module 'llama3'` (importing the llama3 config requires torch 2.11's `activate_flash_attention_impl`). This error documents the container rebuild gate; on the rebuilt container (torch ≥ 2.11), the command resolves to `7` as expected.

---

## Lab 3: Fake-Backend Dry-Run

**Goal:** Verify that TorchTitan's config and model initialization work correctly without requiring GPUs.

**Runs require the rebuilt container (see container/dtitan.def).**

**Command:**
```bash
NGPU=4 python -m torchtitan.train --module llama3 --config llama3_debugmodel --comm.mode=fake_backend
```

**Expected artifact:** Console log showing model loading, rank 0 info, and training loop initialization, ending cleanly without GPU allocation.

**Success criterion:** The job launches, resolves the `llama3_debugmodel` configuration, initializes the model, and completes the first few training steps without errors. Check the log for:
- Model name and parameter count logged
- Rank 0 output (no GPU allocation errors)
- Training loop begins and completes at least one step
- Clean exit (no OOM, no collective communication errors)

**Tip:** This dry-run is fast and safe for verifying config syntax and model compatibility before submitting an expensive multi-GPU job.

---

## Lab 4: 1D FSDP2 Single-GPU-Per-Rank Run

**Goal:** Execute a small distributed training run on one node with 4 GPUs, using FSDP2 (Fully Sharded Data Parallel) for the first time.

**Runs require the rebuilt container (see container/dtitan.def).**

**Command:**
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct
```

(where `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`)

**Expected artifact:** Training log in `outputs/` showing per-step loss and metrics.

**Success criterion:** The job completes successfully and demonstrates:
- All 4 ranks are visible in the logs (look for rank-aware log lines)
- Loss decreases over the 20 training steps (training is not diverging)
- No collective communication errors
- Metrics (loss, memory, throughput) appear in the log — the debug config does not enable checkpointing by default, so no checkpoint files are produced unless `--checkpoint.enable_checkpoint` is set

**Tip:** Inspect the job log with `cat outputs/<jobname>-<jobid>.out` (the launcher's `--output` places it there) to see the full trace. If loss does not decrease, check that the model is initialized correctly and the learning rate is reasonable.

---

## Lab 5: Metrics and Profiler Integration

**Goal:** Enable profiling to observe training dynamics: loss, memory usage, throughput (tokens/sec), and model FLOP utilization (MFU).

**Runs require the rebuilt container (see container/dtitan.def).**

**Command:**
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --profiler.enable_profiling --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct
```

**Expected artifact:**
- Training log in `outputs/` (same as Lab 4)
- Profiler trace file (typically `outputs/trace_*.json` or similar, depending on torch version)
- Metrics output in the log file

**Success criterion:** The profiler runs without error and you can locate in the log:
- **Loss per step** — values should decrease or stabilize
- **Peak memory usage** — GPU memory footprint reported
- **Throughput** — tokens processed per second
- **Model FLOPs Utilization (MFU)** — percentage of peak theoretical FLOPs achieved

**Tip:** Search the log for keywords like `loss`, `memory`, `tokens/sec`, or `MFU` to find these metrics. Compare the throughput and memory to Lab 4 (without profiling) to understand the profiler's overhead.

---

## Lab 6: Failure-Driven Debugging

**Goal:** Learn to read TorchTitan errors by intentionally breaking a configuration and diagnosing the root cause.

**Runs require the rebuilt container (see container/dtitan.def).**

**Task:** Submit a job with an invalid configuration override. For example:
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --parallelism.tensor_parallel_degree=3 --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct
```

(Note: 3 is invalid on 4 GPUs — tensor parallelism degree must divide the number of GPUs evenly.)

**Expected artifact:** Error output in the log file(s).

**Success criterion:** You identify and document:
1. **The error message** — what does TorchTitan / PyTorch report?
2. **Root cause** — e.g., "tensor_parallel_degree=3 does not divide 4 GPUs"
3. **The fix** — e.g., "use `--parallelism.tensor_parallel_degree=2` or `--parallelism.tensor_parallel_degree=4`"

Then, re-run the job with the corrected override and verify it succeeds (as in Lab 4).

**Tip:** Always check the `--help` output or the TorchTitan config dataclasses to understand the constraints on each parameter. Errors often appear in rank-0 output before job termination.

---

## Capstone: Full Observable Run with Artifacts

**Goal:** Execute a complete 1D FSDP2 training run and submit a capstone package demonstrating your understanding of configuration, metrics, and diagnostics.

**Runs require the rebuilt container (see container/dtitan.def).**

**Task:** Design and run a short 1D FSDP2 training scenario using the TorchTitan debug config or a variant. You have flexibility in the number of steps, batch size, and sequence length, but aim for a run that completes in a reasonable time (a few minutes to ~10 minutes) and collects full diagnostics.

**Example command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --training.steps=50 \
  --profiler.enable_profiling \
  --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct
```

**Capstone deliverables:**

1. **Launch command** — Exact `sbatch` command you ran (copy-paste from your bash history or script).

2. **Resolved configuration** — Output of the config-inspection step (Lab 2 pattern) applied to your overrides:
   ```bash
   python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--module','llama3','--config','llama3_debugmodel','--training.steps','50','--profiler.enable_profiling']).training)"
   ```
   (Include the full resolved config dict or dataclass output.)

3. **Training log** — Job log from `outputs/` (e.g., `outputs/<jobname>-<jobid>.out`), showing at least:
   - Model name and parameter count
   - Per-step loss and memory metrics
   - Final step and completion message

4. **Profiler artifact** — The saved trace file (e.g., `trace_*.json`), demonstrating that profiling ran.

5. **Plausibility note** — A 2–3 sentence written summary:
   - Is the loss behavior reasonable? (decreasing, stable, or diverging?)
   - What throughput (tokens/sec) and memory (GB) did you observe?
   - Are these numbers plausible for a 4-GPU run on this model and batch size?

**Success criterion:** All five deliverables are collected, the run completes without error, and you can write a coherent plausibility note on the observed loss and throughput.

---

## Reference Artifacts

**Planned:** known-good reference outputs (logs, profiler traces, resolved configs, memory snapshots, and seed checkpoints) will be provided under `outputs/reference/` for comparison if your environment differs slightly. These do not exist yet; adding them will require an `!outputs/reference/` entry in `.gitignore` (currently `outputs/*` is ignored wholesale).

---

## Common Errors and Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `ImportError: Cannot import torchtitan` | Torch/TorchTitan mismatch (torch 2.10 instead of ≥ 2.11) | Ensure the rebuilt container (see container/dtitan.def) is used |
| `torch.cuda.device_count() != 4` | Not running in the container or on the right node | Check `nvidia-smi` and Singularity bind mounts in the launch script |
| `ConfigManager` cannot parse `llama3_debugmodel` | Missing or wrong config name | Use `--module llama3 --config llama3_debugmodel` exactly as written |
| `tensor_parallel_degree=3` fails on 4 GPUs | Parallelism degree does not divide GPU count | Ensure all parallelism degrees divide the number of GPUs (4 in this case) |
| Loss is NaN or diverging | Bad learning rate or numerical issues | Start with the default `llama3_debugmodel` config and validate with a dry-run (Lab 3) first |
| Job hangs on `all_reduce` | NCCL communication issue; misconfigured ranks | Check rank assignment in the Slurm launch script and ensure all GPUs are visible |

---

## Next Steps

After completing all six labs and the capstone:
- Review your capstone artifacts and reflect on what you learned about TorchTitan's config system and FSDP2 mechanics.
- Compare your observed loss curve and throughput to the reference artifacts.
- Proceed to **Level 2** to explore tensor parallelism, checkpointing, and memory optimization.
