# Level 1: TorchTitan Foundations — Configs, FSDP2, Metrics, and First Debugging

**Duration:** ~half day  
**Hardware:** 1 node, 4 GPUs (`kempner_rtx`)  
**Launch:** `slurm/launch_1node.sbatch` and CLI commands via Singularity

**Goals:** Develop a concrete mental model of how TorchTitan starts a job, applies the selected model spec and parallelism, emits logs/metrics, and saves enough to debug a small run.

**Assets:** The `llama3` **`debugmodel`** flavor hardcodes `vocab_size=2048`, and torchtitan 0.2.2 does not resize the embedding from the tokenizer — so debug runs use a small vendored tokenizer (vocab 2016) at `titan/assets/test_tokenizer/`, passed as `--model.hf_assets_path=assets/test_tokenizer`. The real Llama-3.1 tokenizer (`$MODELS/Llama-3.1-8B-Instruct`, vocab 128256) is reserved for the real-model taste with `--model.flavor 8B` — pairing it with `debugmodel` overflows the embedding (`vectorized_gather_kernel: index out of bounds`). `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`.

Training data is a generated local C4 subset — run once before the training labs:
```bash
python scripts/prepare_c4_subset.py
```
This writes `titan/assets/c4_subset/train.jsonl` (3000 real C4 docs from the Kempner testbed; git-ignored). The training labs then pass `--training.dataset=c4_test --training.dataset_path=assets/c4_subset`.

---

## Lab 1: Preflight Checks

**Goal:** Verify that the environment is ready for TorchTitan runs.

**Command:**
```bash
python preflight.py
```

**Expected artifact:** A pass/fail line for each of 6 checks printed to stdout.

**Success criterion:** All 6 checks pass (run inside a `kempner_rtx` GPU allocation, after the one-time `prepare_c4_subset.py` setup above):
- `import torchtitan` succeeds
- `import torchtitan.train` succeeds (verifies torch/torchtitan compatibility)
- Tokenizer path readable (`assets/test_tokenizer`)
- C4 subset present (`assets/c4_subset/train.jsonl`) — fails with a "run `prepare_c4_subset.py`" hint until generated
- `outputs/` is writable
- At least one GPU visible (`gpu visible: device_count=4` with a `--gpus-per-node=4` allocation)

Config resolution, NCCL collective checks, and the fake-backend dry-run are **not** part of preflight — they're exercised in Labs 2 and 3 below.

---

## Lab 2: Inspect a Config and Apply Overrides

**Goal:** Understand how TorchTitan's config system works and verify that dotted CLI overrides apply correctly.

**Command:**
```bash
python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--model.name','llama3','--model.flavor','debugmodel','--training.steps','7']).training.steps)"
```

**Expected artifact:** A single integer printed to stdout: `7`.

**Success criterion:** The command prints `7`, demonstrating that:
- The config system resolves the `llama3` model's `debugmodel` flavor
- The dotted override `--training.steps 7` applies to the parsed config
- The resolved value matches the override

**Tip:** Try resolving the full config instead of a single field — e.g. print `.model.flavor`, `.training.seq_len`, `.training.steps`, and `.parallelism.data_parallel_shard_degree` — to see how all the pieces you'll use in Labs 4–6 come together before an expensive launch.

---

## Lab 3: Fake-Backend Dry-Run

**Goal:** Verify that TorchTitan's config and model initialization work correctly without requiring real multi-GPU communication.

**Command:**
```bash
NGPU=4 torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --model.name llama3 --model.flavor debugmodel \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --comm.mode=fake_backend
```
(Run inside `singularity exec --nv <image>` or via the launcher's `torchrun` invocation. A bare `NGPU=4 python -m torchtitan.train … --comm.mode=fake_backend` — no `torchrun` — fails with `KeyError: 'LOCAL_RANK'`; torchtitan reads `LOCAL_RANK` for the device.)

**Expected artifact:** Console log showing model build, `Applied FSDP to the model`, and a couple of training steps completing — all in **one process**, no real multi-GPU communication required.

**Success criterion:** The job launches, resolves the `llama3` `debugmodel` configuration, initializes the model, and completes at least one step without errors, ending in `Training completed`. Check the log for:
- Model name and parameter count logged
- `Applied FSDP to the model`
- Training loop begins and completes at least one step
- Clean exit (`Training completed`, no OOM, no collective communication errors)

**Tip:** `NGPU` sets the *simulated* world size and `--comm.mode=fake_backend` fakes the collectives, so this dry-run is fast and safe for verifying config syntax and model compatibility before submitting an expensive multi-GPU job.

---

## Lab 4: 1D FSDP2 Single-GPU-Per-Rank Run

**Goal:** Execute a small distributed training run on one node with 4 GPUs, using FSDP2 (Fully Sharded Data Parallel) for the first time.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4
```
(The launcher already sets `--partition=kempner_rtx`, the `dtitan-torch211.sif` image, `--model.name llama3 --model.flavor debugmodel`, and the writable HF cache env. Run `python scripts/prepare_c4_subset.py` once first if you haven't already.)

**Expected artifact:** Training log in `outputs/` showing per-step loss and metrics.

**Success criterion:** The job completes successfully and demonstrates:
- Model builds: `Model llama3 debugmodel size: 6,163,712 total parameters`
- All 4 ranks are visible in the logs, sharded 4-way FSDP2 (`dp_shard=4`)
- Loss decreases over the 20 training steps: step 1 ≈ **8.12** → step 10 ≈ **6.30** → step 20 ≈ **3.55**
- Per-step metrics appear: `grad_norm`, `memory ≈ 0.32GiB`, `tps ≈ 180k`, `tflops ≈ 8`, `mfu ≈ 2.5%`
- The run ends with `Training completed`
- No collective communication errors

**Note:** A `WARNING - Peak flops undefined for … Blackwell, fallback to A100` line appears — MFU is computed against the A100 peak-FLOPs table (Blackwell isn't in it yet), so treat the MFU% as indicative only, not a true Blackwell utilization number.

**Tip:** Inspect the job log with `cat outputs/<jobname>-<jobid>.out` (the launcher's `--output=outputs/%x-%j.out` places it there) to see the full trace. If loss does not decrease, check that the model is initialized correctly and the learning rate is reasonable.

---

## Lab 5: Metrics and Profiler Integration

**Goal:** Enable profiling to observe training dynamics: loss, memory usage, throughput (tokens/sec), and model FLOP utilization (MFU).

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_profiling
```

**Expected artifact:**
- Training log in `outputs/` (same as Lab 4)
- Log line `Dumping profiler traces at step 10`
- Profiler traces: `outputs/profile_traces/iteration_10/rank{0,1,2,3}_trace.json` (~13 MB total)

**Success criterion:** The profiler runs without error and you can locate in the log:
- **Loss per step** — same downward trend as Lab 4
- **Peak memory usage** — `memory ≈ 0.32GiB`
- **Throughput** — `tps` (tokens/sec)
- **Model FLOPs Utilization (MFU)** — `mfu` percentage (indicative only — see the Blackwell/A100 caveat in Lab 4)

**Tip:** Search the log for `Dumping profiler traces` to confirm the trace was written, and for `loss`/`memory`/`tps`/`mfu` to find the metrics. Compare throughput and memory to Lab 4 (without profiling) to understand the profiler's overhead.

---

## Lab 6: Failure-Driven Debugging

**Goal:** Learn to read TorchTitan errors by intentionally breaking a configuration and diagnosing the root cause.

**Task:** Submit a job with an invalid configuration override. For example:
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=20 --parallelism.tensor_parallel_degree=3
```

(Note: 3 is invalid on 4 GPUs — tensor parallelism degree must divide the number of GPUs evenly.)

**Expected artifact:** Error output in the log file(s).

**Success criterion:** You identify and document:
1. **The error message** — a clean, readable assertion:
   ```
   AssertionError: Invalid parallel dims: dp_replicate(1) * dp_shard(4) * cp(1) * tp(3) * pp(1) != WORLD_SIZE(4)
   ```
2. **Root cause** — the product of all parallelism degrees must equal the world size; `tensor_parallel_degree=3` does not divide 4 GPUs
3. **The fix** — e.g., `--parallelism.tensor_parallel_degree=2` or `--parallelism.tensor_parallel_degree=4` (with a matching `data_parallel_shard_degree`)

Then, re-run the job with the corrected override and verify it succeeds (as in Lab 4).

**Tip:** Always check the `--help` output or the TorchTitan config dataclasses to understand the constraints on each parameter. Errors often appear in rank-0 output before job termination.

---

## Capstone: Full Observable Run with Artifacts

**Goal:** Execute a complete 1D FSDP2 training run and submit a capstone package demonstrating your understanding of configuration, metrics, and diagnostics.

**Task:** Design and run a short 1D FSDP2 training scenario using the debug model. You have flexibility in the number of steps, batch size, and sequence length, but aim for a run that completes in a reasonable time (a few minutes to ~10 minutes) and collects full diagnostics.

**Example command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=50 --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_profiling
```

**Capstone deliverables:**

1. **Launch command** — Exact `sbatch` command you ran (copy-paste from your bash history or script).

2. **Resolved configuration** — Output of the config-inspection step (Lab 2 pattern) applied to your overrides:
   ```bash
   python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--model.name','llama3','--model.flavor','debugmodel','--training.steps','50','--profiling.enable_profiling']).training)"
   ```
   (Include the full resolved config dict or dataclass output.)

3. **Training log** — Job log from `outputs/` (e.g., `outputs/<jobname>-<jobid>.out`), showing at least:
   - Model name and parameter count
   - Per-step loss and memory metrics
   - Final step and completion message

4. **Profiler artifact** — The saved trace files (`outputs/profile_traces/iteration_<freq>/rank{0..3}_trace.json`), demonstrating that profiling ran.

5. **Plausibility note** — A 2–3 sentence written summary:
   - Is the loss behavior reasonable? (decreasing, stable, or diverging?)
   - What throughput (tokens/sec) and memory (GB) did you observe?
   - Are these numbers plausible for a 4-GPU run on this model and batch size? (Remember MFU is computed against the A100 FLOPs table on this Blackwell hardware — treat it as indicative, not exact.)

**Success criterion:** All five deliverables are collected, the run completes without error, and you can write a coherent plausibility note on the observed loss and throughput.

---

## Reference Artifacts

**Planned:** known-good reference outputs (logs, profiler traces, resolved configs, memory snapshots, and seed checkpoints) will be provided under `outputs/reference/` for comparison if your environment differs slightly. These do not exist yet; adding them will require an `!outputs/reference/` entry in `.gitignore` (currently `outputs/*` is ignored wholesale).

---

## Common Errors and Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `ImportError: cannot import torchtitan.train` | Wrong image (torch 2.10) | Use `dtitan-torch211.sif` (torch 2.11) — the launchers already hardcode it |
| `RuntimeError: The NVIDIA driver on your system is too old (found version 12090)` | Job landed on `kempner_h100` (CUDA 12.9 driver) instead of `kempner_rtx` (CUDA 13.2) | Submit with `--partition=kempner_rtx` |
| `torch.cuda.device_count() != 4` | Not running in the container or on the right node | Check `nvidia-smi` and Singularity bind mounts in the launch script |
| `Unrecognized options: --module, --config` | Old (nonexistent) config interface | Use `--model.name llama3 --model.flavor debugmodel` |
| `Unrecognized options: --profiler` (config section typo) | Wrong section name — the section is `profiling`, not `profiler` | Use `--profiling.enable_profiling` |
| `KeyError: 'LOCAL_RANK'` in the fake-backend dry-run | Ran `python -m torchtitan.train` directly instead of via `torchrun` | Launch with `torchrun --standalone --nproc_per_node=1 -m torchtitan.train …` |
| `vectorized_gather_kernel: index out of bounds` | Real Llama-3.1 tokenizer (vocab 128256) paired with `debugmodel` (vocab_size=2048) | Use `--model.hf_assets_path=assets/test_tokenizer` for debug runs; the real tokenizer needs `--model.flavor 8B` |
| `c4 subset` preflight check fails / dataset load error | `assets/c4_subset/train.jsonl` not generated yet | Run `python scripts/prepare_c4_subset.py` once |
| `tensor_parallel_degree=3` fails on 4 GPUs | Parallelism degree does not divide GPU count | Ensure all parallelism degrees divide the number of GPUs (4 in this case) |
| Loss is NaN or diverging | Bad learning rate or numerical issues | Start with the default `debugmodel` flavor and validate with a dry-run (Lab 3) first |
| Job hangs on `all_reduce` | NCCL communication issue; misconfigured ranks | Check rank assignment in the Slurm launch script and ensure all GPUs are visible |

---

## Next Steps

After completing all six labs and the capstone:
- Review your capstone artifacts and reflect on what you learned about TorchTitan's config system and FSDP2 mechanics.
- Compare your observed loss curve and throughput to the reference artifacts.
- Proceed to **Level 2** to explore tensor parallelism, checkpointing, and memory optimization.
