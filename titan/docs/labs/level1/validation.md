# Level 1 GPU Validation Checklist

**Purpose:** Verify that Level 1 labs run correctly on the rebuilt container (torch ≥ 2.11) and GPU hardware (Kempner H100).

**Prerequisites:**
- Container: `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`
- Hardware: 1 node, 4 GPUs (`kempner_h100` partition)
- Account: `kempner_dev` (8-GPU/user cap)
- Offline assets: `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models` (includes Llama-3.1-8B-Instruct)

---

## Item 1: Preflight All-Pass

**Goal:** Verify that the environment is correctly configured for TorchTitan execution.

**Command:** (must run inside a GPU allocation — e.g. `srun -p kempner_h100 -A kempner_dev --gres=gpu:4 --pty bash`, or via the 1-node launcher; run from a login node and `device_count` will be `0`)
```bash
singularity exec --nv "$IMAGE" bash -lc 'cd titan && python preflight.py'
```
(where `$IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`)

**Expected Outcome:**
All checks print `[PASS]`:
- `[PASS] torchtitan import: ok`
- `[PASS] torchtitan.train import: ok`
- `[PASS] tokenizer:/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-8B-Instruct: ok`
- `[PASS] writable:outputs: ok`
- `[PASS] gpu visible: device_count=4`

(If `torchtitan.train import` fails, the container is still torch 2.10; the actual error is `ImportError: cannot import name '_context_parallel_shard' from torch.distributed.tensor.experimental._attention` — rebuild per `container/dtitan.def` is required.)

**Success Criterion:** All 5 checks pass. The `torchtitan.train import` check confirms torch ≥ 2.11 and config availability. GPU visibility (`device_count=4`) confirms the job is running on an allocated GPU node with correct Singularity bind mounts.

---

## Item 2: Debug FSDP2 Run

**Goal:** Verify that a short 1D FSDP2 training run completes successfully and produces expected loss behavior.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Output directory created: `outputs/` contains the job log (e.g., `outputs/<jobname>-<jobid>.out`)
- Loss behavior: Per-step loss values in the log should trend downward over the 20 steps (or stabilize, but not diverge/increase sharply)
- No errors in the log: No OOM, collective communication failures, or model loading errors

**Success Criterion:** The job completes without error, all 4 ranks initialize successfully, and loss decreases monotonically or remains stable. Inspect `outputs/<jobname>-<jobid>.out` to confirm loss values at steps 1–20 show a downward trend.

---

## Item 3: Metrics and Profiler Integration

**Goal:** Verify that profiling runs without overhead issues and that training metrics are correctly logged.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --profiler.enable_profiling
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Profiler artifact present: `outputs/` contains a profiler trace file (e.g., `trace_*.json` or similar torch profiler output)
- Metrics logged: In `outputs/<jobname>-<jobid>.out`, locate lines containing:
  - **Loss per step:** "loss=" or "train_loss=" values
  - **Peak memory usage:** "memory=" or "peak_memory=" values (in GB)
  - **Throughput:** "tokens/sec" or "throughput=" values
  - **Model FLOPs Utilization (MFU):** "MFU=" or "mfu=" values (percentage)

**Success Criterion:** Profiler runs without error, all 4 metrics (loss, memory, tokens/sec, MFU) are logged, and the trace file is parseable. Overhead compared to Item 2 should be minimal (loss curves should be nearly identical).

---

## Item 4: Real Llama-3.1 Tokenizer with the Debug Model

**Goal:** Verify that the debug model runs end-to-end with the real Llama-3.1 tokenizer instead of the debug config's default tokenizer.

**Note:** The launcher hardcodes `--config llama3_debugmodel`; passing `--hf_assets_path` alone does not switch to the 8B model. Running the actual Llama-3.1-8B config requires overriding `--config llama3_8b`, which in turn defaults to the `c4` dataset and would need an offline dataset override — out of scope for this item. This item validates the offline tokenizer path, not 8B-scale model behavior.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch --training.steps=5 --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct
```
(where `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`)

**Expected Outcome:**
- Job status: `COMPLETED`
- Model initialization: `outputs/<jobname>-<jobid>.out` shows the debug model name and its (small) parameter count
- Training steps: All 5 training steps complete without tokenizer errors, OOM, or device mismatches
- Loss values: Per-step loss is finite (not NaN or Inf)

**Success Criterion:** The job completes, the debug model initializes on 4 GPUs under 1D FSDP2, and the real tokenizer (Llama-3.1-8B-Instruct) is loaded and applied without errors offline. This confirms the tokenizer integration; it does not confirm 8B-parameter-scale behavior.

---

## Item 5: Failure Lab

**Goal:** Demonstrate that invalid configurations fail with readable, actionable error messages.

**Task:** Submit a job with an invalid parallelism override:

```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --parallelism.tensor_parallel_degree=3
```

**Expected Outcome:**
- Job status: `FAILED`
- Error message: The log (in `outputs/<jobname>-<jobid>.out`) contains an error message, e.g.:
  - "tensor_parallel_degree=3 does not divide 4 GPUs"
  - "tensor parallelism degree must evenly divide the number of GPUs"
  - Or similar validation error from TorchTitan's config system

**Root Cause Documentation:**
The `--parallelism.tensor_parallel_degree=3` override is invalid because tensor parallelism degree must divide the total number of GPUs evenly. On a 1-node, 4-GPU setup, valid values are `1`, `2`, or `4`. The value `3` does not divide 4, so the job fails during config validation or initialization.

**Fix:** Re-run with a valid override:
```bash
sbatch slurm/launch_1node.sbatch --training.steps=20 --parallelism.tensor_parallel_degree=2
```

**Success Criterion:** The invalid job fails with a clear error. After reading the error, you identify the root cause (divisibility constraint) and can correct it by using `--parallelism.tensor_parallel_degree=2` or `--parallelism.tensor_parallel_degree=4`. The corrected job should complete successfully (verify by running the corrected command above).

---

## Reference Notes

**Container environment:**
- NGC base: `nvcr.io/nvidia/pytorch:26.01-py3` (see `container/dtitan.def`)
- Torch version: ≥ 2.11 (required for FlashAttention support in configs)
- TorchTitan: version 0.2.2
- Key dependencies: torchao, datasets, tokenizers, sentencepiece, pynvml (installed by `dtitan.def`); flash-attn comes from the NGC base image

**Kempner GPU constraints:**
- Partition: `kempner_h100`
- Account: `kempner_dev`
- GPUs per node: 4
- Per-user GPU cap: 8 (so 1-node 4-GPU runs occupy half the cap)
- Typical runtime: Items 1–4 take 1–5 minutes each; Item 5 is quick (<1 minute)

**Common troubleshooting:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Item 1: `[FAIL] torchtitan.train import` | Torch 2.10 (config requires torch ≥ 2.11) | Rebuild container per `container/dtitan.def` |
| Item 2/3: Job times out | Model too large for 4 GPUs; memory thrashing | Reduce `--training.steps` or batch size in config |
| Item 4: OOM during 8B model init | Tokenizer + model weights exceed GPU memory | Reduce per-GPU batch size or sequence length |
| Item 5: Job doesn't fail | Override not applied | Check that `--parallelism.tensor_parallel_degree=3` is parsed correctly; re-run `sbatch` command |

---

## Expected Time

- Item 1: ~30 seconds (preflight only, no GPU allocation)
- Items 2–4: ~2–3 minutes each (short training runs, 4 GPUs, 1 node)
- Item 5: ~1 minute (failed job exits quickly)

**Total:** ~15–20 minutes for the full validation suite.
