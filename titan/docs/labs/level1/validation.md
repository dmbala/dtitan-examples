# Level 1 GPU Validation Checklist

**Purpose:** Verify that Level 1 labs run correctly on the `dtitan-torch211.sif` container (torch 2.11.0a0 / CUDA 13.2) and GPU hardware (Kempner RTX PRO 6000 Blackwell).

**Prerequisites:**
- Container: `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif`
- Hardware: 1 node, 4 GPUs (`kempner_rtx` partition — see the Reference Notes below for why not `kempner_h100`)
- Account: `kempner_dev` (8-GPU/user cap)
- Debug tokenizer: `assets/test_tokenizer` (vendored, vocab 2016)
- Training data: `assets/c4_subset/train.jsonl` — generate once with `python scripts/prepare_c4_subset.py`
- Offline real tokenizer (reference only in L1): `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models` (`Llama-3.1-8B-Instruct`) — valid only with `--model.flavor 8B`, NOT with `debugmodel`

---

## Item 1: Preflight All-Pass

**Goal:** Verify that the environment is correctly configured for TorchTitan execution.

**Command:** (must run inside a GPU allocation — e.g. `srun -p kempner_rtx -A kempner_dev --gres=gpu:4 --pty bash`, or via the 1-node launcher; run from a login node and `device_count` will be `0`)
```bash
singularity exec --nv "$IMAGE" bash -lc 'cd titan && python preflight.py'
```
(where `$IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif`)

**Expected Outcome:**
All 6 checks print `[PASS]`:
- `[PASS] torchtitan import: ok`
- `[PASS] torchtitan.train import: ok`
- `[PASS] tokenizer:assets/test_tokenizer: ok`
- `[PASS] c4 subset: ok`
- `[PASS] writable:outputs: ok`
- `[PASS] gpu visible: device_count=4`

(If `torchtitan.train import` fails, the image is still `dtitan.sif` (torch 2.10) — use `dtitan-torch211.sif` instead. If `c4 subset` fails, run `python scripts/prepare_c4_subset.py` first.)

**Success Criterion:** All 6 checks pass. The `torchtitan.train import` check confirms torch ≥ 2.11 and config availability. GPU visibility (`device_count=4`) confirms the job is running on an allocated `kempner_rtx` GPU node with correct Singularity bind mounts.

---

## Item 2: Config Inspect + Override

**Goal:** Verify that the config system resolves the built-in `llama3` `debugmodel` flavor and that dotted CLI overrides apply.

**Command:**
```bash
singularity exec "$IMAGE" bash -lc 'cd titan && python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args([\"--model.name\",\"llama3\",\"--model.flavor\",\"debugmodel\",\"--training.steps\",\"7\"]).training.steps)"'
```

**Expected Outcome:** Prints `7`.

**Success Criterion:** The resolved config's `training.steps` matches the override. This check does not require a GPU allocation.

---

## Item 3: Fake-Backend Dry-Run

**Goal:** Verify that model build and a training step run correctly with faked collectives, in a single process, with no real multi-GPU communication.

**Command:**
```bash
singularity exec --nv "$IMAGE" bash -lc 'cd titan && NGPU=4 torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
  --model.name llama3 --model.flavor debugmodel \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --comm.mode=fake_backend --training.steps=5'
```
(Or simply run `labs/level1/03_fake_backend.sh`, which submits this via `slurm/launch_fakebackend.sbatch`.)

**Expected Outcome:** The model builds, the log shows `Applied FSDP to the model`, a couple of training steps run, and the job ends with `Training completed` — all in one process.

**Success Criterion:** The dry-run completes cleanly without any real multi-GPU communication. (A bare `NGPU=4 python -m torchtitan.train …` — no `torchrun` — fails with `KeyError: 'LOCAL_RANK'`; this is the wrong form.)

---

## Item 4: Debug FSDP2 Run

**Goal:** Verify that a short 1D FSDP2 training run completes successfully and produces expected loss behavior.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Output directory created: `outputs/` contains the job log (`outputs/<jobname>-<jobid>.out`)
- Model builds: `Model llama3 debugmodel size: 6,163,712 total parameters`
- 4-way FSDP2 (`dp_shard=4`); loss falls step 1 ≈ **8.12** → step 10 ≈ **6.30** → step 20 ≈ **3.55**
- Per-step metrics: `grad_norm`, `memory ≈ 0.32GiB`, `tps ≈ 180k`, `tflops ≈ 8`, `mfu ≈ 2.5%`
- Run ends with `Training completed`
- No errors in the log: no OOM, collective communication failures, or model loading errors

**Note:** A `WARNING - Peak flops undefined for … Blackwell, fallback to A100` line is expected — MFU is computed against the A100 peak-FLOPs table, so treat MFU% as indicative only on this hardware.

**Success Criterion:** The job completes without error, all 4 ranks initialize successfully, and loss decreases as above.

---

## Item 5: Metrics and Profiler Integration

**Goal:** Verify that profiling runs without overhead issues and that training metrics are correctly logged.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_profiling
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Log line: `Dumping profiler traces at step 10`
- Profiler artifacts: `outputs/profile_traces/iteration_10/rank{0,1,2,3}_trace.json` (~13 MB total)
- Metrics logged: same `loss`/`grad_norm`/`memory`/`tps`/`mfu` fields as Item 4

**Success Criterion:** Profiler runs without error, all 4 trace files are present, and the loss curve matches Item 4 (profiling overhead is minimal).

---

## Item 6: Failure Lab

**Goal:** Demonstrate that invalid configurations fail with readable, actionable error messages.

**Task:** Submit a job with an invalid parallelism override:

```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=20 --parallelism.tensor_parallel_degree=3
```

**Expected Outcome:**
- Job status: `FAILED`
- Error message in the log (`outputs/<jobname>-<jobid>.out`):
  ```
  AssertionError: Invalid parallel dims: dp_replicate(1) * dp_shard(4) * cp(1) * tp(3) * pp(1) != WORLD_SIZE(4)
  ```

**Root Cause Documentation:**
The `--parallelism.tensor_parallel_degree=3` override is invalid because the product of all parallelism degrees must equal the world size (4 GPUs here). Valid `tp` values are `1`, `2`, or `4`. The value `3` does not divide 4, so the job fails this assertion during initialization.

**Fix:** Re-run with a valid override:
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=20 --parallelism.tensor_parallel_degree=2 \
  --parallelism.data_parallel_shard_degree=2
```

**Success Criterion:** The invalid job fails with the clean assertion above. After reading the error, you identify the root cause (the parallel-dims product must equal `WORLD_SIZE`) and correct it. The corrected job should complete successfully (verify by running the corrected command above).

---

## Reference Notes

**Container environment:**
- Image: `dtitan-torch211.sif` — NGC base `nvcr.io/nvidia/pytorch:26.03-py3`, torch 2.11.0a0 / CUDA 13.2 (see `container/dtitan.def`)
- TorchTitan: version 0.2.2
- Key dependencies: torchao, datasets, tokenizers, sentencepiece, pynvml (installed by `dtitan.def`); flash-attn comes from the NGC base image
- The plain `dtitan.sif` (torch 2.10 / CUDA 13.0) is the **dtensor** workshop's image — do not use it here.

**Why `kempner_rtx`, not `kempner_h100`:**
The torch-2.11 image is **CUDA 13.2**. `kempner_h100` nodes run driver 575.57.08 (CUDA 12.9); its forward-compat window reaches CUDA 13.0 but not 13.2, so `torch.cuda.init()` fails there with `RuntimeError: The NVIDIA driver on your system is too old (found version 12090)`. `kempner_rtx` (RTX PRO 6000 Blackwell) nodes run driver 595.71.05 (CUDA 13.2), so the image runs there. The older torch-2.10 image, conversely, has no `sm_120` kernels and does not run on Blackwell at all — so Blackwell specifically needs the torch-2.11 image. **Gotcha:** `torch.cuda.device_count()` can return `1` with only a *warning* even when the driver is broken — always confirm with a real tensor op or `torch.cuda.init()`, not just `device_count()`.

**Kempner GPU constraints:**
- Partition: `kempner_rtx` (RTX PRO 6000 Blackwell, 8 GPUs/node, 24 nodes)
- Account: `kempner_dev`, QoS `kempner_base`, MaxTime 2 days
- GPUs used per Level 1 job: 4 (of the 8 available per node — jobs fit on one node)
- Per-user GPU cap: 8
- Typical runtime: Items 1–3 take under a minute each; Items 4–5 take 1–3 minutes; Item 6 is quick (<1 minute)

**Common troubleshooting:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Item 1: `[FAIL] torchtitan.train import` | Wrong image (`dtitan.sif`, torch 2.10) | Use `dtitan-torch211.sif` |
| Item 1: `[FAIL] c4 subset` | `assets/c4_subset/train.jsonl` not generated | Run `python scripts/prepare_c4_subset.py` |
| `RuntimeError: driver too old (found version 12090)` | Job landed on `kempner_h100` | Submit with `--partition=kempner_rtx` |
| Item 3: `KeyError: 'LOCAL_RANK'` | Ran `python -m torchtitan.train` instead of `torchrun` | Use `torchrun --standalone --nproc_per_node=1 -m torchtitan.train …` |
| Item 4/5: Job times out | Model too large for 4 GPUs; memory thrashing | Reduce `--training.steps` or batch size in config |
| `vectorized_gather_kernel: index out of bounds` | Real Llama-3.1 tokenizer paired with `debugmodel` | Use `assets/test_tokenizer` for debug runs; the real tokenizer needs `--model.flavor 8B` |
| Item 6: Job doesn't fail | Override not applied | Check that `--parallelism.tensor_parallel_degree=3` is parsed correctly; re-run `sbatch` command |

---

## Expected Time

- Items 1–3: under a minute each (no multi-GPU training)
- Items 4–5: ~1–3 minutes each (short training runs, 4 GPUs, 1 node)
- Item 6: ~1 minute (failed job exits quickly)

**Total:** ~10–15 minutes for the full validation suite.
