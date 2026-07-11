# Level 3 GPU Validation Checklist

**Purpose:** Verify that Level 3 labs run correctly on the `dtitan-torch211.sif` container (torch 2.11.0a0 / CUDA 13.2) and GPU hardware (Kempner RTX PRO 6000 Blackwell), using the full 8-GPU-per-node topology.

**Prerequisites:**
- Container: `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif`
- Hardware: 1 node, 8 GPUs (`kempner_rtx` partition — RTX PRO 6000 Blackwell has 8 GPUs/node, so the full 3D mesh and the 8-way MoE+EP mesh both fit on one node)
- Account: `kempner_dev`, QoS `kempner_base` (16-GPU/user cap)
- Launcher: `slurm/launch_8gpu.sbatch` — sets the writable HF cache, `TRITON_LIBCUDA_PATH=/.singularity.d/libs`, and `TORCH_FR_BUFFER_SIZE`
- Debug tokenizer: `assets/test_tokenizer` (vendored, vocab 2016) — also valid for the `deepseek_v3` MoE lab, since that debug flavor is also vocab 2048
- Training data: `assets/c4_subset/train.jsonl` — generate once with `python scripts/prepare_c4_subset.py` (same one-time step as Level 1/2)

---

## Item 1: HSDP + TP 3D Mesh

**Goal:** Verify that `dp_replicate=2 × dp_shard=2 × tp=2 = 8` resolves to a real 3D mesh and trains normally.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Mesh line: `dp_replicate=2, dp_shard=2, cp=1, tp=2` (pp=1, ep=1)
- Loss falls step 10 ≈ **6.22** → step 20 ≈ **3.57**
- Run ends with `Training completed`

**Success Criterion:** All 8 ranks initialize on the 3D mesh, the mesh dims match the requested degrees, and loss decreases as above.

---

## Item 2: Pipeline Parallel (last-stage loss)

**Goal:** Verify that `pp=2 × dp_shard=4 = 8` completes, and confirm the sentinel-loss behavior on non-last-stage ranks.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.pipeline_parallel_degree=2 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Non-last-stage ranks log a sentinel loss (e.g. `-4.00000`); `grad_norm` may look large on those ranks — this is expected, not a failure
- The **last pipeline stage's** rank logs the real loss, converging to ≈ **3.61** by step 20

**Success Criterion:** Job completes; the last-stage rank's loss is identified and shows normal convergence, distinguishing it from the sentinel values on other ranks.

---

## Item 3: MoE + Expert Parallel (deepseek_v3)

**Goal:** Verify `deepseek_v3` debugmodel trains under `dp_shard=8 + ep=2`, and confirm the two MoE-specific gotchas are accounted for.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.name deepseek_v3 --model.flavor debugmodel \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Loss falls step 10 ≈ **4.48** → step 20 ≈ **3.47**
- Memory ≈ **2.2 GiB**
- Run ends with `Training completed`

**Success Criterion:** Job completes without `libcuda.so cannot found!` (confirms `TRITON_LIBCUDA_PATH` is set by the launcher) and without a parallel-dims assertion (confirms `dp_shard=8 + ep=2`, not `dp_shard=4 × ep=2`, was used).

---

## Item 4: FP8 (torchao, Blackwell sm_120)

**Goal:** Verify the `quantize.linear.float8` converter applies cleanly and training remains numerically stable.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --model.converters quantize.linear.float8 \
  --parallelism.data_parallel_shard_degree=8 \
  --training.steps=10
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Loss falls step 1 ≈ **8.02** → step 10 ≈ **4.62**
- No numerical errors (no NaN/inf, no `KeyError: 'float8'`)

**Success Criterion:** The converter name `quantize.linear.float8` is accepted (a bare `float8` would fail with `KeyError: 'float8'`), and the job completes with a decreasing loss curve on Blackwell (`sm_120`).

---

## Item 5: Flight Recorder / NCCL Trace Buffer

**Goal:** Verify the NCCL Flight-Recorder ring buffer enables cleanly and comm traces are written.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --comm.trace_buf_size=20000 \
  --training.steps=6
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Comm traces written under `outputs/<...>/comm_traces/`
- No collective errors; the Flight-Recorder ring buffer (`--comm.trace_buf_size` + the launcher's `TORCH_FR_BUFFER_SIZE`) is active throughout

**Success Criterion:** Job completes cleanly with the trace buffer enabled and the `comm_traces/` directory present under the job's `outputs/` folder. (A per-rank Flight-Recorder dump on a hang/abort is not exercised by this healthy run — that behavior is triggered by TorchTitan's `init_timeout_seconds`/`train_timeout_seconds` watchdogs on an actual stuck collective, per the README's framing.)

---

## Reference Notes

**Container environment:**
- Image: `dtitan-torch211.sif` (torch 2.11.0a0 / CUDA 13.2), same image as Levels 1–2
- Partition: `kempner_rtx` (RTX PRO 6000 Blackwell, **8 GPUs/node**) — the full 3D mesh (Item 1) and the 8-way MoE+EP mesh (Item 3) both fit on **one node**, so `slurm/launch_8gpu.sbatch` (not the 2-node launcher) is used for every Level 3 item
- Account: `kempner_dev`, QoS `kempner_base`, **16-GPU/user cap**
- One-time prep step (shared with Levels 1–2, run once before any Level 3 job): `python scripts/prepare_c4_subset.py`

**Blackwell caveat:** logs warn `Peak flops undefined for RTX PRO 6000 Blackwell, fallback to A100` — MFU% is computed against the A100 peak-FLOPs table, so treat MFU as indicative only, not a true Blackwell utilization number.

**Common troubleshooting:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Item 3: `AssertionError: libcuda.so cannot found!` | Triton grouped-GEMM kernel can't find `libcuda.so` | Confirm `TRITON_LIBCUDA_PATH=/.singularity.d/libs` is set — `launch_8gpu.sbatch` sets it by default |
| Item 3: `AssertionError: Invalid parallel dims: … != WORLD_SIZE(8)` | Used `dp_shard=4` + `ep=2` instead of `dp_shard=8` + `ep=2` | `expert_parallel_degree` is orthogonal to the `dp_replicate*dp_shard*cp*tp*pp` product; use `dp_shard=8` |
| Item 4: `KeyError: 'float8'` | Passed bare `--model.converters float8` | Use `--model.converters quantize.linear.float8` |
| Item 2: last-stage rank's loss looks like a normal curve but other ranks show `-4.00000` | Expected PP sentinel-loss behavior, not a bug | Read the **last pipeline stage's** rank log for the real loss |
| Job lands on `kempner_h100` instead of `kempner_rtx` | Wrong partition (torch-2.11 image needs the CUDA-13.2 driver) | Confirm `--partition=kempner_rtx` in the launcher |

---

## Expected Time

- Items 1, 3: ~2–4 minutes each (20-step runs, 8 GPUs)
- Item 2: ~2–4 minutes (20-step PP run)
- Item 4: ~1–3 minutes (10-step run)
- Item 5: ~1 minute (6-step run)

**Total:** ~15–20 minutes for the full Level 3 validation suite.
