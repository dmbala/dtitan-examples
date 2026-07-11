# Level 2 GPU Validation Checklist

**Purpose:** Verify that Level 2 labs run correctly on the `dtitan-torch211.sif` container (torch 2.11.0a0 / CUDA 13.2) and GPU hardware (Kempner RTX PRO 6000 Blackwell).

**Prerequisites:**
- Container: `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif`
- Hardware: 1 node, 4 GPUs (`kempner_rtx` partition)
- Account: `kempner_dev` (8-GPU/user cap)
- Debug tokenizer: `assets/test_tokenizer` (vendored, vocab 2016)
- Training data: `assets/c4_subset/train.jsonl` — generate once with `python scripts/prepare_c4_subset.py`
- Real tokenizer (Item 6 only): `$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models` (`Llama-3.1-8B-Instruct`) — valid only with `--model.flavor 8B`, NOT with `debugmodel`

---

## Item 1: 2D FSDP2 + Tensor Parallel

**Goal:** Verify TorchTitan builds a composed 2D mesh (`dp_shard=2 × tp=2`) and trains correctly.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \
  --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Mesh line: `Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=1`
- Loss falls step 10 ≈ **6.31** → step 20 ≈ **3.72**
- Run ends with `Training completed`

**Success Criterion:** The job completes, the mesh line confirms `dp_shard=2, tp=2`, and loss decreases as above.

---

## Item 2: Context Parallel

**Goal:** Verify TorchTitan builds a composed 2D mesh (`dp_shard=2 × cp=2`) that shards the sequence dimension instead of the model.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=2 \
  --training.steps=20
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Mesh line shows `dp_shard=2, cp=2, tp=1`
- Loss falls step 10 ≈ **6.02** → step 20 ≈ **3.59**
- Run ends with `Training completed`

**Success Criterion:** The job completes, the mesh line confirms `cp=2`, and loss decreases as above.

---

## Item 3: Activation Checkpointing

**Goal:** Verify `full` activation checkpointing applies correctly and the run completes.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --activation_checkpoint.mode=full \
  --training.steps=10
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Log line: `Applied full activation checkpointing to the model`
- Run ends with `Training completed`

**Success Criterion:** The confirmation line appears and the job completes without error. (Modes available: `selective` (default), `full`, `memory_budget`, `none` — compare peak `memory:` in the step logs across modes.)

---

## Item 4: Memory Snapshot

**Goal:** Verify a CUDA memory snapshot is written for later inspection.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_memory_snapshot \
  --training.steps=6
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Pickled memory snapshot written under `outputs/<...>/memory_snapshot/` (viewable at <https://pytorch.org/memory_viz>)

**Success Criterion:** The snapshot file(s) exist under `outputs/<...>/memory_snapshot/` and the job completes without error.

---

## Item 5: DCP Save + Resume (the Restart Lab)

**Goal:** Verify a full (model+optimizer+dataloader) checkpoint saves correctly and a subsequent job resumes from it and continues training.

**SAVE command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=10
```

**Expected Outcome (SAVE):**
- Job status: `COMPLETED`
- `outputs/<...>/checkpoint/step-5` and `outputs/<...>/checkpoint/step-10` written (both full checkpoints)

**RESUME command (same `--job.dump_folder` as the SAVE run):**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=15
```

**Expected Outcome (RESUME):**
- Job status: `COMPLETED`
- Log: `Loading the checkpoint from …/checkpoint/step-10.` → `Finished loading the checkpoint in 0.24 seconds.` → training continues to step 15 → `Training completed`

**Negative-case check (document, do not skip):** Re-running SAVE and RESUME
**without** `--checkpoint.no-last-save-model-only` produces a final checkpoint
that is model-only (`checkpoint.last_save_model_only` defaults to `True`);
resuming from it fails with:
```
RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_0
```
Also confirm the bare assignment form is rejected: passing
`--checkpoint.last_save_model_only=false` fails with `Unrecognized options: false`
— the negated flag form `--checkpoint.no-last-save-model-only` is required.

**Success Criterion:** SAVE produces both checkpoint directories; RESUME loads
`step-10` and completes at step 15 with no errors; the negative case reproduces
the `Missing key ... dataloader` failure exactly as described.

---

## Item 6: Real-Model Taste — Llama-3.1-8B

**Goal:** Verify the real `8B` flavor builds with the correct parameter count and trains under 4-way FSDP2 with the real Llama tokenizer.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.flavor 8B --model.hf_assets_path=$MODELS/Llama-3.1-8B-Instruct \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=1 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.steps=5
```

**Expected Outcome:**
- Job status: `COMPLETED`
- Log line: `Model llama3 8B size: 8,030,261,248 total parameters`
- Step 1 loss ≈ **12.27**
- Memory ≈ **32.8 GiB (34.5%)/GPU**
- Run ends with `Training completed`

**Success Criterion:** The parameter count matches exactly (`8,030,261,248`), the job completes, and memory/loss are in the expected range. (Random-init model — no pretrained weights needed; only the tokenizer is real.)

---

## Reference Notes

**Container environment:**
- Image: `dtitan-torch211.sif` — torch 2.11.0a0 / CUDA 13.2 (same image as Level 1; see `titan/docs/labs/level1/validation.md` for build provenance)
- TorchTitan: version 0.2.2
- Partition: `kempner_rtx` (RTX PRO 6000 Blackwell), account `kempner_dev`, QoS `kempner_base`, MaxTime 2 days
- GPUs used per Level 2 job: **4** (`slurm/launch_1node.sbatch`, 1 node)
- One-time setup before any Level 2 lab: `python scripts/prepare_c4_subset.py`

**Blackwell MFU caveat:** logs warn `Peak flops undefined for RTX PRO 6000 Blackwell, fallback to A100` — MFU% is computed against the A100 peak-FLOPs table, so treat MFU as indicative only, not a true Blackwell utilization number (same caveat as Level 1).

**Common troubleshooting:**

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Item 5 RESUME: `RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_0` | Resumed from a model-only final checkpoint | Re-run SAVE with `--checkpoint.no-last-save-model-only` |
| Item 5: `Unrecognized options: false` | Used the bare `--checkpoint.last_save_model_only=false` assignment form | Use `--checkpoint.no-last-save-model-only` instead |
| Item 6: `vectorized_gather_kernel: index out of bounds` | Real Llama tokenizer paired with `debugmodel` instead of `8B` | Only pair `$MODELS/Llama-3.1-8B-Instruct` with `--model.flavor 8B` |
| Items 1/2: `AssertionError: Invalid parallel dims: … != WORLD_SIZE(4)` | Parallelism degrees don't multiply to 4 | Use degrees that multiply to 4, e.g. `dp_shard=2 × tp=2` or `dp_shard=2 × cp=2` |
| Item 4: no file under `memory_snapshot/` | `--profiling.enable_memory_snapshot` omitted or typoed | Confirm the section name is `profiling`, not `profiler` (same gotcha as Level 1's profiler section) |

---

## Expected Time

- Items 1–4: ~1–3 minutes each (short training runs, 4 GPUs, 1 node)
- Item 5: ~2–4 minutes total (SAVE + RESUME, plus the optional negative-case re-run)
- Item 6: ~1–2 minutes (5 steps on the 8B flavor)

**Total:** ~20–25 minutes for the full validation suite.
