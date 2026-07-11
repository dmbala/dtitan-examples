# Level 3: TorchTitan Production Scaling — HSDP+TP, Pipeline, MoE, FP8, and NCCL Debugging

**Duration:** ~full day
**Hardware:** 1 node, 8 GPUs (`kempner_rtx`)
**Launch:** `slurm/launch_8gpu.sbatch` and CLI commands via Singularity

**Goals:** Compose parallelism dimensions into a real 3D mesh, read pipeline-parallel metrics correctly, enable Mixture-of-Experts with expert parallelism, apply FP8 quantization, and use TorchTitan's Flight-Recorder / NCCL tooling to debug a distributed job. Level 3 is where the failure modes get subtler — a wrong flag combination doesn't just error, it can silently mis-shard, hang a collective, or leave you reading the wrong rank's loss.

**Assets:** Same debug tokenizer as Levels 1–2: `--model.hf_assets_path=assets/test_tokenizer` (vendored, vocab 2016), matching the `llama3` `debugmodel` flavor's `vocab_size=2048`. The `deepseek_v3` `debugmodel` flavor used for the MoE lab is **also** vocab 2048, so the same small tokenizer works there without changes. One-time setup (if not already done in Level 1):
```bash
python scripts/prepare_c4_subset.py
```
Every run below also passes `--training.dataset=c4_test --training.dataset_path=assets/c4_subset`.

**Why 1 node / 8 GPUs, not 2 nodes:** `kempner_rtx` nodes have 8 GPUs each, so a full 3D mesh (`dp_replicate × dp_shard × tp = 8`) or an 8-way expert-parallel MoE run fits on a **single node** — no rendezvous, no inter-node NCCL. `slurm/launch_2node.sbatch` exists only for the >8-GPU planning demos noted below; none of the required Level 3 labs need it.

**Common overrides** used across most labs in this level (spelled out in full in each command, not abbreviated):
```
--model.hf_assets_path=assets/test_tokenizer
--training.dataset=c4_test --training.dataset_path=assets/c4_subset
--training.seq_len=512 --training.local_batch_size=8
```

**Blackwell caveat (carried over from Levels 1–2):** logs warn `Peak flops undefined for RTX PRO 6000 Blackwell, fallback to A100`, so MFU% is computed against the A100 peak-FLOPs table — treat it as indicative only, not a true Blackwell utilization number.

---

## Lab 1: HSDP + Tensor Parallel — the real 3D mesh

**Goal:** Build and confirm a genuine 3D device mesh — `dp_replicate=2 × dp_shard=2 × tp=2 = 8` — combining Hybrid-Sharded Data Parallel (HSDP) with Tensor Parallel in one run.

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

**Expected artifact:** Training log in `outputs/` showing the resolved mesh and per-step loss.

**Success criterion:**
- Mesh line reports `dp_replicate=2, dp_shard=2, cp=1, tp=2` (pp=1, ep=1)
- Loss decreases: step 10 ≈ **6.22** → step 20 ≈ **3.57**
- Run ends with `Training completed`, no collective errors

**Note:** All 8 ranks live on one RTX node, so `dp_replicate`, `dp_shard`, and `tp` all share the same NVLink domain here — unlike a true multi-node deployment where you'd usually put `dp_replicate` across the slower inter-node link and keep `dp_shard`/`tp` inside a node. This lab teaches mesh *composition and correctness*, not topology-aware placement; the latter is a planning exercise (see the scaling demos below).

---

## Lab 2: Pipeline Parallel — reading the last-stage loss

**Goal:** Run Pipeline Parallel (`pp=2`) composed with FSDP2 (`dp_shard=4`), and learn to read PP metrics correctly — most ranks do **not** log the real training loss.

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

**Expected artifact:** Training log in `outputs/`; job completes.

**Success criterion:**
- Job runs to completion on the `pp=2, dp_shard=4` mesh
- You can point to **which rank's loss is real**

**Key teaching point (read this before you conclude the run is broken):** With Pipeline Parallel, only the **last pipeline stage** computes and logs the real loss. Non-last-stage ranks log a **sentinel loss** (e.g. `-4.00000`), and `grad_norm` can look unusually large on those ranks on the debug model — this is expected PP metric behavior, not a bug. Read the **last-stage rank's** loss line: it converges normally (≈ **3.61** by step 20 on this config). If you inspect an early-stage rank's log instead and see a flat `-4.00000` loss, that is not a training failure.

---

## Lab 3: Mixture-of-Experts + Expert Parallel (deepseek_v3)

**Goal:** Train the `deepseek_v3` debug MoE model with Expert Parallel (`ep=2`), and learn the two gotchas that trip up first attempts at MoE+EP.

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

**Expected artifact:** Training log in `outputs/`.

**Success criterion:**
- Loss decreases: step 10 ≈ **4.48** → step 20 ≈ **3.47**
- Memory ≈ **2.2 GiB**
- Run ends with `Training completed`

**Key gotcha #1 — `ep` is orthogonal to the world-size product.** The parallel-dims assertion is `dp_replicate * dp_shard * cp * tp * pp == WORLD_SIZE` — `expert_parallel_degree` is **not** in that product; it's applied *within* the dp ranks. So on 8 GPUs the correct combination is **`--parallelism.data_parallel_shard_degree=8 --parallelism.expert_parallel_degree=2`**, not `dp_shard=4` + `ep=2` (which looks intuitively balanced but fails with `AssertionError: Invalid parallel dims: … != WORLD_SIZE(8)` because `dp_shard=4` alone doesn't cover all 8 ranks and `ep` doesn't fill the gap).

**Key gotcha #2 — the Triton grouped-GEMM kernel needs `TRITON_LIBCUDA_PATH`.** The MoE layer's grouped-GEMM Triton kernel fails with `AssertionError: libcuda.so cannot found!` unless `TRITON_LIBCUDA_PATH=/.singularity.d/libs` is set in the container environment. `slurm/launch_8gpu.sbatch` already sets this for you — if you instead run the equivalent command by hand inside `singularity exec` without going through the launcher, you must export it yourself.

---

## Lab 4: FP8 quantization (torchao, Blackwell sm_120)

**Goal:** Convert eligible linear layers to FP8 via torchao and confirm training remains numerically well-behaved on Blackwell.

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

**Expected artifact:** Training log in `outputs/`.

**Success criterion:**
- Loss decreases: step 1 ≈ **8.02** → step 10 ≈ **4.62**
- Run ends with `Training completed`, no numerical errors

**Key gotcha — the converter's registered name is `quantize.linear.float8`, not `float8`.** Passing bare `--model.converters float8` fails with `KeyError: 'float8'`. The correct, space-separated form is `--model.converters quantize.linear.float8` (the MoE-flavored equivalent, not exercised in this lab, is `quantize.grouped_mm.float8`). FP8 here requires SM89+, which Blackwell's `sm_120` satisfies, and torchao (0.17 in this image). The converter's own knobs live under `--quantize.linear.float8.*` if you want to tune recipe/scaling behavior further.

---

## Lab 5: Flight Recorder / NCCL debugging

**Goal:** Enable the NCCL Flight-Recorder ring buffer and understand how it's used to debug a hung or mismatched collective at scale.

**Command:**
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --comm.trace_buf_size=20000 \
  --training.steps=6
```

**Expected artifact:** Training log in `outputs/`; comm traces under `outputs/<...>/comm_traces/`.

**Success criterion:**
- Job completes normally with `--comm.trace_buf_size=20000` set
- TorchTitan writes comm traces under `outputs/<...>/comm_traces/`

**How this is meant to be used:** `--comm.trace_buf_size` (paired with the launcher's `TORCH_FR_BUFFER_SIZE` environment variable) turns on the NCCL Flight-Recorder ring buffer, which records recent collective operations per rank. On a healthy run like this one, it just runs quietly in the background. The payoff comes on a **hang or abort**: TorchTitan's `init_timeout_seconds`/`train_timeout_seconds` watchdogs trigger a per-rank Flight-Recorder dump when a collective doesn't complete in time, and you use those dumps to isolate which rank/collective is stuck — the same diagnostic pattern as the DTensor workshop's Level 3 NCCL-debug lab, but via TorchTitan's built-in `comm.*` config surface instead of a hand-rolled probe. **Use `TORCH_FR_BUFFER_SIZE`** (current on torch ≥ 2.11) — the older `TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated and the launcher does not set it.

---

## Scaling/architecture demos (planning only)

`deepseek_v3` flavors `16B`, `236B`, `671B` and `llama3` flavors `70B`, `405B` are registered in TorchTitan for **mesh/topology planning walkthroughs only** — none of them fit for actual training on an 8-GPU budget. Use these to reason about the layout you'd need at real scale without submitting a training job: resolve the flavor's config (same pattern as Level 1 Lab 2) to inspect its shape, then sketch the `dp_replicate × dp_shard × tp × pp × ep` layout and memory budget it would require, e.g.:
```bash
python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--model.name','deepseek_v3','--model.flavor','236B']).model)"
```
This is deliberately a **planning-only** exercise — no `sbatch` submission for these flavors in this workshop.

---

## Capstone: HSDP+TP with a Restartable, Quantized (or MoE) Run

**Goal:** Combine what you've built in this level into one observable, restartable run: a real 3D mesh (or the MoE+EP mesh), a DCP checkpoint you can resume from, and one performance/architecture feature — **FP8 or MoE, your choice**.

**Option A — HSDP+TP + DCP + FP8** (extends Lab 1 and Lab 4, plus the DCP-resume pattern from Level 2's checkpoint lab):
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --model.converters quantize.linear.float8 \
  --checkpoint.enable --checkpoint.interval=10 --checkpoint.no-last-save-model-only \
  --training.steps=20
```

**Option B — MoE + EP + DCP** (extends Lab 3; trades away the `tp`/`dp_replicate` dims for `ep`, since MoE+EP combined with a full 3D HSDP+TP mesh has not been run on this hardware — see the honesty note below):
```bash
sbatch slurm/launch_8gpu.sbatch \
  --model.name deepseek_v3 --model.flavor debugmodel \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.expert_parallel_degree=2 \
  --checkpoint.enable --checkpoint.interval=10 --checkpoint.no-last-save-model-only \
  --training.steps=20
```

**Honesty note:** every individual flag above is independently GPU-validated (Lab 1, Lab 3, Lab 4, and Level 2's `--checkpoint.enable/--checkpoint.interval/--checkpoint.no-last-save-model-only` resume pattern each ran cleanly on their own). The specific *combination* in each capstone option was not separately re-run end-to-end as one job at authoring time — treat the composed command as a well-formed extrapolation from validated pieces, not as a claim that this exact line has been observed to produce the numbers above. Recall from Level 2 that `checkpoint.last_save_model_only` defaults to `True`, so without `--checkpoint.no-last-save-model-only` the final checkpoint is model-only and will fail to resume with `RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_0`.

**Capstone deliverables:**
1. Launch command (whichever option you chose, or your own combination)
2. Training log showing the mesh, loss curve, and (if you resumed) the "Loading the checkpoint from …" → "Finished loading the checkpoint" lines
3. A short note: which failure mode from this level did you hit first, and how did the error message point you at the fix?

---

## Common Errors and Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `AssertionError: libcuda.so cannot found!` (MoE lab) | Triton grouped-GEMM kernel can't find `libcuda.so` inside the container | Set `TRITON_LIBCUDA_PATH=/.singularity.d/libs` — `slurm/launch_8gpu.sbatch` already does this; only an issue if you bypass the launcher |
| `AssertionError: Invalid parallel dims: … != WORLD_SIZE(8)` (MoE lab) | Tried `dp_shard=4` + `ep=2`, expecting `ep` to fill the remaining GPUs | `expert_parallel_degree` is orthogonal to the `dp_replicate*dp_shard*cp*tp*pp` product — use `--parallelism.data_parallel_shard_degree=8 --parallelism.expert_parallel_degree=2` on 8 GPUs |
| `KeyError: 'float8'` (FP8 lab) | Passed bare `--model.converters float8` | The registered converter name is `quantize.linear.float8` (or `quantize.grouped_mm.float8` for MoE) |
| Non-last-stage rank's loss is stuck at `-4.00000` (PP lab) | Reading a non-last pipeline-stage rank's log | This is expected: only the **last** PP stage computes and logs the real loss; read that rank's log instead |

---

## What you can do after Level 3

- Compose HSDP with Tensor Parallel into a real, verified 3D `DeviceMesh` and read its resolved dims from the log
- Run Pipeline Parallel and correctly identify which rank's loss is meaningful
- Enable Mixture-of-Experts with Expert Parallel, knowing `ep` sits outside the dense parallel-dims product and that the MoE Triton kernel needs `TRITON_LIBCUDA_PATH`
- Apply FP8 quantization via torchao using the correct converter name, on Blackwell hardware
- Turn on the NCCL Flight-Recorder buffer and know how a dump-on-hang would be used to isolate a stuck collective/rank
- Read a registered model flavor's config for mesh/topology planning at scales (70B/405B/236B/671B) beyond what a single 8-GPU node can train
