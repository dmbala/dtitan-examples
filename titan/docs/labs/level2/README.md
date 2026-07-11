# Level 2: Composable Parallelism, Memory, and Restartable Training

**Duration:** ~full day
**Hardware:** 1 node, 4 GPUs (`kempner_rtx`)
**Launch:** `slurm/launch_1node.sbatch` and CLI commands via Singularity

**Goals:** Compose FSDP2 with tensor parallelism and context parallelism into 2D
meshes, trade memory for recompute with activation checkpointing, capture a CUDA
memory snapshot, and — the centerpiece — save and resume a training run with
Distributed Checkpointing (DCP) without losing dataloader state. Close with a
taste of a real 8B model.

**Assets:** Same debug tokenizer and C4 subset as Level 1 —
`--model.hf_assets_path=assets/test_tokenizer` (vocab 2016, matches the
`llama3` `debugmodel` flavor's `vocab_size=2048`) and
`--training.dataset=c4_test --training.dataset_path=assets/c4_subset`. If you
haven't generated the subset yet, run once from `titan/`:
```bash
python scripts/prepare_c4_subset.py
```
This writes `titan/assets/c4_subset/train.jsonl` (3000 real C4 docs; git-ignored).

All labs below use `slurm/launch_1node.sbatch` (1 node / 4 GPUs on
`kempner_rtx`), the same launcher as Level 1. The launcher already sets
`--model.name llama3 --model.flavor debugmodel`, the `dtitan-torch211.sif`
image, and the writable HF cache env — everything after `launch_1node.sbatch`
in the commands below is appended to that entrypoint, and (per the
"flavor override wins" rule) any `--model.flavor`/`--model.name` you pass
overrides the launcher's default.

---

## Lab L2.1: 2D FSDP2 + Tensor Parallel

**Goal:** Compose two parallelism dimensions in one mesh: FSDP2 for data
parallelism and tensor parallelism for the model's linear layers, and confirm
TorchTitan builds and names the resulting device mesh.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \
  --training.steps=20
```

**Expected artifact:** Training log in `outputs/` showing the 2D mesh and a
falling loss curve.

**Success criterion:**
- Mesh line in the log: `Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=1`
- Loss decreases: step 10 ≈ **6.31** → step 20 ≈ **3.72**
- Run ends with `Training completed`

**Tip:** `dp_shard=2 × tp=2 = 4` — the product of all parallelism degrees must
equal the world size (4 GPUs), the same rule from Level 1's failure lab.

---

## Lab L2.2: Context Parallel

**Goal:** Replace tensor parallelism with context parallelism (CP), which
shards the sequence dimension across ranks instead of the model's weights, and
compare the mesh and loss curve to L2.1.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=2 \
  --training.steps=20
```

**Expected artifact:** Training log in `outputs/` with a mesh line showing
`cp=2` instead of `tp=2`.

**Success criterion:**
- Mesh line: `dp_shard=2, cp=2, tp=1`
- Loss decreases: step 10 ≈ **6.02** → step 20 ≈ **3.59**
- Run ends with `Training completed`

**Tip:** CP splits `seq_len` across the `cp` ranks (the debug model uses sdpa
attention under CP). Compare this loss curve to L2.1's — both are valid 2D
compositions of the same 4 GPUs, but they parallelize different axes.

---

## Lab L2.3: Activation Checkpointing (Memory vs. Recompute)

**Goal:** Trade compute for memory by recomputing activations during the
backward pass instead of storing them, and observe the effect on peak memory.

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

**Expected artifact:** Training log in `outputs/` with an activation
checkpointing confirmation line and per-step `memory:` metrics.

**Success criterion:**
- Log line: `Applied full activation checkpointing to the model`
- Run ends with `Training completed`

**Tip:** `--activation_checkpoint.mode` accepts `selective` (the default),
`full`, `memory_budget`, and `none`. Re-run with `--activation_checkpoint.mode=none`
(or omit the flag) and compare the peak `memory:` figure in the step logs
across modes — `full` should show the lowest peak memory at the cost of more
recompute.

---

## Lab L2.4: Memory Snapshot

**Goal:** Capture a CUDA memory snapshot to see exactly what is consuming GPU
memory at a point in training, the same diagnostic used to root-cause an OOM.

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

**Expected artifact:** A pickled CUDA memory snapshot written under
`outputs/<...>/memory_snapshot/`.

**Success criterion:**
- Run ends with `Training completed`
- Snapshot file(s) present under `outputs/<...>/memory_snapshot/`

**Tip:** Load the snapshot at <https://pytorch.org/memory_viz> to inspect the
allocator timeline and identify the largest tensors/categories. This is the
tool to reach for the moment a run OOMs — capture the snapshot before you
start guessing at batch-size or seq-len fixes.

---

## Lab L2.5: DCP Save + Resume — the Restart Lab

**Goal:** Save a full (model + optimizer + dataloader) Distributed Checkpoint
partway through training, then resume from it and confirm the run continues
from the correct step. This is the flagship lab of Level 2 — restartability is
what makes long training runs on shared, preemptible clusters practical.

**SAVE — run for 10 steps, checkpointing every 5:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=10
```
**Expected artifact:** `outputs/<...>/checkpoint/step-5` and
`outputs/<...>/checkpoint/step-10` (both full checkpoints).

**RESUME — same job's `--job.dump_folder`, extend to 15 steps:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=15
```
**Expected artifact:** Log shows `Loading the checkpoint from …/checkpoint/step-10.`
→ `Finished loading the checkpoint in 0.24 seconds.` → training continues to
step 15 → `Training completed`.

**Success criterion:**
- SAVE run produces both `step-5` and `step-10` checkpoint directories
- RESUME run loads from `step-10` and completes at step 15 with no errors

### THE GOTCHA (read this before you run it)

`checkpoint.last_save_model_only` **defaults to `True`** — meaning the
*final* checkpoint TorchTitan writes is a model-only checkpoint (weights
only, meant for export/inference), **not** resumable. If you omit
`--checkpoint.no-last-save-model-only` and try to resume from that last
checkpoint, you get:
```
RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_0
```
The fix is `--checkpoint.no-last-save-model-only`, which makes the *last*
checkpoint a full checkpoint (model + optimizer + dataloader state) — the same
form the interval checkpoints already use — so it is resumable like any other.

Also note: `--checkpoint.last_save_model_only=false` (the bare `=false` form)
is **rejected** — `Unrecognized options: false`. TorchTitan's boolean flags use
the `--no-<flag>` negation form, not `=false`/`=true` assignment. Always use
`--checkpoint.no-last-save-model-only`.

**Tip:** Reproduce the failure once on purpose — run SAVE and RESUME without
`--checkpoint.no-last-save-model-only` and confirm you hit the
`Missing key ... dataloader` error. Then add the flag and confirm the resume
succeeds. Seeing the failure mode first makes the fix stick.

---

## Lab L2.6: Real-Model Taste — Llama-3.1-8B

**Goal:** Run the real `8B` flavor with the real Llama-3.1 tokenizer (from
random initialization — no pretrained weights needed) under 4-way FSDP2, and
see actual parameter counts and memory usage at a realistic model scale.

**Command:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.flavor 8B --model.hf_assets_path=$MODELS/Llama-3.1-8B-Instruct \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=1 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.steps=5
```
(`$MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`.)

**Expected artifact:** Training log in `outputs/` showing the 8B parameter
count and step metrics.

**Success criterion:**
- Log line: `Model llama3 8B size: 8,030,261,248 total parameters`
- Step 1 loss ≈ **12.27**
- Memory ≈ **32.8 GiB (34.5%)/GPU**
- Run ends with `Training completed`

**Tip:** The real Llama tokenizer has vocab 128256, which matches the `8B`
flavor's embedding — this is the pairing Level 1 warned you *not* to use with
`debugmodel`. `local_batch_size=1` keeps this small run's memory footprint
manageable; raise it (or `seq_len`) to see how memory scales.

---

## Capstone: Composable, Restartable, Checkpointed Run

**Goal:** Compose everything from this level into one run: a 2D FSDP2 + TP
mesh, activation checkpointing for memory headroom, and a DCP save-then-resume
cycle — the profile of a real production training job.

**Task:** Design and run a short 2D FSDP2+TP training scenario with activation
checkpointing enabled and full checkpointing turned on, then kill/resubmit
(or simply resubmit with a longer `--training.steps`) to demonstrate a clean
resume.

**Example — SAVE:**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \
  --activation_checkpoint.mode=full \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=10
```

**Example — RESUME (same `--job.dump_folder`):**
```bash
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \
  --activation_checkpoint.mode=full \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=20
```

**Capstone deliverables:**

1. **Launch commands** — the exact SAVE and RESUME `sbatch` commands you ran.
2. **Mesh confirmation** — the `Building device mesh with parallelism: …`
   line from the log, confirming `dp_shard=2, tp=2`.
3. **Activation checkpointing confirmation** — the
   `Applied full activation checkpointing to the model` line.
4. **Checkpoint evidence** — the `outputs/<...>/checkpoint/step-*` directories
   from SAVE, and the `Loading the checkpoint from … Finished loading the
   checkpoint in … seconds.` lines from RESUME.
5. **Plausibility note** — a 2–3 sentence summary: did loss continue falling
   smoothly across the resume boundary (no restart spike or unexpected jump),
   and does peak memory look lower than the equivalent run without activation
   checkpointing?

**Success criterion:** SAVE completes and writes checkpoints; RESUME loads a
full (not model-only) checkpoint and completes without the `Missing key ...
dataloader` error; loss is continuous across the save/resume boundary.

---

## Common Errors and Troubleshooting

| Error | Likely cause | Fix |
|-------|--------------|-----|
| `RuntimeError: Missing key in checkpoint state_dict: dataloader.dp_rank_0` | Resuming from a model-only final checkpoint (`checkpoint.last_save_model_only` defaults to `True`) | Pass `--checkpoint.no-last-save-model-only` on the SAVE run so the last checkpoint is a full (model+optimizer+dataloader) checkpoint |
| `Unrecognized options: false` when setting a checkpoint/parallelism boolean | Bare `--checkpoint.last_save_model_only=false` assignment form is rejected | Use the `--no-<flag>` negation form, e.g. `--checkpoint.no-last-save-model-only` |
| `vectorized_gather_kernel: index out of bounds` on an 8B run | Real Llama tokenizer (vocab 128256) paired with `debugmodel` instead of `8B` | The real Llama-3.1 tokenizer is only valid with `--model.flavor 8B` — use `assets/test_tokenizer` for all `debugmodel` runs |
| `AssertionError: Invalid parallel dims: … != WORLD_SIZE(4)` | Parallelism degrees (`dp_shard × tp × cp × pp`) don't multiply to 4 GPUs | Pick degrees that multiply to 4, e.g. `dp_shard=2 × tp=2`, or `dp_shard=2 × cp=2` |
| MFU % looks too low to be believed | `WARNING - Peak flops undefined for RTX PRO 6000 Blackwell, fallback to A100` — MFU is computed against the A100 peak-FLOPs table, not a Blackwell one | Treat MFU as indicative only on this hardware, not an exact utilization number |
| `c4 subset` / dataset load error | `assets/c4_subset/train.jsonl` not generated yet | Run `python scripts/prepare_c4_subset.py` once from `titan/` |
| Job stuck / never writes a checkpoint | `--checkpoint.enable` omitted, or `--checkpoint.interval` larger than `--training.steps` | Pass both `--checkpoint.enable` and an `--checkpoint.interval` ≤ the total step count |

---

## Next Steps

After completing all six labs and the capstone:
- Compare your L2.1 (TP) and L2.2 (CP) loss curves and note which mesh
  composition you'd reach for on a longer, real training job.
- Confirm you can explain, in your own words, why the DCP resume gotcha
  happens (the `last_save_model_only` default) and why the fix works.
- Proceed to **Level 3** to explore multi-node scaling, pipeline and expert
  parallelism, FP8, and NCCL flight-recorder debugging.
