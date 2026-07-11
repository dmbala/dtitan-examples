# Level 2 — Composable Parallelism & Restart (lab scripts)

Compose FSDP2 with tensor and context parallelism, make a run restartable with Distributed
Checkpointing, profile memory, and take a real Llama-3.1-8B for a spin — the order real jobs
are built: **correct → restartable → fast**.

**Full teaching write-up:** [`../../docs/labs/level2/README.md`](../../docs/labs/level2/README.md)
· **Validation checklist:** [`../../docs/labs/level2/validation.md`](../../docs/labs/level2/validation.md)

**Hardware:** all labs use `../../slurm/launch_1node.sbatch` (1 node / **4 GPUs**, `kempner_rtx`).
Requires the one-time `../level1/00_prepare_data.sh`. Job output → `../../outputs/<jobname>-<jobid>.out`.

| Script | Teaches | Validated result |
|--------|---------|------------------|
| `01_fsdp_tp.sh` | 2D mesh: FSDP2 shard ×2 × tensor-parallel ×2 | loss **6.31 → 3.72**; mesh `dp_shard=2, tp=2` |
| `02_context_parallel.sh` | context parallelism (split the sequence) | loss **6.02 → 3.59**; mesh `dp_shard=2, cp=2` |
| `03_activation_checkpoint.sh` | `--activation_checkpoint.mode=full` (recompute vs memory) | "Applied full activation checkpointing"; lower peak memory |
| `04_memory_snapshot.sh` | capture a CUDA memory snapshot | pickle under `outputs/<...>/memory_snapshot/` (view at pytorch.org/memory_viz) |
| `05_dcp_save_resume.sh` | **the restart lab** — DCP save + resume | saves `checkpoint/step-5,10`; resume: "Finished loading the checkpoint" → continues |
| `06_real_8b.sh` | real Llama-3.1-8B taste (real tokenizer) | "Model llama3 8B size: **8,030,261,248**"; ~32.8 GiB/GPU |
| `07_capstone.sh` | 2D FSDP2+TP + full AC + DCP in one run | saves full checkpoints; validated end-to-end |

**The restart gotcha (Lab 5):** `checkpoint.last_save_model_only` defaults to `True`, so the
*final* checkpoint is model-only and won't resume (`RuntimeError: Missing key … dataloader.dp_rank_0`).
The lab passes **`--checkpoint.no-last-save-model-only`** to save a full, resumable checkpoint.
`05_dcp_save_resume.sh` submits the SAVE and prints the RESUME command — run it after the save job finishes.

```bash
bash ../level1/00_prepare_data.sh    # once, if you haven't
bash 05_dcp_save_resume.sh           # submits SAVE, prints RESUME
```
