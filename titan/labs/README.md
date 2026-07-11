# TorchTitan Workshop — Runnable Lab Scripts

Each script here wraps the exact, GPU-validated command from the corresponding lab's
README under `docs/labs/levelN/README.md` — read those for the teaching write-up,
expected artifacts, and success criteria. These scripts just make the commands
copy-paste-free.

**Prerequisites**
- Run `level1/00_prepare_data.sh` once before any training lab — it generates
  `assets/c4_subset/train.jsonl` (the local C4 subset every training lab reads).
- All GPU jobs target `kempner_rtx` (the launchers already set `--partition=kempner_rtx`).
- After submitting, check job output at `titan/outputs/<jobname>-<jobid>.out`
  (the launcher's `--output=outputs/%x-%j.out`; the `sbatch` scripts print the jobid).

## Level 1 — `docs/labs/level1/`

Launcher: `slurm/launch_1node.sbatch` (1 node / 4 GPUs) for the training labs;
Labs 1–2 run directly via `singularity exec` on a login node (no GPU needed).

- `00_prepare_data.sh` — generate the local C4 subset (run once)
- `01_preflight.sh` — environment preflight checks (login node)
- `02_inspect_config.sh` — inspect/override a config (login node)
- `03_fake_backend.sh` — fake-backend dry-run (`slurm/launch_fakebackend.sbatch`)
- `04_fsdp2.sh` — 1D FSDP2 4-GPU training run
- `05_profiler.sh` — same run with the profiler enabled
- `06_failure.sh` — intentionally invalid config (expected to fail with a readable AssertionError)
- `07_capstone.sh` — capstone: full observable FSDP2 run with profiling

## Level 2 — `docs/labs/level2/`

Launcher: `slurm/launch_1node.sbatch` (1 node / 4 GPUs).

- `01_fsdp_tp.sh` — 2D FSDP2 + Tensor Parallel
- `02_context_parallel.sh` — FSDP2 + Context Parallel
- `03_activation_checkpoint.sh` — full activation checkpointing
- `04_memory_snapshot.sh` — CUDA memory snapshot
- `05_dcp_save_resume.sh` — DCP save (submitted) + resume (printed; run after the save job finishes)
- `06_real_8b.sh` — real Llama-3.1-8B taste run
- `07_capstone.sh` — capstone: 2D FSDP2+TP + activation checkpointing + DCP save (submitted) / resume (printed)

## Level 3 — `docs/labs/level3/`

Launcher: `slurm/launch_8gpu.sbatch` (1 node / 8 GPUs).

- `01_hsdp_tp.sh` — HSDP + Tensor Parallel, the real 3D mesh
- `02_pipeline.sh` — Pipeline Parallel
- `03_moe_ep.sh` — Mixture-of-Experts (`deepseek_v3`) + Expert Parallel
- `04_fp8.sh` — FP8 quantization (torchao)
- `05_flight_recorder.sh` — NCCL Flight-Recorder buffer
- `06_capstone.sh` — capstone Option A: HSDP+TP + FP8 + DCP (submitted); Option B: MoE+EP + DCP (printed as an alternative)
