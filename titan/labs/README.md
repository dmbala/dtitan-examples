# TorchTitan Workshop — Labs Overview

Hands-on labs that **operate TorchTitan** (configs + launchers + CLI overrides) on the
Kempner cluster, in three levels that build on one another. This directory holds the
**runnable scripts**; the matching **teaching write-ups** (concepts, full expected
outputs, troubleshooting) live under [`../docs/labs/`](../docs/labs/).

- **[`level1/`](level1/README.md)** — Run & Observe: a small, observable 1D FSDP2 run
- **[`level2/`](level2/README.md)** — Composable Parallelism & Restart: 2D FSDP2+TP, context parallel, checkpoint/resume, memory
- **[`level3/`](level3/README.md)** — Production Scaling & Debugging: HSDP+TP 3D, pipeline, MoE+expert-parallel, FP8, NCCL/Flight-Recorder

## The learning arc

The same built-in **`llama3` `debugmodel`** (tiny, fast) runs through Levels 1–2, and
`deepseek_v3` appears for MoE in Level 3, so each level extends the last rather than
starting over:

```
Level 1   one node, 4 GPUs    → launch a run, read loss/throughput/MFU, dry-run a config, break one on purpose
Level 2   one node, 4 GPUs    → compose FSDP2 with tensor & context parallel, checkpoint and resume, profile memory
Level 3   one node, 8 GPUs    → a real 3D mesh, pipeline, Mixture-of-Experts, FP8, and debugging failures at scale
```

Within each level the labs follow the order real distributed jobs are built:

> **correct → restartable → fast → debuggable**

Each level also includes intentional gotchas you learn to diagnose — an invalid
parallel-dims config (L1), the model-only-last-checkpoint resume trap (L2), and the
expert-parallel / Triton-libcuda pitfalls (L3).

## Prerequisites (read once)

- **Container:** the labs use `dtitan-torch211.sif` (torch 2.11 / CUDA 13.2); the launchers
  hardcode it. Do not use the plain `dtitan.sif` (torch 2.10 — that's the DTensor workshop's image).
- **Hardware:** all GPU jobs run on **`kempner_rtx`** (RTX PRO 6000 Blackwell, 8 GPUs/node) —
  the launchers set `--partition=kempner_rtx --account=kempner_dev`. The `kempner_h100`/`h200`
  nodes cannot run this image until their driver is upgraded to CUDA 13.2.
- **One-time data prep:** generate the local C4 subset before any training lab:
  ```bash
  bash level1/00_prepare_data.sh          # writes ../assets/c4_subset/train.jsonl
  ```

## Running a lab

Each lab is one script. Submit it, then read the job output:

```bash
bash level2/01_fsdp_tp.sh                 # prints "Submitted batch job <id>"
squeue -u "$USER"                         # watch it
tail -f ../outputs/titan-l12-<id>.out     # job output: ../outputs/<jobname>-<jobid>.out
```

- Level 1 & 2 use `../slurm/launch_1node.sbatch` (4 GPUs); Level 3 uses
  `../slurm/launch_8gpu.sbatch` (8 GPUs on one node). Level 1's preflight/config-inspect
  scripts run directly on the login node (no GPU).
- The DCP-resume and capstone scripts submit the first (save) job and **print** the
  follow-up (resume) command to run once the save job finishes.
- Read the per-level `README.md` here for the script list + expected result, and
  `../docs/labs/levelN/README.md` for the concepts, full expected outputs, and troubleshooting.
