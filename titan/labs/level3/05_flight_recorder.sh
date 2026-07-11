#!/bin/bash
# Level 3 — Lab 5: Flight Recorder / NCCL debugging (enable the NCCL Flight-Recorder ring buffer)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --comm.trace_buf_size=20000 \
  --training.steps=6
