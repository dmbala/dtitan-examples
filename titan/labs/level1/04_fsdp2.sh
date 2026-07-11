#!/bin/bash
# Level 1 — Lab 4: 1D FSDP2 Single-GPU-Per-Rank Run (first 4-GPU FSDP2 training run)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4
