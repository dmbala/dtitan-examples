#!/bin/bash
# Level 3 — Lab 3: Mixture-of-Experts + Expert Parallel (deepseek_v3 debugmodel, ep=2)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_8gpu.sbatch \
  --model.name deepseek_v3 --model.flavor debugmodel \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=8 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=20
