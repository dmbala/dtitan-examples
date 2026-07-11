#!/bin/bash
# Level 3 — Lab 2: Pipeline Parallel — reading the last-stage loss (pp=2 + dp_shard=4)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.pipeline_parallel_degree=2 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.steps=20
