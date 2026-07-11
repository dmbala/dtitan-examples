#!/bin/bash
# Level 2 — Lab L2.2: Context Parallel (shard the sequence dimension instead of weights)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.context_parallel_degree=2 \
  --training.steps=20
