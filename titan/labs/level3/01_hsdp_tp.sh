#!/bin/bash
# Level 3 — Lab 1: HSDP + Tensor Parallel — the real 3D mesh (dp_replicate=2 x dp_shard=2 x tp=2)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --training.steps=20
