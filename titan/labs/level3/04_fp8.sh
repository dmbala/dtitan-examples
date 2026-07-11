#!/bin/bash
# Level 3 — Lab 4: FP8 quantization (torchao, Blackwell sm_120)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --model.converters quantize.linear.float8 \
  --parallelism.data_parallel_shard_degree=8 \
  --training.steps=10
