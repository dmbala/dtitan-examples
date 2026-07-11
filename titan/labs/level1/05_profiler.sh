#!/bin/bash
# Level 1 — Lab 5: Metrics and Profiler Integration (loss, memory, throughput, MFU via profiler)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.steps=20 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_profiling
