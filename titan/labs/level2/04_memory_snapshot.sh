#!/bin/bash
# Level 2 — Lab L2.4: Memory Snapshot (capture a CUDA memory snapshot for OOM root-causing)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_memory_snapshot \
  --training.steps=6
