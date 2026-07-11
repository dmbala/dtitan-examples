#!/bin/bash
# Level 1 — Capstone: Full Observable Run with Artifacts (1D FSDP2 + profiler)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=50 --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --profiling.enable_profiling
