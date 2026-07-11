#!/bin/bash
# Level 2 — Lab L2.6: Real-Model Taste — Llama-3.1-8B (real 8B flavor + real tokenizer, 4-way FSDP2)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
MODELS="${MODELS:-/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models}"
exec sbatch slurm/launch_1node.sbatch \
  --model.flavor 8B --model.hf_assets_path=$MODELS/Llama-3.1-8B-Instruct \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=1 \
  --parallelism.data_parallel_shard_degree=4 \
  --training.steps=5
