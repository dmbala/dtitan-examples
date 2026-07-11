#!/bin/bash
# Level 3 — Capstone: HSDP+TP with a Restartable, Quantized (or MoE) Run
# Option A (HSDP+TP + FP8 + DCP) is submitted below — it is the option validated end-to-end
# on kempner_rtx (see docs/labs/level3/README.md). Option B (MoE+EP + DCP) is printed as an
# alternative you can submit instead.
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/

# Option A — HSDP+TP + DCP + FP8
sbatch slurm/launch_8gpu.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_replicate_degree=2 \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --model.converters quantize.linear.float8 \
  --checkpoint.enable --checkpoint.interval=10 --checkpoint.no-last-save-model-only \
  --training.steps=20

# Option B — MoE + EP + DCP (alternative; run this instead of Option A, not in addition to it)
echo "Alternative — Option B (MoE + EP + DCP):"
echo "sbatch slurm/launch_8gpu.sbatch \\
  --model.name deepseek_v3 --model.flavor debugmodel \\
  --model.hf_assets_path=assets/test_tokenizer \\
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \\
  --training.seq_len=512 --training.local_batch_size=8 \\
  --parallelism.data_parallel_shard_degree=8 \\
  --parallelism.expert_parallel_degree=2 \\
  --checkpoint.enable --checkpoint.interval=10 --checkpoint.no-last-save-model-only \\
  --training.steps=20"
