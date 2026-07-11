#!/bin/bash
# Level 2 — Lab L2.5: DCP Save + Resume — the Restart Lab (save a full checkpoint, then resume)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/

# SAVE — run for 10 steps, checkpointing every 5.
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=4 \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=10

# RESUME — only run this after the SAVE job above has finished (same --job.dump_folder,
# extended to 15 steps). Printed here rather than submitted automatically.
echo "After the SAVE job completes, resume with:"
echo "sbatch slurm/launch_1node.sbatch \\
  --model.hf_assets_path=assets/test_tokenizer \\
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \\
  --training.seq_len=512 --training.local_batch_size=8 \\
  --parallelism.data_parallel_shard_degree=4 \\
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \\
  --training.steps=15"
