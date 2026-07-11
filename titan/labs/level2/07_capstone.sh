#!/bin/bash
# Level 2 — Capstone: Composable, Restartable, Checkpointed Run (2D FSDP2+TP + act. ckpt + DCP)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/

# SAVE
sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.seq_len=512 --training.local_batch_size=8 \
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \
  --activation_checkpoint.mode=full \
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \
  --training.steps=10

# RESUME — only run this after the SAVE job above has finished (same --job.dump_folder,
# extended to 20 steps). Printed here rather than submitted automatically.
echo "After the SAVE job completes, resume with:"
echo "sbatch slurm/launch_1node.sbatch \\
  --model.hf_assets_path=assets/test_tokenizer \\
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \\
  --training.seq_len=512 --training.local_batch_size=8 \\
  --parallelism.data_parallel_shard_degree=2 --parallelism.tensor_parallel_degree=2 \\
  --activation_checkpoint.mode=full \\
  --checkpoint.enable --checkpoint.interval=5 --checkpoint.no-last-save-model-only \\
  --training.steps=20"
