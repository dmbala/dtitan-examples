#!/bin/bash
# Level 1 — Lab 3: Fake-Backend Dry-Run (verify config + model init without real multi-GPU comms)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_fakebackend.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=5
