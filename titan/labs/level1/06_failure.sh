#!/bin/bash
# Level 1 — Lab 6: Failure-Driven Debugging (intentionally invalid config)
# NOTE: this job is SUPPOSED to fail — tensor_parallel_degree=3 does not divide 4 GPUs,
# so the job log will show a clean, readable AssertionError (see docs/labs/level1/README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
exec sbatch slurm/launch_1node.sbatch \
  --model.hf_assets_path=assets/test_tokenizer \
  --training.dataset=c4_test --training.dataset_path=assets/c4_subset \
  --training.steps=20 --parallelism.tensor_parallel_degree=3
