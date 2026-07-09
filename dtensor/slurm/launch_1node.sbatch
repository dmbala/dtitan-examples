#!/bin/bash
#SBATCH --job-name=dtensor-l12
#SBATCH --partition=kempner_h100
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

singularity exec --nv \
  --bind "$(pwd)/data:/data" \
  --bind "$(pwd)/checkpoints:/checkpoints" \
  --bind "$(pwd)/artifacts:/artifacts" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 "$@"
