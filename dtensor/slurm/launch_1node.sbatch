#!/bin/bash
#SBATCH --job-name=dtensor-l12
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
# Work from the workshop root (the dir containing dtensor_workshop/), whether this
# job was submitted from dtensor/ or from the repo root.
cd "$SLURM_SUBMIT_DIR"
if [ -d dtensor_workshop ]; then :
elif [ -d dtensor/dtensor_workshop ]; then cd dtensor
else echo "ERROR: dtensor_workshop/ not found under $SLURM_SUBMIT_DIR" >&2; exit 1; fi
mkdir -p data checkpoints artifacts
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

singularity exec --nv \
  --bind "$(pwd)/data:/data" \
  --bind "$(pwd)/checkpoints:/checkpoints" \
  --bind "$(pwd)/artifacts:/artifacts" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 "$@"
