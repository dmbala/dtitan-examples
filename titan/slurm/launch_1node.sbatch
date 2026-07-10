#!/bin/bash
#SBATCH --job-name=titan-l12
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -d container ] && [ -d slurm ]; then :
elif [ -d titan/slurm ]; then cd titan
fi
mkdir -p outputs
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
    --module llama3 --config llama3_debugmodel "$@"
