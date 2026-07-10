#!/bin/bash
#SBATCH --job-name=dtensor-l3
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail
# Work from the workshop root (the dir containing dtensor_workshop/), whether this
# job was submitted from dtensor/ or from the repo root.
cd "$SLURM_SUBMIT_DIR"
if [ -d dtensor_workshop ]; then :
elif [ -d dtensor/dtensor_workshop ]; then cd dtensor
else echo "ERROR: dtensor_workshop/ not found under $SLURM_SUBMIT_DIR" >&2; exit 1; fi
mkdir -p data checkpoints artifacts
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500

srun --cpu-bind=none singularity exec --nv \
  --bind "$(pwd)/data:/data" \
  --bind "$(pwd)/checkpoints:/checkpoints" \
  --bind "$(pwd)/artifacts:/artifacts" \
  "$IMAGE" \
  torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    "$@"
