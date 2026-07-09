#!/bin/bash
#SBATCH --job-name=dtensor-l3
#SBATCH --partition=kempner_h100
#SBATCH --account=<account>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500

srun singularity exec --nv \
  --bind "$(pwd)/data:/data" \
  --bind "$(pwd)/checkpoints:/checkpoints" \
  --bind "$(pwd)/artifacts:/artifacts" \
  "$IMAGE" \
  torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    "$@"
