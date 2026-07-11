#!/bin/bash
#SBATCH --job-name=titan-l3
# kempner_rtx (RTX PRO 6000 Blackwell) has the CUDA-13.2 driver the torch-2.11 image needs
# (kempner_h100's CUDA-12.9 driver cannot run it). NOTE: RTX nodes have 8 GPUs/node, so a
# world size of 8 fits on ONE node — this 2-node launcher is for Level 3's larger topologies.
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -d container ] && [ -d slurm ]; then :
elif [ -d titan/slurm ]; then cd titan
else echo "ERROR: run from the titan/ dir or repo root" >&2; exit 1
fi
mkdir -p outputs
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif

# torchtitan's dataset loader writes an Arrow cache; the image's HF_HOME (/data/hf_cache)
# is read-only at runtime, so point HF caches at the (writable) outputs dir.
export SINGULARITYENV_HF_HOME="$(pwd)/outputs/hf_cache"
export SINGULARITYENV_HF_DATASETS_CACHE="$(pwd)/outputs/hf_cache/datasets"
mkdir -p "$(pwd)/outputs/hf_cache/datasets"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500

srun --cpu-bind=none singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" -m torchtitan.train \
    --model.name llama3 --model.flavor debugmodel "$@"
