#!/bin/bash
#SBATCH --job-name=titan-l12
# kempner_rtx (RTX PRO 6000 Blackwell): the torch-2.11 image is CUDA 13.2, which the
# kempner_h100 driver (575.57.08 / CUDA 12.9) cannot run — Blackwell nodes have the 13.2
# driver. See docs/labs/level1/validation.md "Reference Notes".
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
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

singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
    --model.name llama3 --model.flavor debugmodel "$@"
