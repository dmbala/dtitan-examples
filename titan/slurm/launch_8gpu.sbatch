#!/bin/bash
#SBATCH --job-name=titan-l3
# kempner_rtx (RTX PRO 6000 Blackwell) has 8 GPUs/node, so Level 3's 8-GPU 3D meshes
# (HSDP+TP, MoE+EP, pipeline) fit on ONE node. The torch-2.11 image is CUDA 13.2, which
# the kempner_h100 driver (CUDA 12.9) cannot run — Blackwell has the 13.2 driver.
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
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
# MoE grouped-GEMM uses a Triton kernel whose libcuda_dirs() needs libcuda.so; point it at
# the --nv-injected driver dir (harmless for non-MoE runs).
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
# Flight Recorder (Level 3 debugging lab) — torch>=2.11 var (the image's TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated).
export SINGULARITYENV_TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-20971520}"

singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=8 -m torchtitan.train \
    --model.name llama3 --model.flavor debugmodel "$@"
