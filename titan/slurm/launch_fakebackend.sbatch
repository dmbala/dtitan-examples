#!/bin/bash
#SBATCH --job-name=titan-fakebackend
# Fake-backend dry-run (Level 1 Lab 3): validates config + parallelism plan WITHOUT real
# collectives. torchtitan reads NGPU as the simulated world size and LOCAL_RANK for the
# device, so it must launch via torchrun (a bare `python -m torchtitan.train` fails
# KeyError: LOCAL_RANK). One real GPU is enough; NGPU is the *simulated* rank count.
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/%x-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -d container ] && [ -d slurm ]; then :
elif [ -d titan/slurm ]; then cd titan
else echo "ERROR: run from the titan/ dir or repo root" >&2; exit 1
fi
mkdir -p outputs
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif

export SINGULARITYENV_HF_HOME="$(pwd)/outputs/hf_cache"
export SINGULARITYENV_HF_DATASETS_CACHE="$(pwd)/outputs/hf_cache/datasets"
mkdir -p "$(pwd)/outputs/hf_cache/datasets"
export SINGULARITYENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
# NGPU = simulated world size for the fake backend (override by exporting NGPU before sbatch).
export SINGULARITYENV_NGPU="${NGPU:-4}"

singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=1 -m torchtitan.train \
    --model.name llama3 --model.flavor debugmodel --comm.mode=fake_backend "$@"
