import pathlib

SLURM = pathlib.Path(__file__).resolve().parent.parent / "slurm"


def test_1node():
    t = (SLURM / "launch_1node.sbatch").read_text()
    assert "--account=kempner_dev" in t
    # torch-2.11 image is CUDA 13.2 -> runs on kempner_rtx (Blackwell), not kempner_h100
    assert "--partition=kempner_rtx" in t
    assert "dtitan-torch211.sif" in t
    assert "--mem=" in t
    assert "--output=outputs/" in t
    assert "--standalone --nproc_per_node=4" in t
    assert "-m torchtitan.train" in t
    assert "--model.name llama3 --model.flavor debugmodel" in t
    assert "outputs:/outputs" in t
    assert "TRITON_LIBCUDA_PATH" in t  # MoE grouped-GEMM Triton kernel needs libcuda.so


def test_8gpu():
    t = (SLURM / "launch_8gpu.sbatch").read_text()
    assert "--account=kempner_dev" in t
    assert "--partition=kempner_rtx" in t
    assert "dtitan-torch211.sif" in t
    assert "--gpus-per-node=8" in t
    assert "--standalone --nproc_per_node=8" in t
    assert "-m torchtitan.train" in t
    assert "TRITON_LIBCUDA_PATH" in t  # MoE grouped-GEMM
    assert "TORCH_FR_BUFFER_SIZE" in t  # Flight Recorder (Level 3)


def test_2node():
    t = (SLURM / "launch_2node.sbatch").read_text()
    assert "--account=kempner_dev" in t
    assert "--partition=kempner_rtx" in t
    assert "dtitan-torch211.sif" in t
    assert "--mem=" in t
    assert "--output=outputs/" in t
    assert "--nodes=2" in t
    assert "srun --cpu-bind=none" in t
    assert "--nnodes=2 --nproc_per_node=4" in t
    assert "rdzv_backend=c10d" in t
    assert "-m torchtitan.train" in t
