import pathlib

SLURM = pathlib.Path(__file__).resolve().parent.parent / "slurm"


def test_1node():
    t = (SLURM / "launch_1node.sbatch").read_text()
    assert "--account=kempner_dev" in t
    assert "--partition=kempner_h100" in t
    assert "--mem=" in t
    assert "--standalone --nproc_per_node=4" in t
    assert "-m torchtitan.train" in t
    assert "--module llama3 --config llama3_debugmodel" in t
    assert "outputs:/outputs" in t


def test_2node():
    t = (SLURM / "launch_2node.sbatch").read_text()
    assert "--nodes=2" in t
    assert "srun --cpu-bind=none" in t
    assert "--nnodes=2 --nproc_per_node=4" in t
    assert "rdzv_backend=c10d" in t
    assert "-m torchtitan.train" in t
