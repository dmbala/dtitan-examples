import pathlib

SLURM = pathlib.Path(__file__).resolve().parent.parent / "slurm"


def test_1node_launcher_shape():
    text = (SLURM / "launch_1node.sbatch").read_text()
    assert "--partition=kempner_h100" in text
    assert "--nodes=1" in text
    assert "--gpus-per-node=4" in text
    assert "--standalone --nproc_per_node=4" in text
    assert "singularity exec --nv" in text


def test_2node_launcher_shape():
    text = (SLURM / "launch_2node.sbatch").read_text()
    assert "--nodes=2" in text
    assert "--nnodes=2 --nproc_per_node=4" in text
    assert "rdzv_backend=c10d" in text
    assert "MASTER_ADDR" in text
