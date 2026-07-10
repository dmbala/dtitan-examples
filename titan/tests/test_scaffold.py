import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_dirs_exist():
    for d in ("container", "configs", "slurm", "docs", "outputs", "tests"):
        assert (ROOT / d).is_dir(), d


def test_gitignore_ignores_outputs():
    gi = (ROOT / ".gitignore").read_text()
    assert "outputs/*" in gi
    assert "!**/.gitkeep" in gi
