# DTensor Workshop — Foundation + Level 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared workshop infrastructure (Python package, test harness, Slurm launchers, preflight) plus a complete, runnable Level 1 module (DeviceMesh, DTensor sharding/redistribution, a debugging lab, profiling, and a capstone).

**Architecture:** A small importable package `dtensor_workshop/` holds reusable primitives (process-group setup, rank-aware logging, synthetic data, device-mesh builders, a multiprocess test harness). Each lab is a thin module in `labs/level1/` exposing a pure, testable function plus a `main()` for `torchrun`. Correctness is TDD'd on CPU with the `gloo` backend via `torch.multiprocessing.spawn` — no GPUs needed for tests — while GPU-only artifacts (profiler traces) are validated by running and by file-existence smoke tests.

**Tech Stack:** Python 3, PyTorch 2.10 (2.11 preview, from the `dtitan.sif` container), `torch.distributed` (DTensor, DeviceMesh, gloo/NCCL), `torch.profiler`, pytest, Slurm, Apptainer/Singularity.

## Global Constraints

- **Container (all commands run inside it):** `IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`. Invoke with `singularity exec "$IMAGE" …`; add `--nv` only for GPU runs.
- **PyTorch:** torch 2.10.0a0 (2.11 preview). Use the **public** DTensor API paths only: `torch.distributed.tensor` and `torch.distributed.device_mesh`. Do **not** use `torch.distributed._tensor`.
- **No torchtitan.** All workshop code is standalone DTensor; do not import `torchtitan`.
- **Do not re-set NCCL env vars** (`TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, `TORCH_NCCL_TRACE_BUFFER_SIZE`) — the container already exports them.
- **Working directory** for all commands is `dtensor/` (the module root). Labs are launched as modules: `torchrun … -m labs.level1.l1_xxx`.
- **Level 1 hardware:** 1 node, 4 GPUs, partition `kempner_h100` (or `kempner_h200`). Slurm account is a placeholder `<account>` (likely `kempner_dev`).
- **Level 1 mesh:** 1D `DeviceMesh` of size 4 (dim name `dp`).
- **Tests are GPU-free:** backend `gloo`, device type `cpu`, launched via the multiprocess harness. Tests must pass on a login/compute node with no GPU visible.
- **Determinism:** synthetic tensors are seeded and identical across ranks (seed must not depend on rank).

---

## File Structure

```
dtensor/
├── conftest.py                     # puts dtensor/ on sys.path for pytest
├── pyproject.toml                  # pytest config + package metadata
├── .gitignore                      # ignore data/checkpoints/artifacts contents, __pycache__
├── dtensor_workshop/
│   ├── __init__.py
│   ├── distenv.py                  # rank/world/local_rank, device type, pg init/shutdown
│   ├── testing.py                  # run_distributed(): mp.spawn gloo harness for tests
│   ├── synth.py                    # deterministic synthetic tensors
│   ├── mesh.py                     # build_mesh() around init_device_mesh
│   └── rlog.py                     # rank-aware logging
├── labs/
│   ├── __init__.py
│   └── level1/
│       ├── __init__.py
│       ├── l1_hello.py             # Lab 1: rank-aware state
│       ├── l1_shard.py             # Lab 2: shard a large tensor
│       ├── l1_redistribute.py      # Lab 3: Shard <-> Replicate
│       ├── l1_shape_bug.py         # Lab 4: failure-driven shape mismatch
│       ├── l1_profile.py           # Lab 5: profiler trace
│       └── l1_capstone.py          # Capstone
├── slurm/
│   ├── launch_1node.sbatch
│   └── launch_2node.sbatch
├── preflight.py                    # environment checks
├── tests/
│   ├── test_distenv.py
│   ├── test_testing.py
│   ├── test_synth.py
│   ├── test_mesh.py
│   ├── test_rlog.py
│   ├── test_preflight.py
│   ├── test_l1_shard.py
│   ├── test_l1_redistribute.py
│   ├── test_l1_shape_bug.py
│   ├── test_l1_profile.py
│   └── test_l1_capstone.py
├── data/.gitkeep
├── checkpoints/.gitkeep
└── artifacts/.gitkeep
```

**Test command (used throughout):**
```bash
singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests -v'
```
For a single test:
```bash
singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_x.py::test_y -v'
```

---

## Task 1: Repo scaffold and package skeleton

**Files:**
- Create: `dtensor/pyproject.toml`
- Create: `dtensor/conftest.py`
- Create: `dtensor/.gitignore`
- Create: `dtensor/dtensor_workshop/__init__.py`
- Create: `dtensor/labs/__init__.py`, `dtensor/labs/level1/__init__.py`
- Create: `dtensor/data/.gitkeep`, `dtensor/checkpoints/.gitkeep`, `dtensor/artifacts/.gitkeep`
- Test: `dtensor/tests/test_scaffold.py`

**Interfaces:**
- Produces: importable package `dtensor_workshop` (empty for now); pytest rootdir at `dtensor/` with `dtensor/` on `sys.path`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_scaffold.py`:
```python
import importlib


def test_package_imports():
    mod = importlib.import_module("dtensor_workshop")
    assert mod is not None


def test_labs_package_imports():
    assert importlib.import_module("labs.level1") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_scaffold.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop'`.

- [ ] **Step 3: Create the scaffold**

`dtensor/conftest.py`:
```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
```

`dtensor/pyproject.toml`:
```toml
[project]
name = "dtensor-workshop"
version = "0.1.0"
description = "Standalone DTensor distributed-training workshop"
requires-python = ">=3.10"

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

`dtensor/.gitignore`:
```gitignore
__pycache__/
*.pyc
data/*
checkpoints/*
artifacts/*
!**/.gitkeep
```

Create empty files: `dtensor/dtensor_workshop/__init__.py`, `dtensor/labs/__init__.py`, `dtensor/labs/level1/__init__.py`, `dtensor/data/.gitkeep`, `dtensor/checkpoints/.gitkeep`, `dtensor/artifacts/.gitkeep`.

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_scaffold.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/pyproject.toml dtensor/conftest.py dtensor/.gitignore \
        dtensor/dtensor_workshop/__init__.py dtensor/labs/__init__.py \
        dtensor/labs/level1/__init__.py dtensor/data/.gitkeep \
        dtensor/checkpoints/.gitkeep dtensor/artifacts/.gitkeep \
        dtensor/tests/test_scaffold.py
git commit -m "Scaffold dtensor workshop package and test harness layout"
```

---

## Task 2: Process-group environment helpers (`distenv.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/distenv.py`
- Test: `dtensor/tests/test_distenv.py`

**Interfaces:**
- Produces:
  - `rank() -> int`, `world_size() -> int`, `local_rank() -> int` (read from env, default 0/1/0)
  - `is_distributed() -> bool`
  - `device_type() -> str` (`"cuda"` if available else `"cpu"`)
  - `init_process_group(backend: str | None = None) -> str` (defaults `nccl` if CUDA else `gloo`; sets CUDA device from `local_rank()`; idempotent)
  - `shutdown() -> None`

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_distenv.py`:
```python
from dtensor_workshop import distenv


def test_defaults_when_unset(monkeypatch):
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    assert distenv.rank() == 0
    assert distenv.world_size() == 1
    assert distenv.local_rank() == 0
    assert distenv.is_distributed() is False


def test_reads_env(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert distenv.rank() == 3
    assert distenv.world_size() == 4
    assert distenv.local_rank() == 3
    assert distenv.is_distributed() is True


def test_device_type_is_cpu_without_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert distenv.device_type() == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_distenv.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.distenv'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/distenv.py`:
```python
import os

import torch
import torch.distributed as dist


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def device_type() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def init_process_group(backend: str | None = None) -> str:
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
    return backend


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_distenv.py -v'`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/distenv.py dtensor/tests/test_distenv.py
git commit -m "Add distenv process-group and rank helpers"
```

---

## Task 3: Multiprocess test harness (`testing.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/testing.py`
- Test: `dtensor/tests/test_testing.py`

**Interfaces:**
- Consumes: `distenv.init_process_group`, `distenv.shutdown` (used by workers in later tests).
- Produces: `run_distributed(worker, world_size: int = 4, args: tuple = ()) -> None`. Spawns `world_size` processes with a free `MASTER_PORT`, backend-agnostic env (`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_ADDR`/`MASTER_PORT`), calls `worker(rank, world_size, *args)` in each. Worker must be a top-level (picklable) function; exceptions/asserts in any worker propagate to the caller.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_testing.py`:
```python
import pytest

from dtensor_workshop import distenv
from dtensor_workshop.testing import run_distributed


def _allreduce_worker(rank, world_size):
    import torch
    import torch.distributed as dist
    distenv.init_process_group("gloo")
    t = torch.tensor([float(rank)])
    dist.all_reduce(t)
    expected = float(sum(range(world_size)))
    assert t.item() == expected, (t.item(), expected)
    distenv.shutdown()


def _failing_worker(rank, world_size):
    distenv.init_process_group("gloo")
    try:
        assert rank != world_size - 1, "boom on last rank"
    finally:
        distenv.shutdown()


def test_run_distributed_allreduce():
    run_distributed(_allreduce_worker, world_size=4)


def test_run_distributed_propagates_failure():
    with pytest.raises(Exception):
        run_distributed(_failing_worker, world_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_testing.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.testing'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/testing.py`:
```python
import os
import socket

import torch.multiprocessing as mp


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _entry(rank, world_size, port, worker, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    worker(rank, world_size, *args)


def run_distributed(worker, world_size: int = 4, args: tuple = ()) -> None:
    port = _free_port()
    mp.spawn(_entry, args=(world_size, port, worker, args),
             nprocs=world_size, join=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_testing.py -v'`
Expected: PASS (2 passed). The `all_reduce` of ranks 0+1+2+3 equals 6.0; the failing-worker test confirms exceptions propagate.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/testing.py dtensor/tests/test_testing.py
git commit -m "Add multiprocess gloo test harness"
```

---

## Task 4: Deterministic synthetic data (`synth.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/synth.py`
- Test: `dtensor/tests/test_synth.py`

**Interfaces:**
- Produces: `synthetic_tensor(shape: tuple[int, ...], seed: int = 0, dtype=torch.float32) -> torch.Tensor` — deterministic for a given `(shape, seed)`, independent of process rank.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_synth.py`:
```python
import torch

from dtensor_workshop import synth


def test_shape_and_dtype():
    t = synth.synthetic_tensor((8, 4))
    assert tuple(t.shape) == (8, 4)
    assert t.dtype == torch.float32


def test_deterministic_same_seed():
    a = synth.synthetic_tensor((16, 3), seed=7)
    b = synth.synthetic_tensor((16, 3), seed=7)
    assert torch.equal(a, b)


def test_different_seed_differs():
    a = synth.synthetic_tensor((16, 3), seed=1)
    b = synth.synthetic_tensor((16, 3), seed=2)
    assert not torch.equal(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_synth.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.synth'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/synth.py`:
```python
import torch


def synthetic_tensor(shape, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=dtype)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_synth.py -v'`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/synth.py dtensor/tests/test_synth.py
git commit -m "Add deterministic synthetic tensor generator"
```

---

## Task 5: Device-mesh builder (`mesh.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/mesh.py`
- Test: `dtensor/tests/test_mesh.py`

**Interfaces:**
- Consumes: `distenv.device_type`, `distenv.init_process_group`/`shutdown` (in tests).
- Produces: `build_mesh(shape, dim_names, device_type: str | None = None) -> DeviceMesh`. Wraps `init_device_mesh`; `device_type` defaults to `distenv.device_type()`. The product of `shape` must equal the world size (enforced by `init_device_mesh`).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_mesh.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed


def _mesh_1d_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    assert mesh.ndim == 1
    assert mesh.size() == world_size
    assert mesh.mesh_dim_names == ("dp",)
    distenv.shutdown()


def _mesh_2d_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    assert mesh.ndim == 2
    assert mesh["dp"].size() == 2
    assert mesh["tp"].size() == 2
    distenv.shutdown()


def test_build_1d_mesh():
    run_distributed(_mesh_1d_worker, world_size=4)


def test_build_2d_mesh():
    run_distributed(_mesh_2d_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_mesh.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.mesh'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/mesh.py`:
```python
from torch.distributed.device_mesh import init_device_mesh

from . import distenv


def build_mesh(shape, dim_names, device_type: str | None = None):
    if device_type is None:
        device_type = distenv.device_type()
    return init_device_mesh(device_type, tuple(shape),
                            mesh_dim_names=tuple(dim_names))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_mesh.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/mesh.py dtensor/tests/test_mesh.py
git commit -m "Add device-mesh builder"
```

---

## Task 6: Rank-aware logging (`rlog.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/rlog.py`
- Test: `dtensor/tests/test_rlog.py`

**Interfaces:**
- Consumes: `distenv.rank`, `distenv.world_size`.
- Produces:
  - `prefix() -> str` returns `"[rank {rank}/{world_size}]"`.
  - `info(msg: str) -> None` prints the prefixed message to stdout.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_rlog.py`:
```python
from dtensor_workshop import rlog


def test_prefix_default(monkeypatch):
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(k, raising=False)
    assert rlog.prefix() == "[rank 0/1]"


def test_prefix_reads_env(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert rlog.prefix() == "[rank 2/4]"


def test_info_prints_prefix(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    rlog.info("hello")
    out = capsys.readouterr().out
    assert "[rank 1/4] hello" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_rlog.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.rlog'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/rlog.py`:
```python
from . import distenv


def prefix() -> str:
    return f"[rank {distenv.rank()}/{distenv.world_size()}]"


def info(msg: str) -> None:
    print(f"{prefix()} {msg}", flush=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_rlog.py -v'`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/rlog.py dtensor/tests/test_rlog.py
git commit -m "Add rank-aware logging"
```

---

## Task 7: Slurm launchers

**Files:**
- Create: `dtensor/slurm/launch_1node.sbatch`
- Create: `dtensor/slurm/launch_2node.sbatch`
- Test: `dtensor/tests/test_launchers.py`

**Interfaces:**
- Produces: two `sbatch` scripts. Both `cd` into the repo `dtensor/` dir (via `SLURM_SUBMIT_DIR`), define `IMAGE`, bind `data`/`checkpoints`/`artifacts`, and pass `"$@"` (e.g. `-m labs.level1.l1_shard`) to `torchrun`. The 1-node script uses `--standalone --nproc_per_node=4`; the 2-node script uses c10d rendezvous with `--nnodes=2 --nproc_per_node=4`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_launchers.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_launchers.py -v'`
Expected: FAIL — `FileNotFoundError` for `launch_1node.sbatch`.

- [ ] **Step 3: Write the launchers**

`dtensor/slurm/launch_1node.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=dtensor-l12
#SBATCH --partition=kempner_h100
#SBATCH --account=<account>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

singularity exec --nv \
  --bind "$(pwd)/data:/data" \
  --bind "$(pwd)/checkpoints:/checkpoints" \
  --bind "$(pwd)/artifacts:/artifacts" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 "$@"
```

`dtensor/slurm/launch_2node.sbatch`:
```bash
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_launchers.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/slurm/launch_1node.sbatch dtensor/slurm/launch_2node.sbatch dtensor/tests/test_launchers.py
git commit -m "Add 1-node and 2-node Slurm launchers"
```

---

## Task 8: Preflight checks (`preflight.py`)

**Files:**
- Create: `dtensor/preflight.py`
- Test: `dtensor/tests/test_preflight.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (self-contained so it can run before the package is trusted).
- Produces:
  - `check_torch_import() -> tuple[str, bool, str]`
  - `check_dtensor_import() -> tuple[str, bool, str]`
  - `check_dir_writable(path: str) -> tuple[str, bool, str]`
  - `check_gpu_visible() -> tuple[str, bool, str]` (CUDA device count > 0)
  - `run_cpu_checks(dirs: list[str]) -> list[tuple[str, bool, str]]` (the GPU-free subset)
  - `main() -> int` runs all checks, prints `PASS`/`FAIL` per check, returns non-zero if any fail. Each check returns `(name, ok, detail)`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_preflight.py`:
```python
import preflight


def test_torch_import_ok():
    name, ok, _ = preflight.check_torch_import()
    assert name == "torch import"
    assert ok is True


def test_dtensor_import_ok():
    _, ok, _ = preflight.check_dtensor_import()
    assert ok is True


def test_dir_writable(tmp_path):
    _, ok, _ = preflight.check_dir_writable(str(tmp_path))
    assert ok is True


def test_dir_not_writable():
    _, ok, _ = preflight.check_dir_writable("/nonexistent/path/xyz")
    assert ok is False


def test_run_cpu_checks_returns_rows(tmp_path):
    rows = preflight.run_cpu_checks([str(tmp_path)])
    assert all(len(r) == 3 for r in rows)
    assert any(r[0] == "torch import" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_preflight.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'preflight'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/preflight.py`:
```python
import os
import sys
import uuid


def check_torch_import():
    try:
        import torch
        return ("torch import", True, torch.__version__)
    except Exception as exc:  # noqa: BLE001
        return ("torch import", False, repr(exc))


def check_dtensor_import():
    try:
        from torch.distributed.tensor import Shard  # noqa: F401
        from torch.distributed.device_mesh import init_device_mesh  # noqa: F401
        return ("dtensor import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("dtensor import", False, repr(exc))


def check_dir_writable(path):
    try:
        probe = os.path.join(path, f".preflight_{uuid.uuid4().hex}")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return (f"writable:{path}", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return (f"writable:{path}", False, repr(exc))


def check_gpu_visible():
    try:
        import torch
        n = torch.cuda.device_count()
        return ("gpu visible", n > 0, f"device_count={n}")
    except Exception as exc:  # noqa: BLE001
        return ("gpu visible", False, repr(exc))


def run_cpu_checks(dirs):
    rows = [check_torch_import(), check_dtensor_import()]
    rows += [check_dir_writable(d) for d in dirs]
    return rows


def main():
    dirs = ["/data", "/checkpoints", "/artifacts"]
    rows = run_cpu_checks(dirs)
    rows.append(check_gpu_visible())
    failed = 0
    for name, ok, detail in rows:
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{status}] {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_preflight.py -v'`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/preflight.py dtensor/tests/test_preflight.py
git commit -m "Add preflight environment checks"
```

---

## Task 9: Level 1 Lab 1 — rank-aware hello (`l1_hello.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_hello.py`
- Test: (covered by earlier `rlog`/`distenv` tests; this lab is a `main()`-only demo — smoke-run documented, no new unit test)

**Interfaces:**
- Consumes: `distenv`, `rlog`.
- Produces: `main()` that initializes the process group, logs rank-aware state, barriers, and shuts down.

- [ ] **Step 1: Write the lab**

`dtensor/labs/level1/l1_hello.py`:
```python
import torch
import torch.distributed as dist

from dtensor_workshop import distenv, rlog


def main():
    distenv.init_process_group()
    rlog.info(
        f"world_size={distenv.world_size()} local_rank={distenv.local_rank()} "
        f"cuda={torch.cuda.is_available()}"
    )
    dist.barrier()
    if distenv.rank() == 0:
        rlog.info("all ranks reached the barrier")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on CPU (no GPU needed)**

Run:
```bash
singularity exec "$IMAGE" bash -lc 'cd dtensor && torchrun --standalone --nproc_per_node=4 -m labs.level1.l1_hello'
```
Expected: four `[rank i/4] world_size=4 …` lines plus one `all ranks reached the barrier`.

**Success criterion (spec):** 4 distinct ranks report `world_size=4` and the correct `local_rank`.

- [ ] **Step 3: Commit**

```bash
git add dtensor/labs/level1/l1_hello.py
git commit -m "Add Level 1 Lab 1: rank-aware hello"
```

---

## Task 10: Level 1 Lab 2 — shard a large tensor (`l1_shard.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_shard.py`
- Test: `dtensor/tests/test_l1_shard.py`

**Interfaces:**
- Consumes: `distenv`, `mesh.build_mesh`, `synth.synthetic_tensor`, `rlog`.
- Produces:
  - `shard_report(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> dict` with keys `local_shape` and `global_shape` (tuples). Shards a `(rows, cols)` tensor along dim 0.
  - `main()` for `torchrun`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l1_shard.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_shard


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    rep = l1_shard.shard_report(mesh, rows=1024, cols=8)
    assert rep["global_shape"] == (1024, 8)
    assert rep["local_shape"] == (1024 // world_size, 8)
    distenv.shutdown()


def test_shard_report():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_shard.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l1_shard'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level1/l1_shard.py`:
```python
from torch.distributed.tensor import Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def shard_report(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> dict:
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    dtensor = distribute_tensor(full, mesh, [Shard(0)])
    return {
        "local_shape": tuple(dtensor.to_local().shape),
        "global_shape": tuple(dtensor.shape),
    }


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    rep = shard_report(mesh)
    rlog.info(f"local={rep['local_shape']} global={rep['global_shape']}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_shard.py -v'`
Expected: PASS (1 passed). Local shard is `(256, 8)` for world_size 4; global is `(1024, 8)`.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level1/l1_shard.py dtensor/tests/test_l1_shard.py
git commit -m "Add Level 1 Lab 2: shard a large tensor"
```

---

## Task 11: Level 1 Lab 3 — redistribute Shard ↔ Replicate (`l1_redistribute.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_redistribute.py`
- Test: `dtensor/tests/test_l1_redistribute.py`

**Interfaces:**
- Consumes: `distenv`, `mesh.build_mesh`, `synth`, `rlog`.
- Produces:
  - `replicate_max_diff(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> float` — shards along dim 0, redistributes to `Replicate()`, returns the max absolute difference between the replicated local tensor and the original full tensor (expected `0.0`).
  - `main()` for `torchrun`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l1_redistribute.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_redistribute


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    diff = l1_redistribute.replicate_max_diff(mesh, rows=1024, cols=8)
    assert diff == 0.0, diff
    distenv.shutdown()


def test_replicate_max_diff_is_zero():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_redistribute.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l1_redistribute'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level1/l1_redistribute.py`:
```python
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def replicate_max_diff(mesh, rows: int = 1024, cols: int = 8, seed: int = 0) -> float:
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    replicated = sharded.redistribute(mesh, [Replicate()])
    return (replicated.to_local() - full).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    diff = replicate_max_diff(mesh)
    rlog.info(f"replicate max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_redistribute.py -v'`
Expected: PASS (1 passed). After all-gather each rank holds the full tensor, so the diff is exactly 0.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level1/l1_redistribute.py dtensor/tests/test_l1_redistribute.py
git commit -m "Add Level 1 Lab 3: redistribute shard to replicate"
```

---

## Task 12: Level 1 Lab 4 — failure-driven shape mismatch (`l1_shape_bug.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_shape_bug.py`
- Test: `dtensor/tests/test_l1_shape_bug.py`

**Interfaces:**
- Consumes: `distenv`, `mesh.build_mesh`, `synth`.
- Produces:
  - `buggy_matmul(mesh) -> None` — builds two sharded tensors whose contracting dimensions do not agree and attempts a matmul; raises `RuntimeError` (the teaching bug).
  - `fixed_matmul(mesh) -> tuple[int, ...]` — the corrected version; returns the global output shape.
  - `main()` runs the fixed version and logs success; a `--bug` flag runs the buggy version to show the error.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l1_shape_bug.py`:
```python
import pytest

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_shape_bug


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    with pytest.raises(RuntimeError):
        l1_shape_bug.buggy_matmul(mesh)
    out = l1_shape_bug.fixed_matmul(mesh)
    assert out == (256, 256)
    distenv.shutdown()


def test_bug_raises_and_fix_works():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_shape_bug.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l1_shape_bug'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level1/l1_shape_bug.py`:
```python
import sys

from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def buggy_matmul(mesh):
    # BUG: left operand is (256, 128), right operand is (64, 256) -> the
    # contracting dims (128 vs 64) do not match, so matmul raises.
    left = distribute_tensor(synth.synthetic_tensor((256, 128), seed=1), mesh, [Shard(0)])
    right = distribute_tensor(synth.synthetic_tensor((64, 256), seed=2), mesh, [Replicate()])
    return (left @ right).to_local()


def fixed_matmul(mesh):
    # FIX: make the contracting dims agree (128 == 128).
    left = distribute_tensor(synth.synthetic_tensor((256, 128), seed=1), mesh, [Shard(0)])
    right = distribute_tensor(synth.synthetic_tensor((128, 256), seed=2), mesh, [Replicate()])
    out = left @ right
    return tuple(out.shape)


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    if "--bug" in sys.argv:
        buggy_matmul(mesh)
    else:
        rlog.info(f"fixed matmul global shape = {fixed_matmul(mesh)}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_shape_bug.py -v'`
Expected: PASS (1 passed). The buggy matmul raises `RuntimeError` (shape mismatch); the fixed one yields global shape `(256, 256)`.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level1/l1_shape_bug.py dtensor/tests/test_l1_shape_bug.py
git commit -m "Add Level 1 Lab 4: failure-driven shape mismatch"
```

---

## Task 13: Level 1 Lab 5 — profiler trace (`l1_profile.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_profile.py`
- Test: `dtensor/tests/test_l1_profile.py`

**Interfaces:**
- Consumes: `distenv`, `mesh.build_mesh`, `synth`, `rlog`.
- Produces:
  - `profiled_run(mesh, out_path: str) -> int` — runs elementwise + a redistribute (all-gather) under `torch.profiler.profile`, exports a chrome trace to `out_path`, and returns the number of profiled events (`len(prof.key_averages())`).
  - `main()` writes to `artifacts/l1_trace.json`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l1_profile.py`:
```python
import os

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_profile


def _worker(rank, world_size, out_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    out_path = os.path.join(out_dir, f"trace_rank{rank}.json")
    n_events = l1_profile.profiled_run(mesh, out_path)
    assert n_events > 0
    if rank == 0:
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
    distenv.shutdown()


def test_profiled_run(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_profile.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l1_profile'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level1/l1_profile.py`:
```python
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def profiled_run(mesh, out_path: str) -> int:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    sharded = distribute_tensor(synth.synthetic_tensor((4096, 32)), mesh, [Shard(0)])
    with profile(activities=activities) as prof:
        scaled = sharded * 2.0
        gathered = scaled.redistribute(mesh, [Replicate()])  # all-gather = comm op
        _ = gathered.to_local().sum()
    prof.export_chrome_trace(out_path)
    return len(prof.key_averages())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    n = profiled_run(mesh, "artifacts/l1_trace.json")
    rlog.info(f"exported trace with {n} profiled event rows to artifacts/l1_trace.json")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_profile.py -v'`
Expected: PASS (1 passed). Trace file is created and non-empty; at least one profiled event row exists.

**Success criterion (spec, GPU run):** on a real GPU node the exported trace shows both compute and a communication op (e.g. all-gather). Verify by loading `artifacts/l1_trace.json` in `chrome://tracing` or `perfetto`, or inspecting `prof.key_averages()` for a collective op name.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level1/l1_profile.py dtensor/tests/test_l1_profile.py
git commit -m "Add Level 1 Lab 5: profiler trace"
```

---

## Task 14: Level 1 Capstone (`l1_capstone.py`)

**Files:**
- Create: `dtensor/labs/level1/l1_capstone.py`
- Test: `dtensor/tests/test_l1_capstone.py`

**Interfaces:**
- Consumes: `distenv`, `mesh.build_mesh`, `synth`, `rlog`.
- Produces:
  - `distributed_global_sum(mesh, rows: int = 2048, cols: int = 16, seed: int = 0) -> tuple[float, float]` — shards a tensor, computes `y = x*2 + 1`, reduces `y.sum()` (a `Partial`) to `Replicate()`, and returns `(distributed_sum, single_device_reference_sum)`.
  - `main()` runs the capstone, validates parity, and exports `artifacts/l1_capstone_trace.json`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l1_capstone.py`:
```python
import math

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level1 import l1_capstone


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    dist_sum, ref_sum = l1_capstone.distributed_global_sum(mesh)
    assert math.isclose(dist_sum, ref_sum, rel_tol=1e-4, abs_tol=1e-3), (dist_sum, ref_sum)
    distenv.shutdown()


def test_global_sum_matches_reference():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_capstone.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l1_capstone'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level1/l1_capstone.py`:
```python
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog, synth


def distributed_global_sum(mesh, rows: int = 2048, cols: int = 16, seed: int = 0):
    full = synth.synthetic_tensor((rows, cols), seed=seed)
    sharded = distribute_tensor(full, mesh, [Shard(0)])
    result = (sharded * 2.0 + 1.0).sum()          # Partial placement
    replicated = result.redistribute(mesh, [Replicate()])  # all-reduce
    dist_sum = replicated.to_local().item()
    ref_sum = (full * 2.0 + 1.0).sum().item()
    return dist_sum, ref_sum


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        dist_sum, ref_sum = distributed_global_sum(mesh)
    prof.export_chrome_trace("artifacts/l1_capstone_trace.json")
    ok = abs(dist_sum - ref_sum) <= 1e-3 + 1e-4 * abs(ref_sum)
    rlog.info(f"distributed_sum={dist_sum:.4f} reference={ref_sum:.4f} parity_ok={ok}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l1_capstone.py -v'`
Expected: PASS (1 passed). The distributed global sum matches the single-device reference within tolerance.

- [ ] **Step 5: Run the full Level 1 suite and smoke-run the capstone**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests -v'`
Expected: all tests PASS.

Optional GPU smoke run (on a `kempner_h100` node):
```bash
cd dtensor && sbatch slurm/launch_1node.sbatch -m labs.level1.l1_capstone
```
Expected: log line `parity_ok=True` and `artifacts/l1_capstone_trace.json` created.

- [ ] **Step 6: Commit**

```bash
git add dtensor/labs/level1/l1_capstone.py dtensor/tests/test_l1_capstone.py
git commit -m "Add Level 1 capstone: distributed global sum with parity check"
```

---

## Self-Review

**1. Spec coverage (Foundation + Level 1 scope):**

| Spec item | Task |
| --- | --- |
| Container invocation / bind mounts | Task 7 (launchers), Global Constraints |
| Directory layout (`data`/`checkpoints`/`artifacts`) | Task 1 |
| 1-node & 2-node Slurm launchers | Task 7 |
| Preflight checks | Task 8 (CPU subset tested; GPU/NCCL checks run on node) |
| Running example — L1 raw tensors + linear | Tasks 10–14 (tensor ops; a linear-style matmul appears in Task 12/14) |
| L1 Lab 1 rank-aware state | Task 9 |
| L1 Lab 2 shard a tensor | Task 10 |
| L1 Lab 3 redistribute Shard↔Replicate | Task 11 |
| L1 Lab 4 failure-driven shape mismatch | Task 12 |
| L1 Lab 5 profiler trace | Task 13 |
| L1 capstone (mesh, shard, ops, validate, trace) | Task 14 |
| DTensor placements Shard/Replicate/Partial | Task 10 (Shard), 11 (Replicate), 14 (Partial via `.sum()`) |
| Rank-aware logging | Task 6 |
| Milestone order correct → observable | Task ordering (correctness labs before profiling) |

Deferred to later plans (out of scope here, by design): Level 2 (2D mesh, TP, DCP, memory/OOM) and Level 3 (FSDP2, MoE, FP8, NCCL debugging), the shared Transformer block model, reference artifacts, and the assessment rubric.

**2. Placeholder scan:** `<account>` in launchers is an intentional, spec-mandated placeholder (documented in Global Constraints), not a plan gap. No `TBD`/`TODO`/"handle edge cases"/"similar to Task N" present; every code step shows complete code.

**3. Type consistency:** Names are stable across tasks — `distenv.init_process_group`/`shutdown`/`rank`/`world_size`/`local_rank`/`device_type`, `build_mesh(shape, dim_names, device_type=None)`, `synthetic_tensor(shape, seed, dtype)`, `run_distributed(worker, world_size, args)`, `rlog.prefix`/`info`. Lab functions (`shard_report`, `replicate_max_diff`, `buggy_matmul`/`fixed_matmul`, `profiled_run`, `distributed_global_sum`) are each referenced with matching signatures in their tests. DTensor imports use the public `torch.distributed.tensor` / `torch.distributed.device_mesh` paths throughout, per Global Constraints.
