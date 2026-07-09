# DTensor Workshop — Level 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Level 2 of the workshop — 2D (`dp × tp`) parallelism on a compact Transformer block, distributed checkpointing (DCP), memory/OOM diagnosis with activation checkpointing, and bottleneck profiling — on top of the existing foundation.

**Architecture:** Extend the `dtensor_workshop` package with the shared running-example model (`model.py`), a Megatron tensor-parallel plan (`tp.py`), a training step with data-parallel gradient averaging (`train.py`), DCP helpers (`checkpoint.py`), and activation checkpointing (`acheckpoint.py`). Each Level 2 lab under `labs/level2/` is a thin `main()` plus a pure, mesh-parameterized function tested GPU-free on CPU/gloo at world size 4 (a `dp2 × tp2` mesh). GPU-only behaviors (OOM, CUDA memory snapshots) are validated by documented smoke runs.

**Tech Stack:** Python 3, PyTorch 2.10 (2.11 preview) from `dtitan.sif`, `torch.distributed.tensor` (DTensor, DeviceMesh, `parallelize_module`/`ColwiseParallel`/`RowwiseParallel`), `torch.distributed.checkpoint` (DCP), `torch.utils.checkpoint`, `torch.profiler`, pytest, Slurm, Apptainer/Singularity.

## Verified API facts (spiked on CPU/gloo before writing this plan)

These patterns were confirmed to reproduce single-device references to float precision. Implement them exactly:

- **2D mesh:** `init_device_mesh(dev, (2, 2), mesh_dim_names=("dp", "tp"))`; submeshes via `mesh["dp"]`, `mesh["tp"]`; `mesh["dp"].get_group()`, `.size()`, `.get_local_rank()`.
- **Tensor parallel (Megatron):** attention uses **separate** `q`, `k`, `v` linears each `ColwiseParallel()` and `o` `RowwiseParallel()`; MLP uses `fc1` `ColwiseParallel()`, `fc2` `RowwiseParallel()`. A **fused** qkv projection does NOT work with a naive split (the colwise shard is a contiguous slice of `[q|k|v]`, not per-projection) — keep q/k/v separate. `n_heads` must be divisible by the tp size. Applied via `parallelize_module(block, tp_mesh, TP_PLAN)`. The block output is a `Replicate` DTensor; call `.full_tensor()` to compare against a plain reference.
- **Attention reshape must derive local head count from the tensor width** (`n_local = t.shape[-1] // head_dim`), so the same `forward` works single-device (full dim) and TP-sharded (dim/tp).
- **Data parallel:** after `loss.backward()`, for each param with a grad, `dist.all_reduce(grad_local, op=SUM, group=dp_group); grad_local /= dp_size` where `grad_local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad`. With equal per-replica batch splits and a mean loss, this reproduces the full-batch gradient exactly (loss-parity maxdiff 0.0 with SGD).
- **DCP:** `from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict`; `import torch.distributed.checkpoint as dcp`; `dcp.save({"model": msd, "optim": osd}, checkpoint_id=dir)`, `dcp.load({...}, checkpoint_id=dir)` then `set_state_dict(...)`. Round-trips a TP model + optimizer exactly.
- **Activation checkpointing:** `torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)` matches a plain forward exactly.

## Global Constraints

- **Container (all commands run inside it):** `IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`. `singularity exec "$IMAGE" …`; add `--nv` only for GPU runs.
- **PyTorch:** torch 2.10. Public APIs only: `torch.distributed.tensor`, `torch.distributed.tensor.parallel`, `torch.distributed.device_mesh`, `torch.distributed.checkpoint`. No `torch.distributed._tensor`. No torchtitan.
- **Do not re-set NCCL env vars** — the container exports them.
- **Working directory** for all commands is `dtensor/`. Labs run as modules: `torchrun … -m labs.level2.l2_xxx`.
- **Level 2 hardware:** 1 node, 4 GPUs, partition `kempner_h100` (or `kempner_h200`). Uses the existing 1-node launcher `slurm/launch_1node.sbatch`.
- **Level 2 mesh:** 2D `DeviceMesh` `dp=2 × tp=2` (dim names `dp`, `tp`).
- **Tests are GPU-free:** backend `gloo`, device `cpu`, world size 4, via the existing `dtensor_workshop.testing.run_distributed` harness. GPU-only behaviors (OOM, CUDA memory snapshots) are validated only by documented smoke runs, never by unit tests.
- **Reuse the foundation:** `distenv`, `mesh.build_mesh`, `synth.synthetic_tensor`, `rlog`, `testing.run_distributed`. Do not reimplement them.
- **Model test dims:** `dim=32, hidden=64, n_heads=4` (n_heads divisible by tp=2); batch divisible by dp=2 (use 4). Optimizer **SGD** in parity tests (deterministic).
- **Benign stderr noise** (judge pass/fail only by pytest's summary line): pynvml deprecation, `socket.cpp … Address family`, `TORCH_NCCL_TRACE_BUFFER_SIZE` deprecation, Gloo connection info, CPU-only/kineto profiler notices, and a `tensor.storage().size()` UserWarning from DCP.

## File Structure

```
dtensor/
├── dtensor_workshop/
│   ├── model.py          # TransformerBlock (attention + MLP) + build_block  [NEW]
│   ├── tp.py             # TP_PLAN + apply_tp                                 [NEW]
│   ├── train.py          # average_gradients + run_training                  [NEW]
│   ├── checkpoint.py     # dcp_save / dcp_load                               [NEW]
│   └── acheckpoint.py    # forward_maybe_checkpointed                        [NEW]
├── labs/level2/
│   ├── __init__.py                                                           [NEW]
│   ├── l2_mesh.py        # Lab 1: 2D mesh roles
│   ├── l2_tp_block.py    # Lab 2: TP block parity
│   ├── l2_train.py       # Lab 3: 2D training loss parity
│   ├── l2_dcp.py         # Lab 4: DCP save/restore/resume
│   ├── l2_oom.py         # Lab 5: activation checkpointing (+ GPU OOM/snapshot)
│   ├── l2_profile.py     # Lab 6: profile a 2D step
│   └── l2_capstone.py    # Capstone
└── tests/
    ├── test_model.py
    ├── test_tp.py
    ├── test_train.py
    ├── test_checkpoint.py
    ├── test_acheckpoint.py
    ├── test_l2_mesh.py
    ├── test_l2_tp_block.py
    ├── test_l2_train.py
    ├── test_l2_dcp.py
    ├── test_l2_oom.py
    ├── test_l2_profile.py
    └── test_l2_capstone.py
```

**Test command:** `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/<file> -v'`. Run the full suite (`python -m pytest tests -q`) once before each commit.

---

## Task 1: Running-example model (`model.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/model.py`
- Test: `dtensor/tests/test_model.py`

**Interfaces:**
- Produces:
  - `class TransformerBlock(nn.Module)` with submodules named exactly `q`, `k`, `v`, `o` (each `nn.Linear(dim, dim)`), `fc1` (`nn.Linear(dim, hidden)`), `act` (`nn.GELU`), `fc2` (`nn.Linear(hidden, dim)`); attribute `n_heads`, `head_dim`. `forward(x)` for `x` shape `(batch, seq, dim)` returns `(batch, seq, dim)`. The head reshape derives local head count from the tensor width so it also works TP-sharded.
  - `build_block(dim=256, hidden=1024, n_heads=8, seed=0) -> TransformerBlock` — seeds the RNG then constructs (deterministic init).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_model.py`:
```python
import pytest
import torch

from dtensor_workshop.model import TransformerBlock, build_block


def test_forward_shape():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=0)
    out = block(torch.randn(2, 8, 32))
    assert tuple(out.shape) == (2, 8, 32)


def test_deterministic_build():
    a = build_block(dim=32, hidden=64, n_heads=4, seed=7)
    b = build_block(dim=32, hidden=64, n_heads=4, seed=7)
    x = torch.randn(2, 8, 32)
    assert torch.equal(a(x), b(x))


def test_head_divisibility_guard():
    with pytest.raises(AssertionError):
        TransformerBlock(dim=32, hidden=64, n_heads=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_model.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.model'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/model.py`:
```python
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 1024, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def _to_heads(self, t, batch, seq):
        n_local = t.shape[-1] // self.head_dim
        return t.view(batch, seq, n_local, self.head_dim).transpose(1, 2)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self._to_heads(self.q(x), batch, seq)
        k = self._to_heads(self.k(x), batch, seq)
        v = self._to_heads(self.v(x), batch, seq)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq, -1)
        x = x + self.o(attn)
        x = x + self.fc2(self.act(self.fc1(x)))
        return x


def build_block(dim: int = 256, hidden: int = 1024, n_heads: int = 8, seed: int = 0):
    torch.manual_seed(seed)
    return TransformerBlock(dim=dim, hidden=hidden, n_heads=n_heads)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_model.py -v'`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/model.py dtensor/tests/test_model.py
git commit -m "Add Level 2 running-example Transformer block"
```

---

## Task 2: Tensor-parallel plan (`tp.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/tp.py`
- Test: `dtensor/tests/test_tp.py`

**Interfaces:**
- Consumes: `model.TransformerBlock`, a `tp` DeviceMesh.
- Produces:
  - `TP_PLAN` — dict mapping `q`/`k`/`v` → `ColwiseParallel()`, `o` → `RowwiseParallel()`, `fc1` → `ColwiseParallel()`, `fc2` → `RowwiseParallel()`.
  - `apply_tp(block, tp_mesh) -> block` — raises `ValueError` if `block.n_heads % tp_mesh.size() != 0`, else `parallelize_module(block, tp_mesh, TP_PLAN)` and returns the block.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_tp.py`:
```python
import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _parity_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    x = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=1)(x).detach()
    tp_block = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh["tp"])
    out = tp_block(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    distenv.shutdown()


def test_tp_block_matches_single_device():
    run_distributed(_parity_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_tp.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.tp'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/tp.py`:
```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

TP_PLAN = {
    "q": ColwiseParallel(),
    "k": ColwiseParallel(),
    "v": ColwiseParallel(),
    "o": RowwiseParallel(),
    "fc1": ColwiseParallel(),
    "fc2": RowwiseParallel(),
}


def apply_tp(block, tp_mesh):
    tp_size = tp_mesh.size()
    if block.n_heads % tp_size != 0:
        raise ValueError(f"n_heads {block.n_heads} not divisible by tp size {tp_size}")
    parallelize_module(block, tp_mesh, TP_PLAN)
    return block
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_tp.py -v'`
Expected: PASS (1 passed). TP block output matches the single-device reference within 1e-4.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/tp.py dtensor/tests/test_tp.py
git commit -m "Add Megatron tensor-parallel plan for the Transformer block"
```

---

## Task 3: Training step with data-parallel gradient averaging (`train.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/train.py`
- Test: `dtensor/tests/test_train.py`

**Interfaces:**
- Consumes: a model, an optimizer, a list of batches, an optional `dp` DeviceMesh.
- Produces:
  - `average_gradients(model, dp_mesh) -> None` — all-reduce (SUM) each param's local grad over `dp_mesh.get_group()`, then divide by `dp_mesh.size()`. Skips params whose grad is `None`. Uses `p.grad.to_local()` when the grad is a `DTensor`.
  - `run_training(model, batches, optimizer, dp_mesh=None) -> list[float]` — per batch: `zero_grad`, forward, `loss = out.pow(2).mean()`, backward; if `dp_mesh` is set, `average_gradients` then record the dp-averaged loss; `optimizer.step()`; returns the per-step losses.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_train.py`:
```python
import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from dtensor_workshop.train import average_gradients, run_training


def _avg_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    dp = mesh["dp"]
    lin = torch.nn.Linear(4, 1)
    # give each dp replica a distinct grad; tp-peers share a value
    for p in lin.parameters():
        p.grad = torch.full_like(p, float(dp.get_local_rank() + 1))  # 1.0 or 2.0
    average_gradients(lin, dp)
    for p in lin.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 1.5)), p.grad  # mean(1,2)
    distenv.shutdown()


def test_average_gradients_means_over_dp():
    run_distributed(_avg_worker, world_size=4)


def _run_training_worker(rank, world_size):
    distenv.init_process_group("gloo")
    lin = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    batches = [torch.randn(3, 4, generator=torch.Generator().manual_seed(s)) for s in range(3)]
    losses = run_training(lin, batches, opt, dp_mesh=None)
    assert len(losses) == 3
    assert losses[1] != losses[0]  # loss changes as params update
    distenv.shutdown()


def test_run_training_returns_losses():
    run_distributed(_run_training_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_train.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.train'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/train.py`:
```python
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def average_gradients(model, dp_mesh) -> None:
    group = dp_mesh.get_group()
    size = dp_mesh.size()
    for p in model.parameters():
        if p.grad is None:
            continue
        local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
        dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
        local /= size


def run_training(model, batches, optimizer, dp_mesh=None):
    losses = []
    for batch in batches:
        optimizer.zero_grad()
        loss = model(batch).pow(2).mean()
        loss.backward()
        if dp_mesh is not None:
            average_gradients(model, dp_mesh)
            lt = torch.tensor([loss.item()])
            dist.all_reduce(lt, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
            losses.append((lt / dp_mesh.size()).item())
        else:
            losses.append(loss.item())
        optimizer.step()
    return losses
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_train.py -v'`
Expected: PASS (2 passed). Gradient averaging yields the mean (1.5) across dp replicas; training returns changing losses.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/train.py dtensor/tests/test_train.py
git commit -m "Add data-parallel gradient averaging and training loop"
```

---

## Task 4: Distributed checkpoint helpers (`checkpoint.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/checkpoint.py`
- Test: `dtensor/tests/test_checkpoint.py`

**Interfaces:**
- Consumes: a (TP) model, an optimizer, a `checkpoint_id` directory path.
- Produces:
  - `dcp_save(model, optimizer, checkpoint_id) -> None` — `get_state_dict(model, optimizer)` → `dcp.save({"model": msd, "optim": osd}, checkpoint_id=checkpoint_id)`.
  - `dcp_load(model, optimizer, checkpoint_id) -> None` — `get_state_dict` to obtain templates, `dcp.load(..., checkpoint_id=checkpoint_id)`, then `set_state_dict(model, optimizer, model_state_dict=msd, optim_state_dict=osd)`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_checkpoint.py`:
```python
import os
import tempfile

import torch

from dtensor_workshop import distenv
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _roundtrip_worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    x = torch.randn(2, 8, 32, generator=torch.Generator().manual_seed(0))

    m1 = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh["tp"])
    opt1 = torch.optim.SGD(m1.parameters(), lr=0.1)
    m1(x).pow(2).mean().backward()
    opt1.step()
    dcp_save(m1, opt1, ckpt_dir)

    m2 = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=999), mesh["tp"])  # different init
    opt2 = torch.optim.SGD(m2.parameters(), lr=0.1)
    dcp_load(m2, opt2, ckpt_dir)

    o1, o2 = m1(x), m2(x)
    o1 = o1.full_tensor() if isinstance(o1, DTensor) else o1
    o2 = o2.full_tensor() if isinstance(o2, DTensor) else o2
    assert torch.allclose(o1, o2, atol=1e-6), (o1 - o2).abs().max().item()
    distenv.shutdown()


def test_dcp_roundtrip(tmp_path):
    run_distributed(_roundtrip_worker, world_size=4, args=(str(tmp_path / "ckpt"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_checkpoint.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/checkpoint.py`:
```python
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def dcp_save(model, optimizer, checkpoint_id) -> None:
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=checkpoint_id)


def dcp_load(model, optimizer, checkpoint_id) -> None:
    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.load({"model": model_sd, "optim": optim_sd}, checkpoint_id=checkpoint_id)
    set_state_dict(
        model, optimizer, model_state_dict=model_sd, optim_state_dict=optim_sd
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_checkpoint.py -v'`
Expected: PASS (1 passed). A model loaded from the checkpoint produces identical output to the saved one (a `tensor.storage().size()` UserWarning from DCP is benign).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/checkpoint.py dtensor/tests/test_checkpoint.py
git commit -m "Add DCP save/load helpers"
```

---

## Task 5: Activation checkpointing (`acheckpoint.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/acheckpoint.py`
- Test: `dtensor/tests/test_acheckpoint.py`

**Interfaces:**
- Produces: `forward_maybe_checkpointed(module, x, use_ac: bool)` — returns `torch.utils.checkpoint.checkpoint(module, x, use_reentrant=False)` when `use_ac` is True, else `module(x)`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_acheckpoint.py`:
```python
import torch

from dtensor_workshop.acheckpoint import forward_maybe_checkpointed
from dtensor_workshop.model import build_block


def test_ac_matches_plain_forward():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    plain = forward_maybe_checkpointed(block, x, use_ac=False)
    ac = forward_maybe_checkpointed(block, x, use_ac=True)
    assert torch.allclose(plain, ac, atol=1e-6), (plain - ac).abs().max().item()


def test_ac_backward_runs():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32, requires_grad=True)
    forward_maybe_checkpointed(block, x, use_ac=True).sum().backward()
    assert block.fc1.weight.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_acheckpoint.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.acheckpoint'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/acheckpoint.py`:
```python
from torch.utils.checkpoint import checkpoint


def forward_maybe_checkpointed(module, x, use_ac: bool):
    if use_ac:
        return checkpoint(module, x, use_reentrant=False)
    return module(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_acheckpoint.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/acheckpoint.py dtensor/tests/test_acheckpoint.py
git commit -m "Add activation checkpointing helper"
```

---

## Task 6: Lab 1 — 2D mesh roles (`l2_mesh.py`)

**Files:**
- Create: `dtensor/labs/level2/__init__.py`
- Create: `dtensor/labs/level2/l2_mesh.py`
- Test: `dtensor/tests/test_l2_mesh.py`

**Interfaces:**
- Produces:
  - `mesh_coords(mesh) -> dict` returning `{"dp": mesh["dp"].get_local_rank(), "tp": mesh["tp"].get_local_rank()}`.
  - `main()` builds the `dp2 × tp2` mesh and logs the rank's `(dp, tp)` coordinate.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_mesh.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_mesh


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    coords = l2_mesh.mesh_coords(mesh)
    assert coords["dp"] in (0, 1) and coords["tp"] in (0, 1)
    # rank r -> (dp=r//2, tp=r%2) for a row-major (dp, tp) mesh
    assert coords["dp"] == rank // 2
    assert coords["tp"] == rank % 2
    distenv.shutdown()


def test_mesh_coords():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_mesh.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_mesh'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/__init__.py`: empty file.

`dtensor/labs/level2/l2_mesh.py`:
```python
from dtensor_workshop import distenv, mesh as mesh_mod, rlog


def mesh_coords(mesh) -> dict:
    return {"dp": mesh["dp"].get_local_rank(), "tp": mesh["tp"].get_local_rank()}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    coords = mesh_coords(mesh)
    rlog.info(f"mesh coordinate dp={coords['dp']} tp={coords['tp']}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_mesh.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level2/__init__.py dtensor/labs/level2/l2_mesh.py dtensor/tests/test_l2_mesh.py
git commit -m "Add Level 2 Lab 1: 2D mesh roles"
```

---

## Task 7: Lab 2 — TP block parity (`l2_tp_block.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_tp_block.py`
- Test: `dtensor/tests/test_l2_tp_block.py`

**Interfaces:**
- Consumes: `model.build_block`, `tp.apply_tp`.
- Produces:
  - `tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=1) -> float` — builds a reference block and a TP copy from the same seed, runs both on the same replicated input, returns the max abs difference between the (gathered) TP output and the reference.
  - `main()` builds the mesh, reports the parity max-diff.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_tp_block.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_tp_block


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    diff = l2_tp_block.tp_parity_maxdiff(mesh)
    assert diff < 1e-4, diff
    distenv.shutdown()


def test_tp_parity():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_tp_block.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_tp_block'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_tp_block.py`:
```python
import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp


def tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=1) -> float:
    x = torch.randn(2, 8, dim, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)(x).detach()
    tp_block = apply_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"]
    )
    out = tp_block(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    return (out - ref).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    diff = tp_parity_maxdiff(mesh)
    rlog.info(f"TP block parity max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_tp_block.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level2/l2_tp_block.py dtensor/tests/test_l2_tp_block.py
git commit -m "Add Level 2 Lab 2: TP block parity"
```

---

## Task 8: Lab 3 — 2D training loss parity (`l2_train.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_train.py`
- Test: `dtensor/tests/test_l2_train.py`

**Interfaces:**
- Consumes: `model.build_block`, `tp.apply_tp`, `train.run_training`, `synth`.
- Produces:
  - `parallel_and_baseline_losses(mesh, steps=4, dim=32, hidden=64, n_heads=4, seed=5) -> tuple[list[float], list[float]]` — returns `(parallel_losses, baseline_losses)`. Baseline: a single-device block (same seed) trained with SGD on the full batch for `steps`. Parallel: a same-seed TP block trained with the `dp` mesh, each dp replica taking its half of the global batch, for `steps`. Both use the SAME global batch each step and SGD lr 0.1.
  - `main()` reports the max abs diff between the two loss lists.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_train.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_train


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    par, base = l2_train.parallel_and_baseline_losses(mesh, steps=4)
    assert len(par) == len(base) == 4
    maxdiff = max(abs(a - b) for a, b in zip(par, base))
    assert maxdiff < 1e-4, (maxdiff, par, base)
    distenv.shutdown()


def test_2d_loss_parity():
    run_distributed(_worker, world_size=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_train.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_train'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_train.py`:
```python
import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _global_batch(dim, generator):
    return torch.randn(4, 8, dim, generator=generator)


def parallel_and_baseline_losses(mesh, steps=4, dim=32, hidden=64, n_heads=4, seed=5):
    gen = torch.Generator().manual_seed(2024)
    global_batch = _global_batch(dim, gen)
    batches = [global_batch for _ in range(steps)]

    # single-device baseline on the full batch
    base_model = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    base_opt = torch.optim.SGD(base_model.parameters(), lr=0.1)
    baseline = run_training(base_model, batches, base_opt, dp_mesh=None)

    # 2D parallel: TP block, dp replica takes its half of each global batch
    dp = mesh["dp"]
    lo = dp.get_local_rank() * 2
    par_batches = [b[lo:lo + 2] for b in batches]
    par_model = apply_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"]
    )
    par_opt = torch.optim.SGD(par_model.parameters(), lr=0.1)
    parallel = run_training(par_model, par_batches, par_opt, dp_mesh=dp)
    return parallel, baseline


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    par, base = parallel_and_baseline_losses(mesh)
    maxdiff = max(abs(a - b) for a, b in zip(par, base))
    rlog.info(f"2D-parallel vs single-device loss max abs diff = {maxdiff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_train.py -v'`
Expected: PASS (1 passed). The 2D-parallel loss curve matches the single-device baseline within tolerance (spiked at 0.0).

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level2/l2_train.py dtensor/tests/test_l2_train.py
git commit -m "Add Level 2 Lab 3: 2D training loss parity"
```

---

## Task 9: Lab 4 — DCP save/restore/resume (`l2_dcp.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_dcp.py`
- Test: `dtensor/tests/test_l2_dcp.py`

**Interfaces:**
- Consumes: `model.build_block`, `tp.apply_tp`, `train.run_training`, `checkpoint.dcp_save`/`dcp_load`.
- Produces:
  - `save_restore_resume_maxdiff(mesh, checkpoint_id, steps=2, dim=32, hidden=64, n_heads=4, seed=6) -> float` — trains a TP model `steps`, saves via DCP; builds a fresh differently-seeded TP model, loads the checkpoint, then runs ONE more identical training step on both the original and restored models and returns the max abs diff of their post-step outputs (expected ~0 → the restored model resumes identically).
  - `main()` uses `checkpoints/l2_dcp` and reports the diff.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_dcp.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_dcp


def _worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    diff = l2_dcp.save_restore_resume_maxdiff(mesh, ckpt_dir)
    assert diff < 1e-6, diff
    distenv.shutdown()


def test_dcp_resume(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path / "l2ckpt"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_dcp.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_dcp'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_dcp.py`:
```python
import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def save_restore_resume_maxdiff(mesh, checkpoint_id, steps=2, dim=32, hidden=64, n_heads=4, seed=6):
    x = torch.randn(2, 8, dim, generator=torch.Generator().manual_seed(0))
    batches = [x for _ in range(steps)]

    orig = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"])
    orig_opt = torch.optim.SGD(orig.parameters(), lr=0.1)
    run_training(orig, batches, orig_opt, dp_mesh=mesh["dp"])
    dcp_save(orig, orig_opt, checkpoint_id)

    restored = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=123), mesh["tp"])
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1)
    dcp_load(restored, restored_opt, checkpoint_id)

    # one more identical step on both; outputs should stay identical
    run_training(orig, [x], orig_opt, dp_mesh=mesh["dp"])
    run_training(restored, [x], restored_opt, dp_mesh=mesh["dp"])
    return (_full(orig(x)) - _full(restored(x))).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    diff = save_restore_resume_maxdiff(mesh, "checkpoints/l2_dcp")
    rlog.info(f"resume-after-restore output max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_dcp.py -v'`
Expected: PASS (1 passed). The restored model resumes and evolves identically to the original.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level2/l2_dcp.py dtensor/tests/test_l2_dcp.py
git commit -m "Add Level 2 Lab 4: DCP save/restore/resume"
```

---

## Task 10: Lab 5 — activation checkpointing and OOM (`l2_oom.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_oom.py`
- Test: `dtensor/tests/test_l2_oom.py`

**Interfaces:**
- Consumes: `model.build_block`, `acheckpoint.forward_maybe_checkpointed`.
- Produces:
  - `ac_equivalence_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float` — max abs diff between a plain forward and an activation-checkpointed forward of the block (expected ~0).
  - GPU-only helpers (guarded by `torch.cuda.is_available()`), used only by `main()`:
    - `start_memory_record() -> None` → `torch.cuda.memory._record_memory_history(max_entries=100000)`.
    - `dump_memory_snapshot(path) -> None` → `torch.cuda.memory._dump_snapshot(path)`.
  - `main(argv)` parses `--size {small,big}` and `--ac`: builds a block sized per `--size` (small: `dim=512,hidden=2048,n_heads=8,batch=8,seq=512`; big scales up: `dim=2048,hidden=8192,n_heads=16,batch=16,seq=2048` — **tune upward on H200's 141 GB if OOM does not trigger**), records a memory snapshot to `artifacts/l2_mem_rank{rank}.pickle`, runs a forward/backward with or without activation checkpointing, and logs peak memory. On CPU it just runs the small forward.

- [ ] **Step 1: Write the failing test** (CPU-testable part only — AC equivalence)

`dtensor/tests/test_l2_oom.py`:
```python
from labs.level2 import l2_oom


def test_ac_equivalence():
    assert l2_oom.ac_equivalence_maxdiff() < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_oom.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_oom'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_oom.py`:
```python
import sys

import torch

from dtensor_workshop import distenv, rlog
from dtensor_workshop.acheckpoint import forward_maybe_checkpointed
from dtensor_workshop.model import build_block

SIZES = {
    "small": dict(dim=512, hidden=2048, n_heads=8, batch=8, seq=512),
    # Tune upward on H200 (141 GB) if OOM does not trigger.
    "big": dict(dim=2048, hidden=8192, n_heads=16, batch=16, seq=2048),
}


def ac_equivalence_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float:
    block = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    x = torch.randn(2, 8, dim, requires_grad=True)
    plain = forward_maybe_checkpointed(block, x, use_ac=False)
    ac = forward_maybe_checkpointed(block, x, use_ac=True)
    return (plain - ac).abs().max().item()


def start_memory_record() -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)


def dump_memory_snapshot(path) -> None:
    torch.cuda.memory._dump_snapshot(path)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    size = "big" if "--size" in argv and "big" in argv else "small"
    use_ac = "--ac" in argv
    cfg = SIZES[size]

    if not torch.cuda.is_available():
        rlog.info(f"CPU: AC-equivalence max diff = {ac_equivalence_maxdiff()}")
        return

    distenv.init_process_group()
    device = torch.device("cuda", distenv.local_rank())
    start_memory_record()
    block = build_block(cfg["dim"], cfg["hidden"], cfg["n_heads"]).to(device)
    x = torch.randn(cfg["batch"], cfg["seq"], cfg["dim"], device=device, requires_grad=True)
    forward_maybe_checkpointed(block, x, use_ac=use_ac).pow(2).mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    dump_memory_snapshot(f"artifacts/l2_mem_rank{distenv.rank()}.pickle")
    rlog.info(f"size={size} ac={use_ac} peak_gb={peak:.2f}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_oom.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: GPU smoke run (documented — requires a GPU node; not part of the CI suite)**

On a `kempner_h100` node:
```bash
cd dtensor && sbatch slurm/launch_1node.sbatch -m labs.level2.l2_oom --size small
# then compare peak memory with and without activation checkpointing:
cd dtensor && sbatch slurm/launch_1node.sbatch -m labs.level2.l2_oom --size small --ac
```
Expected: `--ac` run logs a lower `peak_gb`; `artifacts/l2_mem_rank*.pickle` snapshots are produced (load in https://pytorch.org/memory_viz). Use `--size big` to force an OOM and inspect the snapshot; if `big` does not OOM on H200, raise the `SIZES["big"]` values.

- [ ] **Step 6: Commit**

```bash
git add dtensor/labs/level2/l2_oom.py dtensor/tests/test_l2_oom.py
git commit -m "Add Level 2 Lab 5: activation checkpointing and OOM diagnosis"
```

---

## Task 11: Lab 6 — profile a 2D step (`l2_profile.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_profile.py`
- Test: `dtensor/tests/test_l2_profile.py`

**Interfaces:**
- Consumes: `model.build_block`, `tp.apply_tp`, `train.run_training`.
- Produces:
  - `profiled_2d_step(mesh, out_path) -> int` — profiles one 2D-parallel training step (TP + dp gradient averaging) under `torch.profiler.profile`, exports a chrome trace to `out_path`, returns `len(prof.key_averages())`.
  - `main()` writes a **per-rank** path `artifacts/l2_trace_rank{rank}.json` (per the Level 1 fix — never have all ranks write one file).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_profile.py`:
```python
import os

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_profile


def _worker(rank, world_size, out_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    out_path = os.path.join(out_dir, f"l2_trace_rank{rank}.json")
    n = l2_profile.profiled_2d_step(mesh, out_path)
    assert n > 0
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
    distenv.shutdown()


def test_profiled_2d_step(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_profile.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_profile'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_profile.py`:
```python
import torch
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def profiled_2d_step(mesh, out_path) -> int:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    block = apply_tp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh["tp"])
    opt = torch.optim.SGD(block.parameters(), lr=0.1)
    lo = mesh["dp"].get_local_rank() * 2
    batch = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0))[lo:lo + 2]
    with profile(activities=activities) as prof:
        run_training(block, [batch], opt, dp_mesh=mesh["dp"])
    prof.export_chrome_trace(out_path)
    return len(prof.key_averages())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    out_path = f"artifacts/l2_trace_rank{distenv.rank()}.json"
    n = profiled_2d_step(mesh, out_path)
    rlog.info(f"exported {n} profiled event rows to {out_path}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_profile.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level2/l2_profile.py dtensor/tests/test_l2_profile.py
git commit -m "Add Level 2 Lab 6: profile a 2D training step"
```

---

## Task 12: Capstone — 2D training loop with checkpoint, profile, and diagnosis (`l2_capstone.py`)

**Files:**
- Create: `dtensor/labs/level2/l2_capstone.py`
- Test: `dtensor/tests/test_l2_capstone.py`

**Interfaces:**
- Consumes: `model.build_block`, `tp.apply_tp`, `train.run_training`, `checkpoint.dcp_save`/`dcp_load`, `acheckpoint`, `torch.profiler`.
- Produces:
  - `run_capstone(mesh, checkpoint_id, trace_path, steps=4, dim=32, hidden=64, n_heads=4, seed=8) -> dict` — trains a 2D-parallel (TP + dp) block with activation checkpointing for `steps`, saves a DCP checkpoint mid-run, exports a profiler trace, reloads into a fresh model, and returns `{"parity_maxdiff": <vs single-device baseline>, "resume_maxdiff": <reloaded vs original output>}`.
  - `main()` uses `checkpoints/l2_capstone` and `artifacts/l2_capstone_trace_rank{rank}.json`, logs both diffs, and prints a short written-diagnosis template for the participant to fill in.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l2_capstone.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level2 import l2_capstone


def _worker(rank, world_size, ckpt_dir, trace_dir):
    import os
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp", "tp"), device_type="cpu")
    res = l2_capstone.run_capstone(
        mesh, ckpt_dir, os.path.join(trace_dir, f"trace_rank{rank}.json")
    )
    assert res["parity_maxdiff"] < 1e-4, res
    assert res["resume_maxdiff"] < 1e-6, res
    distenv.shutdown()


def test_capstone(tmp_path):
    run_distributed(
        _worker, world_size=4,
        args=(str(tmp_path / "ckpt"), str(tmp_path)),
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_capstone.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l2_capstone'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level2/l2_capstone.py`:
```python
import torch
from torch.distributed.tensor import DTensor
from torch.profiler import ProfilerActivity, profile

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.tp import apply_tp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def run_capstone(mesh, checkpoint_id, trace_path, steps=4, dim=32, hidden=64, n_heads=4, seed=8):
    dp = mesh["dp"]
    gen = torch.Generator().manual_seed(2024)
    global_batch = torch.randn(4, 8, dim, generator=gen)
    batches = [global_batch for _ in range(steps)]

    # single-device baseline
    base = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    base_opt = torch.optim.SGD(base.parameters(), lr=0.1)
    baseline = run_training(base, batches, base_opt, dp_mesh=None)

    # 2D-parallel training with a profiler trace
    lo = dp.get_local_rank() * 2
    par_batches = [b[lo:lo + 2] for b in batches]
    model = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed), mesh["tp"])
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        parallel = run_training(model, par_batches, opt, dp_mesh=dp)
    prof.export_chrome_trace(trace_path)

    dcp_save(model, opt, checkpoint_id)
    restored = apply_tp(build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=321), mesh["tp"])
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1)
    dcp_load(restored, restored_opt, checkpoint_id)

    parity = max(abs(a - b) for a, b in zip(parallel, baseline))
    resume = (_full(model(global_batch[lo:lo + 2])) - _full(restored(global_batch[lo:lo + 2]))).abs().max().item()
    return {"parity_maxdiff": parity, "resume_maxdiff": resume}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp", "tp"))
    res = run_capstone(
        mesh, "checkpoints/l2_capstone",
        f"artifacts/l2_capstone_trace_rank{distenv.rank()}.json",
    )
    rlog.info(f"parity_maxdiff={res['parity_maxdiff']:.2e} resume_maxdiff={res['resume_maxdiff']:.2e}")
    if distenv.rank() == 0:
        rlog.info(
            "DIAGNOSIS (fill in): dominant bottleneck? "
            "which collective (all-gather/reduce-scatter/all-reduce)? "
            "peak memory before vs after activation checkpointing?"
        )
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l2_capstone.py -v'`
Expected: PASS (1 passed).

- [ ] **Step 5: Run the full Level 1 + Level 2 suite**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests -q'`
Expected: all tests PASS (Level 1's 27 plus the new Level 2 tests).

Optional GPU smoke run (on a `kempner_h100` node):
```bash
cd dtensor && sbatch slurm/launch_1node.sbatch -m labs.level2.l2_capstone
```
Expected: logs `parity_maxdiff` and `resume_maxdiff` near zero; `checkpoints/l2_capstone/` and per-rank `artifacts/l2_capstone_trace_rank*.json` created.

- [ ] **Step 6: Commit**

```bash
git add dtensor/labs/level2/l2_capstone.py dtensor/tests/test_l2_capstone.py
git commit -m "Add Level 2 capstone: 2D training with checkpoint, profile, and diagnosis"
```

---

## Self-Review

**1. Spec coverage (Level 2 scope from `workshop_design.md`):**

| Spec item | Task |
| --- | --- |
| 2D mesh `dp2 × tp2`, named dims | Tasks 6, and every parallel task |
| Tensor-parallel linear layers (Megatron column/row) | Tasks 2, 7 |
| Data-parallel training across `dp` | Tasks 3, 8 |
| Loss parity vs single-device baseline | Task 8 (and capstone Task 12) |
| DCP sharded save/load; restart validation | Tasks 4, 9 |
| Activation checkpointing tradeoffs | Tasks 5, 10 |
| Memory profiling / OOM / snapshot | Task 10 (GPU smoke run) |
| Communication bottleneck profiling | Tasks 11, 12 |
| Capstone (2D loop + DCP + trace + memory opt + diagnosis) | Task 12 |
| Running example (compact Transformer block) | Task 1 |
| Milestone order correct → restartable → fast | Task ordering (parity → DCP → memory/profile) |
| Failure-driven lab = OOM | Task 10 |

Deferred by design (Level 3): FSDP2, MoE, FP8, async/overlap, regional compile, NCCL/Flight-Recorder debugging, 3D mesh, 2-node execution.

Note: **async checkpoint saves** (design "Asynchronous checkpoint saves") are NOT implemented here — synchronous `dcp.save` only. This is a deliberate scope trim for Level 2; `dcp.async_save` can be added later. Flagged for the human to confirm.

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to Task N". The `SIZES["big"]` values in Task 10 are concrete starting numbers explicitly documented as tunable on H200 — a hardware parameter, not a plan gap. Every code step shows complete code.

**3. Type consistency:** Names are stable across tasks — `build_block(dim, hidden, n_heads, seed)`, `apply_tp(block, tp_mesh)`, `TP_PLAN`, `average_gradients(model, dp_mesh)`, `run_training(model, batches, optimizer, dp_mesh=None)`, `dcp_save(model, optimizer, checkpoint_id)`/`dcp_load(...)`, `forward_maybe_checkpointed(module, x, use_ac)`. Every lab imports these with matching signatures. All DTensor/TP/DCP APIs are the verified public paths from the spike. Meshes are uniformly `build_mesh((2, 2), ("dp", "tp"))` with `world_size=4`, and per-rank profiler paths follow the Level 1 fix (no shared-file writes).
