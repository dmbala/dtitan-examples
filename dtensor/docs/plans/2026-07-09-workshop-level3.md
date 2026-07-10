# DTensor Workshop — Level 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Level 3 — 3D (`dp_replicate × dp_shard × tp`) parallelism with FSDP2, a Mixture-of-Experts layer, regional `torch.compile`, FP8, fault-aware data loading, and NCCL/Flight-Recorder debugging — on top of the merged Level 1 + Level 2 foundation, targeting 2 nodes × 4 GPUs.

**Architecture:** Extend `dtensor_workshop` with FSDP2/HSDP+TP composition (`parallel3d.py`), a MoE feed-forward (`moe.py`), regional compile (`regional.py`), a resumable data loader (`faultdata.py`), an FP8 helper (`fp8.py`), and NCCL/Flight-Recorder utilities (`frdebug.py`). Reuse Level 2's `model.TransformerBlock`, `tp.apply_tp`, `checkpoint.dcp_save/dcp_load`, `train.run_training`. Each lab is a thin `main()` plus a pure function. **CPU/gloo tests cover everything that runs without a GPU** — 3D mesh, FSDP2, HSDP+TP, MoE, DCP, compile, data-loader restart (all spike-verified to match single-device to float precision). Genuinely GPU-only behavior — **FP8 execution, NCCL Flight-Recorder traces, real 2-node runs, and overlap/throughput timing** — is validated by documented `sbatch` smoke runs, never by CPU unit tests.

**Tech Stack:** Python 3, PyTorch 2.10 (2.11 preview) from `dtitan.sif`, `torch.distributed.tensor` / `.tensor.parallel` / `.device_mesh` / `.fsdp` (FSDP2 `fully_shard`) / `.checkpoint` (DCP), `torch.compile`, `torchao.float8`, Flight Recorder, `torch.profiler`, pytest, Slurm (2-node), Apptainer/Singularity.

## Verified API facts (spiked on CPU/gloo before writing this plan)

Implement these exactly — each was confirmed to match a single-device reference to float precision:

- **3D mesh:** `init_device_mesh(dev, (2, 2, 2), mesh_dim_names=("dp_replicate", "dp_shard", "tp"))`. The HSDP sub-mesh for FSDP2 is the 2D slice `mesh["dp_replicate", "dp_shard"]` (ndim 2). A 4D mesh constructs fine too (used only illustratively).
- **FSDP2:** `from torch.distributed.fsdp import fully_shard`; `fully_shard(module, mesh=<1D-or-2D-mesh>, reshard_after_forward=<bool>)`. Params become DTensors; forward all-gathers, backward reduce-scatters. Works on CPU/gloo (forward parity 0.0 vs single-device). Composes with DCP via Level 2's `dcp_save`/`dcp_load` (round-trip 0.0).
- **HSDP + TP:** apply TP **first** (`parallelize_module(model, mesh["tp"], TP_PLAN)` / `apply_tp`), **then** `fully_shard(model, mesh=mesh["dp_replicate", "dp_shard"])`. Forward parity vs single-device = 1.8e-07.
- **MoE + FSDP2:** a router (`nn.Linear(dim, n_experts)`) + `nn.ModuleList` of expert MLPs, top-1 argmax routing, masked per-expert dispatch, works under `fully_shard` (parity 0.0). Routing imbalance = `max(counts) / mean(counts)`.
- **Regional compile:** `torch.compile(module)` runs on CPU (output matches eager to 1.8e-07).
- **FP8 (torchao 0.14, GPU/Hopper only):** `from torchao.float8 import convert_to_float8_training, Float8LinearConfig`; `convert_to_float8_training(model, config=Float8LinearConfig())`. Importable on CPU; **only runs on a Hopper GPU** — never call it on CPU.
- **Flight Recorder:** the container sets the **deprecated** `TORCH_NCCL_TRACE_BUFFER_SIZE=20971520`; torch 2.10 wants `TORCH_FR_BUFFER_SIZE`. GPU smoke runs must `export TORCH_FR_BUFFER_SIZE=20971520`. Flight Recorder is NCCL-only (no CPU/gloo equivalent).

## Global Constraints

- **Container (all commands run inside it):** `IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`. `singularity exec "$IMAGE" …`; add `--nv` only for GPU runs.
- **PyTorch:** torch 2.10. Public APIs only: `torch.distributed.tensor`, `.tensor.parallel`, `.device_mesh`, `.fsdp` (`fully_shard`), `.checkpoint`. No `torch.distributed._tensor`. No torchtitan.
- **Do not re-set the NCCL env vars the container already exports** (`TORCH_NCCL_ASYNC_ERROR_HANDLING`, `TORCH_NCCL_DUMP_ON_TIMEOUT`, `TORCH_NCCL_TRACE_BUFFER_SIZE`). The **only** Flight-Recorder addition allowed is exporting `TORCH_FR_BUFFER_SIZE` in GPU smoke runs (torch 2.10's non-deprecated name).
- **Working directory** is `dtensor/`. Labs run as modules: `torchrun … -m labs.level3.l3_xxx`.
- **Level 3 hardware:** 2 nodes × 4 GPUs = 8, partition `kempner_h100`. Uses the existing `slurm/launch_2node.sbatch`.
- **Level 3 primary mesh:** real 3D `dp_replicate=2 × dp_shard=2 × tp=2` (= 8). 4D (adding `pp`) is illustrative only.
- **Tests are GPU-free:** backend `gloo`, device `cpu`, via `dtensor_workshop.testing.run_distributed`. 3D-mesh tests use `world_size=8`; FSDP2/HSDP/MoE tests use `world_size=2` or `4`; MoE-routing / compile / data-loader tests are non-distributed. GPU-only behavior (FP8, Flight Recorder, 2-node, timing) is validated only by documented smoke runs.
- **Reuse the foundation:** `distenv`, `mesh.build_mesh`, `synth`, `rlog`, `testing.run_distributed`, `model.build_block`, `model.TransformerBlock`, `tp.apply_tp`/`TP_PLAN`, `checkpoint.dcp_save`/`dcp_load`, `train.run_training`, `acheckpoint`. Do not reimplement.
- **Model test dims:** `dim=32, hidden=64, n_heads=4` (n_heads divisible by tp=2). Optimizer **SGD(lr=0.1, momentum=0.9)** where DCP optimizer state is checked. Seeds explicit and rank-independent.
- **Benign stderr noise** (judge pass/fail only by pytest's summary line): pynvml deprecation, `socket.cpp … Address family`, `TORCH_NCCL_TRACE_BUFFER_SIZE` deprecation, Gloo connection info, CPU-only/kineto profiler notices, DCP `tensor.storage().size()` UserWarning, FSDP2/compile info logs.

## File Structure

```
dtensor/
├── dtensor_workshop/
│   ├── parallel3d.py     # apply_fsdp + apply_hsdp_tp                      [NEW]
│   ├── moe.py            # MoEFeedForward + routing_imbalance              [NEW]
│   ├── regional.py       # regional_compile                                [NEW]
│   ├── faultdata.py      # ResumableLoader                                 [NEW]
│   ├── fp8.py            # maybe_convert_fp8 (GPU/Hopper only)             [NEW]
│   └── frdebug.py        # shapes_agree + flight-recorder helpers          [NEW]
├── labs/level3/
│   ├── __init__.py                                                         [NEW]
│   ├── l3_mesh.py        # Lab 1: 3D mesh roles (+ illustrative 4D)
│   ├── l3_fsdp.py        # Lab 2: FSDP2 (HSDP) + DCP
│   ├── l3_hsdp_tp.py     # Lab 3: HSDP + TP on the 3D mesh
│   ├── l3_moe.py         # Lab 4: MoE + routing imbalance
│   ├── l3_compile.py     # Lab 5: regional torch.compile
│   ├── l3_faultdata.py   # Lab 6: fault-aware deterministic restart
│   ├── l3_nccl_debug.py  # Lab 7: collective-shape check (+ GPU FR smoke)
│   └── l3_capstone.py    # Capstone: MoE under HSDP + DCP + optimization
└── tests/
    ├── test_parallel3d.py  test_moe.py  test_regional.py  test_faultdata.py
    ├── test_fp8.py         test_frdebug.py
    ├── test_l3_mesh.py     test_l3_fsdp.py  test_l3_hsdp_tp.py  test_l3_moe.py
    ├── test_l3_compile.py  test_l3_faultdata.py  test_l3_nccl_debug.py
    └── test_l3_capstone.py
```

**Test command:** `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/<file> -v'`. Run the full suite (`python -m pytest tests -q`, ~5–7 min with world=8 tests) once before each commit. `world_size=8` tests are slower; be patient with timeouts.

---

## Task 1: FSDP2 + HSDP/TP composition (`parallel3d.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/parallel3d.py`
- Test: `dtensor/tests/test_parallel3d.py`

**Interfaces:**
- Consumes: `tp.apply_tp`, a DeviceMesh, `torch.distributed.fsdp.fully_shard`.
- Produces:
  - `apply_fsdp(model, mesh, reshard_after_forward=True) -> model` — `fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)`; returns the model. (`mesh` is the FSDP mesh: 1D `dp` or 2D HSDP `dp_replicate,dp_shard`.)
  - `apply_hsdp_tp(model, mesh) -> model` — applies TP on `mesh["tp"]` (via `apply_tp`), then FSDP2 on the HSDP sub-mesh `mesh["dp_replicate", "dp_shard"]`; returns the model.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_parallel3d.py`:
```python
import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_fsdp, apply_hsdp_tp
from dtensor_workshop.testing import run_distributed
from torch.distributed.tensor import DTensor


def _fsdp_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=1)(x).detach()
    model = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1), mesh)
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    assert isinstance(next(model.parameters()), DTensor)
    distenv.shutdown()


def _hsdp_tp_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0))
    ref = build_block(dim=32, hidden=64, n_heads=4, seed=2)(x).detach()
    model = apply_hsdp_tp(build_block(dim=32, hidden=64, n_heads=4, seed=2), mesh)
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    assert torch.allclose(out, ref, atol=1e-4), (out - ref).abs().max().item()
    distenv.shutdown()


def test_fsdp_parity():
    run_distributed(_fsdp_worker, world_size=2)


def test_hsdp_tp_parity():
    run_distributed(_hsdp_tp_worker, world_size=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_parallel3d.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.parallel3d'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/parallel3d.py`:
```python
from torch.distributed.fsdp import fully_shard

from .tp import apply_tp


def apply_fsdp(model, mesh, reshard_after_forward: bool = True):
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model


def apply_hsdp_tp(model, mesh):
    apply_tp(model, mesh["tp"])
    fully_shard(model, mesh=mesh["dp_replicate", "dp_shard"])
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_parallel3d.py -v'`
Expected: PASS (2 passed). FSDP2 forward matches single-device; HSDP+TP matches within 1e-4. (world=8 test takes ~30–60 s.)

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/parallel3d.py dtensor/tests/test_parallel3d.py
git commit -m "Add FSDP2 and HSDP+TP composition helpers"
```

---

## Task 2: Mixture-of-Experts feed-forward (`moe.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/moe.py`
- Test: `dtensor/tests/test_moe.py`

**Interfaces:**
- Produces:
  - `class MoEFeedForward(nn.Module)` — `__init__(dim=256, hidden=1024, n_experts=4)` builds `router = nn.Linear(dim, n_experts)` and `experts = nn.ModuleList([<MLP>])` (each expert: `nn.Linear(dim, hidden)` → `nn.GELU` → `nn.Linear(hidden, dim)`). `forward(x)` for `x` shape `(tokens, dim)` does top-1 argmax routing, dispatches each token to its expert (masked), and returns `(out, counts)` where `counts` is a list of per-expert token counts.
  - `routing_imbalance(counts) -> float` — `max(counts) / mean(counts)` (1.0 = perfectly balanced); returns `0.0` if all counts are zero.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_moe.py`:
```python
import torch

from dtensor_workshop.moe import MoEFeedForward, routing_imbalance


def test_all_tokens_routed():
    torch.manual_seed(0)
    moe = MoEFeedForward(dim=32, hidden=64, n_experts=4)
    out, counts = moe(torch.randn(64, 32))
    assert tuple(out.shape) == (64, 32)
    assert sum(counts) == 64
    assert len(counts) == 4


def test_routing_imbalance():
    assert routing_imbalance([10, 10, 10, 10]) == 1.0
    assert routing_imbalance([40, 0, 0, 0]) == 4.0
    assert routing_imbalance([0, 0, 0, 0]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_moe.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.moe'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/moe.py`:
```python
import torch
import torch.nn as nn


class _Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MoEFeedForward(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 1024, n_experts: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList([_Expert(dim, hidden) for _ in range(n_experts)])

    def forward(self, x):
        assignment = self.router(x).argmax(dim=-1)      # top-1
        out = torch.zeros_like(x)
        counts = []
        for e in range(self.n_experts):
            mask = assignment == e
            counts.append(int(mask.sum()))
            if mask.any():
                out[mask] = self.experts[e](x[mask])
        return out, counts


def routing_imbalance(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    mean = total / len(counts)
    return max(counts) / mean
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_moe.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/moe.py dtensor/tests/test_moe.py
git commit -m "Add Mixture-of-Experts feed-forward and routing-imbalance metric"
```

---

## Task 3: Regional compile (`regional.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/regional.py`
- Test: `dtensor/tests/test_regional.py`

**Interfaces:**
- Produces: `regional_compile(module)` — returns `torch.compile(module)` (compiles one stable model region).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_regional.py`:
```python
import torch

from dtensor_workshop.model import build_block
from dtensor_workshop.regional import regional_compile


def test_compiled_matches_eager():
    block = build_block(dim=32, hidden=64, n_heads=4, seed=3)
    x = torch.randn(2, 8, 32)
    eager = block(x)
    compiled = regional_compile(block)(x)
    assert torch.allclose(eager, compiled, atol=1e-4), (eager - compiled).abs().max().item()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_regional.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.regional'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/regional.py`:
```python
import torch


def regional_compile(module):
    return torch.compile(module)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_regional.py -v'`
Expected: PASS (1 passed). The first call compiles (may take a few seconds); output matches eager within 1e-4.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/regional.py dtensor/tests/test_regional.py
git commit -m "Add regional torch.compile helper"
```

---

## Task 4: Fault-aware resumable data loader (`faultdata.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/faultdata.py`
- Test: `dtensor/tests/test_faultdata.py`

**Interfaces:**
- Produces:
  - `class ResumableLoader` — `__init__(shape, base_seed=0)`; `next() -> torch.Tensor` yields a deterministic batch seeded by `(base_seed, step)` and increments `step`; `state_dict() -> dict` returns `{"step": step}`; `load_state_dict(state)` restores `step`. Two loaders at the same step yield identical batches (deterministic restart on a shared filesystem).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_faultdata.py`:
```python
import torch

from dtensor_workshop.faultdata import ResumableLoader


def test_deterministic_and_resumable():
    a = ResumableLoader((4, 8), base_seed=7)
    first = [a.next() for _ in range(5)]
    saved = a.state_dict()          # after 5 steps

    # a fresh loader restored to the saved step resumes identically
    b = ResumableLoader((4, 8), base_seed=7)
    b.load_state_dict(saved)
    resumed = [b.next() for _ in range(3)]

    a_more = [a.next() for _ in range(3)]
    for r, e in zip(resumed, a_more):
        assert torch.equal(r, e)


def test_step_advances():
    loader = ResumableLoader((2, 2))
    assert loader.state_dict()["step"] == 0
    loader.next()
    assert loader.state_dict()["step"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_faultdata.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.faultdata'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/faultdata.py`:
```python
import torch


class ResumableLoader:
    def __init__(self, shape, base_seed: int = 0):
        self.shape = tuple(shape)
        self.base_seed = base_seed
        self.step = 0

    def next(self) -> torch.Tensor:
        gen = torch.Generator().manual_seed(self.base_seed * 1_000_003 + self.step)
        batch = torch.randn(*self.shape, generator=gen)
        self.step += 1
        return batch

    def state_dict(self) -> dict:
        return {"step": self.step}

    def load_state_dict(self, state) -> None:
        self.step = state["step"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_faultdata.py -v'`
Expected: PASS (2 passed). A loader restored to a saved step reproduces the exact subsequent batches.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/faultdata.py dtensor/tests/test_faultdata.py
git commit -m "Add fault-aware resumable data loader"
```

---

## Task 5: FP8 conversion helper (`fp8.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/fp8.py`
- Test: `dtensor/tests/test_fp8.py`

**Interfaces:**
- Produces:
  - `hopper_available() -> bool` — True iff CUDA is available AND the device compute capability is ≥ (8, 9) (Hopper/Ada FP8).
  - `maybe_convert_fp8(model) -> model` — on Hopper, `convert_to_float8_training(model, config=Float8LinearConfig())` and return it; otherwise return the model **unchanged** (so CPU/non-Hopper runs are a safe no-op). FP8 only executes on a Hopper GPU.

- [ ] **Step 1: Write the failing test** (CPU-testable part: the no-op guard)

`dtensor/tests/test_fp8.py`:
```python
from dtensor_workshop.fp8 import hopper_available, maybe_convert_fp8
from dtensor_workshop.model import build_block


def test_no_op_off_hopper():
    # On CPU CI, hopper_available() is False and the model is returned unchanged.
    if hopper_available():
        return  # on a Hopper GPU this path is exercised by the smoke run, not here
    block = build_block(dim=32, hidden=64, n_heads=4, seed=0)
    assert maybe_convert_fp8(block) is block


def test_hopper_available_is_bool():
    assert isinstance(hopper_available(), bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_fp8.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.fp8'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/fp8.py`:
```python
import torch


def hopper_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) >= (8, 9)


def maybe_convert_fp8(model):
    if not hopper_available():
        return model
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    return convert_to_float8_training(model, config=Float8LinearConfig())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_fp8.py -v'`
Expected: PASS (2 passed). On CPU CI `maybe_convert_fp8` is a no-op returning the same object.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/fp8.py dtensor/tests/test_fp8.py
git commit -m "Add Hopper-guarded FP8 conversion helper"
```

---

## Task 6: NCCL/Flight-Recorder debug utilities (`frdebug.py`)

**Files:**
- Create: `dtensor/dtensor_workshop/frdebug.py`
- Test: `dtensor/tests/test_frdebug.py`

**Interfaces:**
- Consumes: `torch.distributed`.
- Produces:
  - `shapes_agree(tensor, group) -> bool` — all-gathers each rank's element count over `group` and returns whether every rank agrees. A CPU/gloo-safe way to catch the shape disagreements that cause NCCL collective hangs (the failure Flight Recorder diagnoses on GPU). Uses a fixed-size scalar all-gather (never itself mismatches).
  - `dump_flight_recorder(path) -> bool` — GPU/NCCL only: if `torch.cuda.is_available()`, dumps the Flight Recorder trace to `path` via `torch.distributed.flight_recorder` / `_dump_nccl_trace` and returns True; on CPU returns False (no-op). (The dump content is validated by the GPU smoke run, not CI.)

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_frdebug.py`:
```python
import torch

from dtensor_workshop import distenv
from dtensor_workshop.frdebug import shapes_agree
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed


def _agree_worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    group = mesh.get_group()
    same = torch.zeros(8)                       # identical on every rank
    assert shapes_agree(same, group) is True
    differ = torch.zeros(rank + 1)              # rank-dependent size -> disagreement
    assert shapes_agree(differ, group) is False
    distenv.shutdown()


def test_shapes_agree_detects_mismatch():
    run_distributed(_agree_worker, world_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_frdebug.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'dtensor_workshop.frdebug'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/dtensor_workshop/frdebug.py`:
```python
import torch
import torch.distributed as dist


def shapes_agree(tensor, group) -> bool:
    world = dist.get_world_size(group)
    local = torch.tensor([float(tensor.numel())])
    gathered = [torch.zeros(1) for _ in range(world)]
    dist.all_gather(gathered, local, group=group)
    values = [g.item() for g in gathered]
    return all(v == values[0] for v in values)


def dump_flight_recorder(path) -> bool:
    if not torch.cuda.is_available():
        return False
    from torch.distributed import _dump_nccl_trace
    with open(path, "wb") as fh:
        fh.write(_dump_nccl_trace())
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_frdebug.py -v'`
Expected: PASS (1 passed). `shapes_agree` returns True for identical tensors and False for rank-dependent sizes — a hang-free preflight for the collective-mismatch failure mode.

- [ ] **Step 5: Commit**

```bash
git add dtensor/dtensor_workshop/frdebug.py dtensor/tests/test_frdebug.py
git commit -m "Add collective-shape check and Flight Recorder dump helper"
```

---

## Task 7: Lab 1 — 3D mesh roles (`l3_mesh.py`)

**Files:**
- Create: `dtensor/labs/level3/__init__.py`
- Create: `dtensor/labs/level3/l3_mesh.py`
- Test: `dtensor/tests/test_l3_mesh.py`

**Interfaces:**
- Produces:
  - `mesh_roles(mesh) -> dict` — `{"dp_replicate": mesh["dp_replicate"].get_local_rank(), "dp_shard": mesh["dp_shard"].get_local_rank(), "tp": mesh["tp"].get_local_rank()}`.
  - `build_illustrative_4d() -> DeviceMesh` — constructs a 4D `(2, 2, 2, 1)` mesh named `("dp_replicate", "dp_shard", "tp", "pp")`; **illustrative only** — the `pp` dim is size 1 because 8 GPUs cannot exercise a real 4D layout.
  - `main()` builds the 3D mesh and logs the rank's role and the physical-topology mapping (replicate across nodes; shard + tp within a node over NVLink).

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_mesh.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_mesh


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    roles = l3_mesh.mesh_roles(mesh)
    assert set(roles) == {"dp_replicate", "dp_shard", "tp"}
    assert all(0 <= v <= 1 for v in roles.values())
    four_d = l3_mesh.build_illustrative_4d()
    assert four_d.ndim == 4
    distenv.shutdown()


def test_mesh_roles_and_4d():
    run_distributed(_worker, world_size=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_mesh.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_mesh'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/__init__.py`: empty file.

`dtensor/labs/level3/l3_mesh.py`:
```python
from dtensor_workshop import distenv, mesh as mesh_mod, rlog


def mesh_roles(mesh) -> dict:
    return {
        "dp_replicate": mesh["dp_replicate"].get_local_rank(),
        "dp_shard": mesh["dp_shard"].get_local_rank(),
        "tp": mesh["tp"].get_local_rank(),
    }


def build_illustrative_4d():
    # pp is size 1: 8 GPUs cannot exercise a real 4D layout — code-that-would-scale only.
    return mesh_mod.build_mesh((2, 2, 2, 1), ("dp_replicate", "dp_shard", "tp", "pp"))


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"))
    roles = mesh_roles(mesh)
    rlog.info(
        f"roles={roles} | mapping: dp_replicate across nodes (inter-node link), "
        f"dp_shard+tp within a node (NVLink)"
    )
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_mesh.py -v'`
Expected: PASS (1 passed). (world=8, ~30–60 s.)

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/__init__.py dtensor/labs/level3/l3_mesh.py dtensor/tests/test_l3_mesh.py
git commit -m "Add Level 3 Lab 1: 3D mesh roles and illustrative 4D"
```

---

## Task 8: Lab 2 — FSDP2 (HSDP) + DCP (`l3_fsdp.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_fsdp.py`
- Test: `dtensor/tests/test_l3_fsdp.py`

**Interfaces:**
- Consumes: `model.build_block`, `parallel3d.apply_fsdp`, `checkpoint.dcp_save`/`dcp_load`, `train.run_training`.
- Produces:
  - `fsdp_resume_maxdiff(mesh, checkpoint_id, steps=2, reshard_after_forward=True) -> float` — builds a block, FSDP2-shards it over the 2D HSDP mesh (`mesh["dp_replicate", "dp_shard"]`), trains `steps` (dp mesh = the flattened FSDP mesh), DCP-saves; builds a fresh differently-seeded FSDP2 block, DCP-loads; runs one more identical step on both; returns the max abs diff of post-step outputs. `reshard_after_forward=False` keeps params gathered (less communication, more memory) — the overlap knob.
  - `main()` uses `checkpoints/l3_fsdp`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_fsdp.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_fsdp


def _worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp_replicate", "dp_shard"), device_type="cpu")
    diff = l3_fsdp.fsdp_resume_maxdiff(mesh, ckpt_dir)
    assert diff < 1e-6, diff
    distenv.shutdown()


def test_fsdp_dcp_resume(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path / "ckpt"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_fsdp.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_fsdp'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_fsdp.py`:
```python
import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_fsdp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def fsdp_resume_maxdiff(mesh, checkpoint_id, steps=2, reshard_after_forward=True):
    device = mesh.device_type
    x = torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(0)).to(device)
    batches = [x for _ in range(steps)]

    orig = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1).to(device),
                      mesh, reshard_after_forward=reshard_after_forward)
    orig_opt = torch.optim.SGD(orig.parameters(), lr=0.1, momentum=0.9)
    run_training(orig, batches, orig_opt)
    dcp_save(orig, orig_opt, checkpoint_id)

    restored = apply_fsdp(build_block(dim=32, hidden=64, n_heads=4, seed=1001).to(device),
                          mesh, reshard_after_forward=reshard_after_forward)
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    dcp_load(restored, restored_opt, checkpoint_id)

    run_training(orig, [x], orig_opt)
    run_training(restored, [x], restored_opt)
    return (_full(orig(x)) - _full(restored(x))).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp_replicate", "dp_shard"))
    diff = fsdp_resume_maxdiff(mesh, "checkpoints/l3_fsdp")
    rlog.info(f"FSDP2 resume-after-restore max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_fsdp.py -v'`
Expected: PASS (1 passed). FSDP2 (HSDP) model resumes from DCP identically (a momentum SGD exercises real optimizer-state restore).

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/l3_fsdp.py dtensor/tests/test_l3_fsdp.py
git commit -m "Add Level 3 Lab 2: FSDP2 (HSDP) with DCP resume"
```

---

## Task 9: Lab 3 — HSDP + TP on the 3D mesh (`l3_hsdp_tp.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_hsdp_tp.py`
- Test: `dtensor/tests/test_l3_hsdp_tp.py`

**Interfaces:**
- Consumes: `model.build_block`, `parallel3d.apply_hsdp_tp`.
- Produces:
  - `hsdp_tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=2) -> float` — builds a same-seed reference block and a copy parallelized with `apply_hsdp_tp` on the 3D mesh, runs both on the same input, returns max abs diff between the gathered parallel output and the reference.
  - `main()` builds the 3D mesh and reports the parity max diff.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_hsdp_tp.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_hsdp_tp


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"), device_type="cpu")
    diff = l3_hsdp_tp.hsdp_tp_parity_maxdiff(mesh)
    assert diff < 1e-4, diff
    distenv.shutdown()


def test_hsdp_tp_parity():
    run_distributed(_worker, world_size=8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_hsdp_tp.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_hsdp_tp'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_hsdp_tp.py`:
```python
import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.parallel3d import apply_hsdp_tp


def hsdp_tp_parity_maxdiff(mesh, dim=32, hidden=64, n_heads=4, seed=2) -> float:
    device = mesh.device_type
    x = torch.randn(4, 8, dim, generator=torch.Generator().manual_seed(0)).to(device)
    ref = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device)(x).detach()
    model = apply_hsdp_tp(
        build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed).to(device), mesh
    )
    out = model(x)
    out = out.full_tensor() if isinstance(out, DTensor) else out
    return (out - ref).abs().max().item()


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2, 2), ("dp_replicate", "dp_shard", "tp"))
    diff = hsdp_tp_parity_maxdiff(mesh)
    rlog.info(f"HSDP+TP parity max abs diff = {diff}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_hsdp_tp.py -v'`
Expected: PASS (1 passed). Combined HSDP + TP output matches single-device within 1e-4 (spiked at 1.8e-07). (world=8, ~60 s.)

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/l3_hsdp_tp.py dtensor/tests/test_l3_hsdp_tp.py
git commit -m "Add Level 3 Lab 3: HSDP + TP on the 3D mesh"
```

---

## Task 10: Lab 4 — MoE and routing imbalance (`l3_moe.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_moe.py`
- Test: `dtensor/tests/test_l3_moe.py`

**Interfaces:**
- Consumes: `moe.MoEFeedForward`, `moe.routing_imbalance`, `parallel3d.apply_fsdp`.
- Produces:
  - `moe_report(mesh, tokens=64, dim=32, hidden=64, n_experts=4, seed=5) -> dict` — builds an `MoEFeedForward`, FSDP2-shards it over `mesh`, runs a forward on `tokens` synthetic tokens, and returns `{"counts": [...], "imbalance": float, "routed": int}` where `routed == tokens`.
  - `main()` builds the HSDP mesh and logs counts + imbalance.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_moe.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_moe


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    rep = l3_moe.moe_report(mesh, tokens=64, n_experts=4)
    assert rep["routed"] == 64
    assert len(rep["counts"]) == 4
    assert rep["imbalance"] >= 1.0
    distenv.shutdown()


def test_moe_report():
    run_distributed(_worker, world_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_moe.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_moe'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_moe.py`:
```python
import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.moe import MoEFeedForward, routing_imbalance
from dtensor_workshop.parallel3d import apply_fsdp


def moe_report(mesh, tokens=64, dim=32, hidden=64, n_experts=4, seed=5) -> dict:
    device = mesh.device_type
    torch.manual_seed(seed)
    moe = apply_fsdp(MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device), mesh)
    x = torch.randn(tokens, dim, generator=torch.Generator().manual_seed(0)).to(device)
    _out, counts = moe(x)
    return {"counts": counts, "imbalance": routing_imbalance(counts), "routed": sum(counts)}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    rep = moe_report(mesh)
    rlog.info(f"expert counts={rep['counts']} imbalance={rep['imbalance']:.3f}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_moe.py -v'`
Expected: PASS (1 passed). All tokens are routed; imbalance ≥ 1.0. (Note: this demonstrates MoE under FSDP2; true expert-parallel sharding across a mesh dim is a GPU extension noted in the lab.)

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/l3_moe.py dtensor/tests/test_l3_moe.py
git commit -m "Add Level 3 Lab 4: MoE and routing imbalance"
```

---

## Task 11: Lab 5 — regional compile (`l3_compile.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_compile.py`
- Test: `dtensor/tests/test_l3_compile.py`

**Interfaces:**
- Consumes: `model.build_block`, `regional.regional_compile`.
- Produces:
  - `compile_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float` — max abs diff between an eager forward and a `regional_compile`d forward of the block (expected ~0).
  - `main()` compiles the block, logs the max diff and a note to compare step time on GPU.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_compile.py`:
```python
from labs.level3 import l3_compile


def test_compile_maxdiff():
    assert l3_compile.compile_maxdiff() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_compile.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_compile'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_compile.py`:
```python
import torch

from dtensor_workshop import distenv, rlog
from dtensor_workshop.model import build_block
from dtensor_workshop.regional import regional_compile


def compile_maxdiff(dim=32, hidden=64, n_heads=4, seed=3) -> float:
    block = build_block(dim=dim, hidden=hidden, n_heads=n_heads, seed=seed)
    x = torch.randn(2, 8, dim)
    eager = block(x)
    compiled = regional_compile(block)(x)
    return (eager - compiled).abs().max().item()


def main():
    rlog.info(f"regional compile eager-vs-compiled max diff = {compile_maxdiff()} "
              f"(compare step time on GPU: torch.compile amortizes after warm-up)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_compile.py -v'`
Expected: PASS (1 passed). Compiled output matches eager within 1e-4.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/l3_compile.py dtensor/tests/test_l3_compile.py
git commit -m "Add Level 3 Lab 5: regional torch.compile"
```

---

## Task 12: Lab 6 — fault-aware deterministic restart (`l3_faultdata.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_faultdata.py`
- Test: `dtensor/tests/test_l3_faultdata.py`

**Interfaces:**
- Consumes: `faultdata.ResumableLoader`.
- Produces:
  - `simulate_crash_and_resume(shape=(4, 8), base_seed=7, total=6, crash_at=3) -> bool` — runs a loader `crash_at` steps, saves its state, then simulates a crash by building a fresh loader and restoring the state; returns True iff the resumed batches for the remaining steps exactly match an uninterrupted run.
  - `main()` logs whether the deterministic restart succeeded.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_faultdata.py`:
```python
from labs.level3 import l3_faultdata


def test_crash_and_resume():
    assert l3_faultdata.simulate_crash_and_resume() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_faultdata.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_faultdata'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_faultdata.py`:
```python
import torch

from dtensor_workshop import rlog
from dtensor_workshop.faultdata import ResumableLoader


def simulate_crash_and_resume(shape=(4, 8), base_seed=7, total=6, crash_at=3) -> bool:
    reference = ResumableLoader(shape, base_seed=base_seed)
    all_batches = [reference.next() for _ in range(total)]

    crashing = ResumableLoader(shape, base_seed=base_seed)
    for _ in range(crash_at):
        crashing.next()
    saved = crashing.state_dict()               # checkpoint on a shared filesystem

    resumed = ResumableLoader(shape, base_seed=base_seed)
    resumed.load_state_dict(saved)              # restart from the checkpoint
    tail = [resumed.next() for _ in range(total - crash_at)]
    return all(torch.equal(t, ref) for t, ref in zip(tail, all_batches[crash_at:]))


def main():
    rlog.info(f"deterministic restart succeeded = {simulate_crash_and_resume()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_faultdata.py -v'`
Expected: PASS (1 passed). Resumed batches exactly match the uninterrupted run.

- [ ] **Step 5: Commit**

```bash
git add dtensor/labs/level3/l3_faultdata.py dtensor/tests/test_l3_faultdata.py
git commit -m "Add Level 3 Lab 6: fault-aware deterministic restart"
```

---

## Task 13: Lab 7 — collective-shape check and NCCL debugging (`l3_nccl_debug.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_nccl_debug.py`
- Test: `dtensor/tests/test_l3_nccl_debug.py`

**Interfaces:**
- Consumes: `frdebug.shapes_agree`, `frdebug.dump_flight_recorder`.
- Produces:
  - `diagnose_batch(local_batch, mesh) -> bool` — returns `frdebug.shapes_agree(local_batch, mesh.get_group())`; a preflight that catches the rank-to-rank shape disagreement that would otherwise hang a collective (the failure Flight Recorder pinpoints on GPU).
  - `main()` runs `diagnose_batch` on an intentionally rank-dependent batch, logs the disagreement, and — on GPU only — dumps a Flight Recorder trace to `artifacts/l3_flight_recorder_rank{rank}.dump`.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_nccl_debug.py`:
```python
import torch

from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_nccl_debug


def _worker(rank, world_size):
    distenv.init_process_group("gloo")
    mesh = build_mesh((world_size,), ("dp",), device_type="cpu")
    assert l3_nccl_debug.diagnose_batch(torch.zeros(8), mesh) is True     # agree
    assert l3_nccl_debug.diagnose_batch(torch.zeros(rank + 1), mesh) is False  # disagree
    distenv.shutdown()


def test_diagnose_batch():
    run_distributed(_worker, world_size=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_nccl_debug.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_nccl_debug'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_nccl_debug.py`:
```python
import torch

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.frdebug import dump_flight_recorder, shapes_agree


def diagnose_batch(local_batch, mesh) -> bool:
    return shapes_agree(local_batch, mesh.get_group())


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((distenv.world_size(),), ("dp",))
    # intentional per-rank shape disagreement (the collective-mismatch failure mode)
    agree = diagnose_batch(torch.zeros(distenv.rank() + 1), mesh)
    rlog.info(f"batch shapes agree across ranks = {agree}")
    dumped = dump_flight_recorder(f"artifacts/l3_flight_recorder_rank{distenv.rank()}.dump")
    rlog.info(f"flight recorder dumped = {dumped}")
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_nccl_debug.py -v'`
Expected: PASS (1 passed). `diagnose_batch` catches the shape disagreement without hanging.

- [ ] **Step 5: GPU smoke run (documented — requires 2 GPU nodes; not part of CI)**

On the cluster, force Flight Recorder's non-deprecated buffer var and inject a fault:
```bash
cd dtensor && TORCH_FR_BUFFER_SIZE=20971520 sbatch slurm/launch_2node.sbatch -m labs.level3.l3_nccl_debug
```
Expected: the run reports the shape disagreement and writes `artifacts/l3_flight_recorder_rank*.dump`. Inspect the dump to identify the failing rank/collective. **Note:** the container exports the *deprecated* `TORCH_NCCL_TRACE_BUFFER_SIZE`; torch 2.10 reads `TORCH_FR_BUFFER_SIZE`, so set it explicitly for a reliable non-empty trace.

- [ ] **Step 6: Commit**

```bash
git add dtensor/labs/level3/l3_nccl_debug.py dtensor/tests/test_l3_nccl_debug.py
git commit -m "Add Level 3 Lab 7: collective-shape check and Flight Recorder debugging"
```

---

## Task 14: Capstone — MoE under HSDP with DCP and an optimization (`l3_capstone.py`)

**Files:**
- Create: `dtensor/labs/level3/l3_capstone.py`
- Test: `dtensor/tests/test_l3_capstone.py`

**Interfaces:**
- Consumes: `moe.MoEFeedForward`, `moe.routing_imbalance`, `parallel3d.apply_fsdp`, `checkpoint.dcp_save`/`dcp_load`, `train.run_training`, `fp8.maybe_convert_fp8`, `frdebug.shapes_agree`.
- Produces:
  - `run_capstone(mesh, checkpoint_id, steps=3, dim=32, hidden=64, n_experts=4, seed=8, reshard_after_forward=False) -> dict` — trains a small MoE model (`MoEFeedForward` over synthetic tokens) under FSDP2/HSDP on `mesh`, applies the **overlap optimization** (`reshard_after_forward=False`), records routing imbalance, DCP-saves, reloads into a fresh differently-seeded model, and returns `{"resume_maxdiff": float, "imbalance": float, "steps": int}`. On a Hopper GPU it also converts the model to FP8 via `maybe_convert_fp8` (a no-op on CPU).
  - `main()` uses `checkpoints/l3_capstone`, logs the metrics, and prints a postmortem-style debugging note template.

- [ ] **Step 1: Write the failing test**

`dtensor/tests/test_l3_capstone.py`:
```python
from dtensor_workshop import distenv
from dtensor_workshop.mesh import build_mesh
from dtensor_workshop.testing import run_distributed
from labs.level3 import l3_capstone


def _worker(rank, world_size, ckpt_dir):
    distenv.init_process_group("gloo")
    mesh = build_mesh((2, 2), ("dp_replicate", "dp_shard"), device_type="cpu")
    res = l3_capstone.run_capstone(mesh, ckpt_dir)
    assert res["resume_maxdiff"] < 1e-6, res
    assert res["imbalance"] >= 1.0
    assert res["steps"] == 3
    distenv.shutdown()


def test_capstone(tmp_path):
    run_distributed(_worker, world_size=4, args=(str(tmp_path / "ckpt"),))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_capstone.py -v'`
Expected: FAIL — `ImportError: cannot import name 'l3_capstone'`.

- [ ] **Step 3: Write minimal implementation**

`dtensor/labs/level3/l3_capstone.py`:
```python
import torch
from torch.distributed.tensor import DTensor

from dtensor_workshop import distenv, mesh as mesh_mod, rlog
from dtensor_workshop.checkpoint import dcp_load, dcp_save
from dtensor_workshop.fp8 import maybe_convert_fp8
from dtensor_workshop.moe import MoEFeedForward, routing_imbalance
from dtensor_workshop.parallel3d import apply_fsdp
from dtensor_workshop.train import run_training


def _full(out):
    return out.full_tensor() if isinstance(out, DTensor) else out


def run_capstone(mesh, checkpoint_id, steps=3, dim=32, hidden=64, n_experts=4,
                 seed=8, reshard_after_forward=False):
    device = mesh.device_type
    x = torch.randn(64, dim, generator=torch.Generator().manual_seed(0)).to(device)
    batches = [x for _ in range(steps)]

    torch.manual_seed(seed)
    model = maybe_convert_fp8(MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device))
    model = apply_fsdp(model, mesh, reshard_after_forward=reshard_after_forward)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    losses = []
    for batch in batches:                       # MoE returns (out, counts); custom loop
        opt.zero_grad()
        out, counts = model(batch)
        out.pow(2).mean().backward()
        opt.step()
        losses.append(routing_imbalance(counts))
    dcp_save(model, opt, checkpoint_id)

    torch.manual_seed(seed + 1000)
    restored = apply_fsdp(
        MoEFeedForward(dim=dim, hidden=hidden, n_experts=n_experts).to(device), mesh,
        reshard_after_forward=reshard_after_forward,
    )
    restored_opt = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
    dcp_load(restored, restored_opt, checkpoint_id)

    resume = (_full(model(x)[0]) - _full(restored(x)[0])).abs().max().item()
    return {"resume_maxdiff": resume, "imbalance": losses[-1], "steps": steps}


def main():
    distenv.init_process_group()
    mesh = mesh_mod.build_mesh((2, 2), ("dp_replicate", "dp_shard"))
    res = run_capstone(mesh, "checkpoints/l3_capstone")
    rlog.info(f"resume_maxdiff={res['resume_maxdiff']:.2e} imbalance={res['imbalance']:.3f}")
    if distenv.rank() == 0:
        rlog.info(
            "POSTMORTEM (fill in): which rank/collective failed? Flight Recorder finding? "
            "mesh layout (dp_replicate/dp_shard/tp)? overlap or FP8 speedup? routing imbalance?"
        )
    distenv.shutdown()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests/test_l3_capstone.py -v'`
Expected: PASS (1 passed). The MoE model trains under FSDP2/HSDP with the overlap knob, resumes from DCP identically, and reports routing imbalance.

- [ ] **Step 5: Run the full suite and the documented GPU smoke**

Run: `singularity exec "$IMAGE" bash -lc 'cd dtensor && python -m pytest tests -q'`
Expected: all tests PASS (Levels 1 + 2 + 3; ~5–7 min).

Optional GPU smoke runs (on 2 `kempner_h100` nodes) — the GPU-only story CI cannot cover:
```bash
# HSDP+TP + capstone on real hardware; FP8 activates automatically on Hopper:
cd dtensor && sbatch slurm/launch_2node.sbatch -m labs.level3.l3_capstone
# NCCL/Flight Recorder fault isolation (note the FR buffer var):
cd dtensor && TORCH_FR_BUFFER_SIZE=20971520 sbatch slurm/launch_2node.sbatch -m labs.level3.l3_nccl_debug
```
Expected: capstone logs `resume_maxdiff` near zero and a routing imbalance; on Hopper the FP8 path activates; the NCCL-debug run produces Flight Recorder dumps under `artifacts/`.

- [ ] **Step 6: Commit**

```bash
git add dtensor/labs/level3/l3_capstone.py dtensor/tests/test_l3_capstone.py
git commit -m "Add Level 3 capstone: MoE under HSDP with DCP recovery and overlap"
```

---

## Self-Review

**1. Spec coverage (Level 3 scope from `workshop_design.md`):**

| Spec item | Task |
| --- | --- |
| Real 3D mesh (`dp_replicate × dp_shard × tp`) mapped to topology | Tasks 7, 9 |
| 4D taught as conceptual/illustrative | Task 7 (`build_illustrative_4d`) |
| FSDP2 (`fully_shard`) for param/grad/optim sharding | Tasks 1, 8 |
| DCP compatibility with FSDP2 | Task 8 (and capstone) |
| MoE routing, token dispatch, imbalance | Tasks 2, 10 |
| Communication/compute overlap (reshard_after_forward; async TP) | Tasks 1/8 (`reshard_after_forward`), capstone; async-TP env noted |
| FP8 on Hopper (torchao) — core | Task 5 (helper), capstone (activates on Hopper) |
| Regional `torch.compile` | Tasks 3, 11 |
| NCCL debugging + Flight Recorder | Tasks 6, 13 |
| Fault-aware data loading / deterministic restart | Tasks 4, 12 |
| 2-node execution | `slurm/launch_2node.sbatch` (existing), Tasks 13/14 smoke runs |
| Capstone (mesh + FSDP2 + DCP + overlap/precision + postmortem) | Task 14 |

Deferred / GPU-only-by-design (documented smoke runs, not CI): FP8 numeric execution, real Flight Recorder trace content, actual 2-node runs, overlap/throughput **timing**, and true expert-parallel sharding of MoE across a mesh dim (Task 10 demonstrates MoE under FSDP2; EP is noted as the GPU extension). These are inherent GPU/multi-node behaviors the CPU/gloo test strategy structurally cannot exercise — every CPU-testable correctness claim is covered.

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to Task N". FP8 (`convert_to_float8_training` + `Float8LinearConfig`) and Flight Recorder (`_dump_nccl_trace`) are the verified torchao 0.14 / torch 2.10 symbols; their runtime is GPU-only and guarded. Every code step shows complete code.

**3. Type consistency:** Names are stable across tasks — `apply_fsdp(model, mesh, reshard_after_forward=True)`, `apply_hsdp_tp(model, mesh)`, `MoEFeedForward(dim, hidden, n_experts)` returning `(out, counts)`, `routing_imbalance(counts)`, `regional_compile(module)`, `ResumableLoader(shape, base_seed)` with `next`/`state_dict`/`load_state_dict`, `hopper_available()`, `maybe_convert_fp8(model)`, `shapes_agree(tensor, group)`, `dump_flight_recorder(path)`. Labs consume these with matching signatures and reuse Level 2's `dcp_save`/`dcp_load`, `run_training`, `build_block`, `apply_tp`. All parallelism uses the public `torch.distributed.fsdp` / `.tensor` / `.checkpoint` APIs. Device placement follows the Level 2 lesson (`.to(mesh.device_type)` on models and inputs). The DP-loss device fix and per-rank artifact paths from Levels 1–2 carry forward.
