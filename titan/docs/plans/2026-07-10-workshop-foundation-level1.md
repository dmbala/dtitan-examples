# TorchTitan Workshop — Foundation + Level 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared TorchTitan-workshop scaffolding (repo layout, the rebuilt-container recipe, Slurm launchers, a preflight check) plus a complete, runnable **Level 1** module (config inspection, fake-backend dry-run, a 1D FSDP2 run, metrics + profiler, and a failure-driven debugging lab) driven entirely through TorchTitan's `--module/--config` interface and CLI overrides.

**Architecture:** The workshop *operates* TorchTitan rather than adding model code. Deliverables are therefore **configuration + launchers + guided lab docs + a preflight script**, not a Python library. Runs use the built-in `--module llama3 --config llama3_debugmodel` config with dotted CLI overrides; the real Llama-3.1 tokenizer and (optionally) the Llama-3.1-8B model come from the offline testbed. Artifacts land under `outputs/`.

**Tech Stack:** TorchTitan 0.2.2, PyTorch ≥ 2.11 (rebuilt container), torchao, flash-attn, Slurm, Apptainer/Singularity, pytest (for the statically-testable pieces).

## ⚠️ Execution gating (read first)

TorchTitan's runtime is **blocked on a container rebuild**. On the *current* `dtitan.sif` (torch 2.10.0a0), `import torchtitan.train` and `import torchtitan.models.llama3` both fail on missing torch-2.11 symbols (`_context_parallel_shard`, `activate_flash_attention_impl`); only `torchtitan.config` imports. Therefore each task below is tagged:

- **[now]** — authorable and testable on the current container (no torchtitan model/train import, no GPU).
- **[rebuilt]** — authored now, but **validated only on the rebuilt torch ≥ 2.11 container** (Task 1). Its "test" is a documented run with an expected result.

Task 1 (the rebuild recipe) is the prerequisite for every **[rebuilt]** task. Building the image is a manual/infra step; this plan authors the recipe and the validation runs, it does not build the image.

## Global Constraints

- **Container:** rebuild `dtitan.sif` on an **NGC base with torch ≥ 2.11 stable** + **torchtitan==0.2.2** + torchao + flash-attn. `IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif`. Invoke with `singularity exec "$IMAGE" …`; add `--nv` for GPU.
- **Config interface (verified in-container):** `torchrun … -m torchtitan.train --module <model> --config <registered-config>` + **dotted CLI overrides**. Level 1 uses the built-in **`--module llama3 --config llama3_debugmodel`** + overrides — **no custom config registration** (keeps Level 1 free of the unverified registry API). Verified override paths: `--training.steps`, `--training.local_batch_size`, `--training.seq_len`, `--parallelism.data_parallel_shard_degree`, `--parallelism.tensor_parallel_degree`, `--profiler.enable_profiling`.
- **Assets (offline):** `MODELS=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`. Tokenizer: `--hf_assets_path=$MODELS/Llama-3.1-8B-Instruct`. Trainable real target: `$MODELS/Llama-3.1-8B` (`llama3` spec). `HF_HUB_OFFLINE=1` (container-set — do not change). The testbed holds weights+tokenizers, not datasets → use TorchTitan's offline/synthetic datasets.
- **Cluster:** `--account=kempner_dev`, `--partition=kempner_h100` (H100 80 GB, 4/node); 8-GPU/user cap. Reuse the **GPU-validated** dtensor launcher settings: `--mem=128G`, and (2-node) `srun --cpu-bind=none`.
- **Working directory** for all commands is `titan/`. Statically-testable pieces run with pytest inside the container: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests -q'`.
- **Flight Recorder** (Level 3, later): use `TORCH_FR_BUFFER_SIZE` (the container's `TORCH_NCCL_TRACE_BUFFER_SIZE` is deprecated on torch ≥ 2.11).

---

## File Structure

```
titan/
├── container/
│   └── dtitan.def              # rebuilt-container recipe (torch>=2.11 + torchtitan 0.2.2)
├── configs/
│   └── README.md               # Level-1 uses built-in llama3_debugmodel + overrides (no custom registry yet)
├── slurm/
│   ├── launch_1node.sbatch     # torchtitan.train, 1 node/4 GPU
│   └── launch_2node.sbatch     # torchtitan.train, 2 node/8 GPU (Level 3)
├── preflight.py                # environment self-check (detects the torch/torchtitan gate)
├── docs/
│   ├── workshop_design.md       (spec, already written)
│   └── labs/level1/
│       ├── README.md            # Level 1 overview + the 6 labs + capstone
│       └── validation.md        # exact run commands + expected results (on rebuilt container)
├── tests/
│   ├── test_scaffold.py
│   ├── test_container_def.py
│   ├── test_launchers.py
│   └── test_preflight.py
├── outputs/.gitkeep            # logs/checkpoints/snapshots/traces (git-ignored)
├── pyproject.toml
├── conftest.py
└── .gitignore
```

**Test command:** `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests -q'`.

---

## Task 1: Rebuilt-container recipe (`container/dtitan.def`)  [now]

**Files:**
- Create: `titan/container/dtitan.def`
- Test: `titan/tests/test_container_def.py`

**Interfaces:**
- Produces: an Apptainer definition targeting **torch ≥ 2.11** with **torchtitan==0.2.2**, torchao, flash-attn, offline HF env, and a build-time import sanity check for `torchtitan.train`.

- [ ] **Step 1: Write the failing test**

`titan/tests/test_container_def.py`:
```python
import pathlib

DEF = pathlib.Path(__file__).resolve().parent.parent / "container" / "dtitan.def"


def test_targets_torch_211_and_pinned_torchtitan():
    text = DEF.read_text()
    assert "torchtitan==0.2.2" in text
    assert "HF_HUB_OFFLINE=1" in text
    # base image comment must state the torch>=2.11 requirement
    assert "torch" in text and "2.11" in text
    # build must fail loudly if torchtitan.train can't import
    assert "import torchtitan.train" in text


def test_documents_rebuild_reason():
    text = DEF.read_text()
    assert "_context_parallel_shard" in text or "activate_flash_attention_impl" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_container_def.py -v'`
Expected: FAIL — `FileNotFoundError` for `dtitan.def`.

- [ ] **Step 3: Write the definition**

`titan/container/dtitan.def`:
```
Bootstrap: docker
# Base MUST provide torch >= 2.11 stable. NGC 25.11 ships torch 2.10.0a0, on which
# torchtitan 0.2.2 fails to import (_context_parallel_shard / activate_flash_attention_impl
# are torch-2.11 symbols). Use a later NGC monthly tag whose torch is >= 2.11.
From: nvcr.io/nvidia/pytorch:26.01-py3

%labels
    Maintainer bala_desinghu@harvard.edu
    Purpose    TorchTitan workshop on Kempner H100/H200 (torch >= 2.11, torchtitan 0.2.2)

%environment
    export HF_HOME=/data/hf_cache
    export HF_HUB_OFFLINE=1
    export PYTHONUNBUFFERED=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_FR_BUFFER_SIZE=20971520

%post
    set -eux
    export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
    pip install torchao
    pip install torchtitan==0.2.2
    pip install datasets tokenizers sentencepiece tomli-w pynvml huggingface_hub
    # Fail the build if the workshop entrypoint or a model can't import.
    python -c "import torchtitan.train; import torchtitan.models.llama3; print('torchtitan import OK')"
    mkdir -p /data/hf_cache && chmod 777 /data/hf_cache

%runscript
    exec "$@"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_container_def.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Document the build (does not run here)**

Add to `titan/container/dtitan.def` `%help` (or a sibling note) the build command:
`apptainer build dtitan.sif dtitan.def`, and that the `%post` import check will **fail the build** if the base torch is < 2.11 — the guardrail against the exact drift that broke the current image. Confirm the chosen NGC tag actually ships torch ≥ 2.11 before building.

- [ ] **Step 6: Commit**

```bash
git add titan/container/dtitan.def titan/tests/test_container_def.py
git commit -m "Add rebuilt-container recipe: torch>=2.11 + torchtitan 0.2.2 with import guard"
```

---

## Task 2: Repo scaffold + test harness  [now]

**Files:**
- Create: `titan/pyproject.toml`, `titan/conftest.py`, `titan/.gitignore`, `titan/outputs/.gitkeep`, `titan/configs/README.md`
- Test: `titan/tests/test_scaffold.py`

**Interfaces:**
- Produces: pytest rootdir at `titan/`; `outputs/` git-ignored; a `configs/README.md` stating Level 1 uses the built-in debug config.

- [ ] **Step 1: Write the failing test**

`titan/tests/test_scaffold.py`:
```python
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_dirs_exist():
    for d in ("container", "configs", "slurm", "docs", "outputs", "tests"):
        assert (ROOT / d).is_dir(), d


def test_gitignore_ignores_outputs():
    gi = (ROOT / ".gitignore").read_text()
    assert "outputs/*" in gi
    assert "!**/.gitkeep" in gi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_scaffold.py -v'`
Expected: FAIL — `outputs/` dir / `.gitignore` missing.

- [ ] **Step 3: Create the scaffold**

`titan/pyproject.toml`:
```toml
[project]
name = "titan-workshop"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests"]
```

`titan/conftest.py`:
```python
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
```

`titan/.gitignore`:
```gitignore
__pycache__/
*.pyc
.pytest_cache/
outputs/*
!**/.gitkeep
```

`titan/configs/README.md`:
```markdown
# Workshop configs

Level 1 runs the **built-in** TorchTitan config `--module llama3 --config
llama3_debugmodel` plus dotted CLI overrides — no custom config registration.
Later levels may register workshop-specific configs in the model's
`config_registry`; that API is confirmed on the rebuilt torch>=2.11 container.
```

Create empty `titan/outputs/.gitkeep`.

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_scaffold.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add titan/pyproject.toml titan/conftest.py titan/.gitignore titan/outputs/.gitkeep titan/configs/README.md titan/tests/test_scaffold.py
git commit -m "Scaffold titan workshop repo layout and pytest harness"
```

---

## Task 3: Slurm launchers  [now]

**Files:**
- Create: `titan/slurm/launch_1node.sbatch`, `titan/slurm/launch_2node.sbatch`
- Test: `titan/tests/test_launchers.py`

**Interfaces:**
- Produces: two launchers that `cd` into `titan/`, bind `outputs/`, and invoke `torchrun … -m torchtitan.train --module llama3 --config llama3_debugmodel "$@"`. 1-node: `--standalone --nproc_per_node=4`; 2-node: c10d rendezvous + `srun --cpu-bind=none`. Reuse the dtensor-validated `--account`/`--mem`/`--cpu-bind` settings.

- [ ] **Step 1: Write the failing test**

`titan/tests/test_launchers.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_launchers.py -v'`
Expected: FAIL — `FileNotFoundError`.

- [ ] **Step 3: Write the launchers**

`titan/slurm/launch_1node.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=titan-l12
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -d container ] && [ -d slurm ]; then :
elif [ -d titan/slurm ]; then cd titan
fi
mkdir -p outputs
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun --standalone --nproc_per_node=4 -m torchtitan.train \
    --module llama3 --config llama3_debugmodel "$@"
```

`titan/slurm/launch_2node.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=titan-l3
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -d container ] && [ -d slurm ]; then :
elif [ -d titan/slurm ]; then cd titan
fi
mkdir -p outputs
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan.sif

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR MASTER_PORT=29500

srun --cpu-bind=none singularity exec --nv \
  --bind "$(pwd)/outputs:/outputs" \
  "$IMAGE" \
  torchrun \
    --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" -m torchtitan.train \
    --module llama3 --config llama3_debugmodel "$@"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_launchers.py -v'`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add titan/slurm/launch_1node.sbatch titan/slurm/launch_2node.sbatch titan/tests/test_launchers.py
git commit -m "Add TorchTitan Slurm launchers (1-node, 2-node)"
```

---

## Task 4: Preflight check (`preflight.py`)  [now — detects the gate]

**Files:**
- Create: `titan/preflight.py`
- Test: `titan/tests/test_preflight.py`

**Interfaces:**
- Produces:
  - `check_torchtitan_import() -> tuple[str, bool, str]` — `import torchtitan` (config machinery).
  - `check_torchtitan_train() -> tuple[str, bool, str]` — `import torchtitan.train`; **False on torch 2.10, True on the rebuilt ≥ 2.11 image** (the key gate).
  - `check_tokenizer(path) -> tuple[str, bool, str]` — tokenizer dir readable.
  - `check_dir_writable(path) -> tuple[str, bool, str]`.
  - `check_gpu_visible() -> tuple[str, bool, str]`.
  - `run_checks() -> list[tuple[str, bool, str]]` + `main() -> int` (non-zero if any fail).

- [ ] **Step 1: Write the failing test**

`titan/tests/test_preflight.py`:
```python
import preflight


def test_torchtitan_config_imports():
    _, ok, _ = preflight.check_torchtitan_import()
    assert ok is True  # torchtitan.config imports even on torch 2.10


def test_train_check_is_bool():
    name, ok, detail = preflight.check_torchtitan_train()
    assert name == "torchtitan.train import"
    assert isinstance(ok, bool)  # False now (torch 2.10), True after rebuild


def test_tokenizer_check_true_for_existing(tmp_path):
    _, ok, _ = preflight.check_tokenizer(str(tmp_path))
    assert ok is True


def test_tokenizer_check_false_for_missing():
    _, ok, _ = preflight.check_tokenizer("/nonexistent/xyz")
    assert ok is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_preflight.py -v'`
Expected: FAIL — `ModuleNotFoundError: No module named 'preflight'`.

- [ ] **Step 3: Write minimal implementation**

`titan/preflight.py`:
```python
import os
import sys
import uuid

MODELS = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models"


def check_torchtitan_import():
    try:
        import torchtitan  # noqa: F401
        return ("torchtitan import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("torchtitan import", False, repr(exc))


def check_torchtitan_train():
    try:
        import torchtitan.train  # noqa: F401
        return ("torchtitan.train import", True, "ok")
    except Exception as exc:  # noqa: BLE001
        return ("torchtitan.train import", False, repr(exc)[:160])


def check_tokenizer(path=f"{MODELS}/Llama-3.1-8B-Instruct"):
    ok = os.path.isdir(path) and os.access(path, os.R_OK)
    return (f"tokenizer:{path}", ok, "ok" if ok else "unreadable")


def check_dir_writable(path):
    try:
        probe = os.path.join(path, f".pf_{uuid.uuid4().hex}")
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


def run_checks():
    return [
        check_torchtitan_import(),
        check_torchtitan_train(),
        check_tokenizer(),
        check_dir_writable("outputs"),
        check_gpu_visible(),
    ]


def main():
    failed = 0
    for name, ok, detail in run_checks():
        status = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"[{status}] {name}: {detail}")
    if failed:
        print(f"\n{failed} check(s) failed. On torch 2.10 the 'torchtitan.train import' "
              f"failure is expected — rebuild the container (torch>=2.11) per container/dtitan.def.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python -m pytest tests/test_preflight.py -v'`
Expected: PASS (4 passed). On the current container `check_torchtitan_train()` returns `False` (the gate); the test only asserts it returns a bool.

- [ ] **Step 5: Manual preflight run (documents the gate)**

Run: `singularity exec "$IMAGE" bash -lc 'cd titan && python preflight.py'`
Expected (current container): `[PASS] torchtitan import`, **`[FAIL] torchtitan.train import`** (torch-2.11 symbol), tokenizer PASS. After the rebuild, all PASS.

- [ ] **Step 6: Commit**

```bash
git add titan/preflight.py titan/tests/test_preflight.py
git commit -m "Add TorchTitan preflight check (detects the torch/torchtitan gate)"
```

---

## Task 5: Level 1 lab guides (`docs/labs/level1/README.md`)  [now — authored; runs are [rebuilt]]

**Files:**
- Create: `titan/docs/labs/level1/README.md`

**Interfaces:**
- Produces: the participant-facing Level 1 guide — the 6 labs + capstone, each with the exact `torchtitan.train` command (built-in `llama3_debugmodel` + overrides), expected artifact, and success criterion.

- [ ] **Step 1: Write the guide**

`titan/docs/labs/level1/README.md` covering, with copy-paste commands:

1. **Preflight** — `python preflight.py` (all checks pass on the rebuilt container).
2. **Inspect + override a config** — `python -c "import torchtitan.config as c; print(c.ConfigManager().parse_args(['--module','llama3','--config','llama3_debugmodel','--training.steps','7']).training.steps)"` → prints `7`.
3. **Fake-backend dry-run** — `NGPU=4 python -m torchtitan.train --module llama3 --config llama3_debugmodel --comm.mode=fake_backend` → resolves + dry-runs without GPUs.
4. **1D FSDP2 run** — `sbatch slurm/launch_1node.sbatch --training.steps=20 --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct` → loss decreases in `outputs/`.
5. **Metrics + profiler** — add `--profiler.enable_profiling` → trace artifact; locate loss/memory/tokens-per-sec/MFU in the log.
6. **Failure-driven** — pass an invalid override (e.g. `--parallelism.tensor_parallel_degree=3` on 4 GPUs) → read the failure, state the root cause, fix it.
- **Capstone** — a short 1D FSDP2 run submitting: launch command, resolved config, training log, a profiler artifact, and a plausibility note on loss/throughput.

Each command is written verbatim; each lab states its success criterion. Mark the section header **"Runs require the rebuilt torch ≥ 2.11 container (Task 1)."**

- [ ] **Step 2: Sanity-check the guide's static command**

Run the config-inspection one-liner (lab 2) — it uses only `torchtitan.config`, which imports on the current container:
`singularity exec "$IMAGE" bash -lc "cd titan && python -c \"import torchtitan.config as c; print(c.ConfigManager().parse_args(['--module','llama3','--config','llama3_debugmodel','--training.steps','7']).training.steps)\""`
Expected: on the rebuilt container prints `7`; on the current container it errors importing the `llama3` config_registry (documents the gate — note this in the guide).

- [ ] **Step 3: Commit**

```bash
git add titan/docs/labs/level1/README.md
git commit -m "Add Level 1 lab guide (config, fake-backend, FSDP2, metrics, debugging)"
```

---

## Task 6: Level 1 validation checklist (`docs/labs/level1/validation.md`)  [rebuilt]

**Files:**
- Create: `titan/docs/labs/level1/validation.md`

**Interfaces:**
- Produces: the exact GPU validation runs for Level 1 on the rebuilt container, with expected results — the "did it actually work on hardware" gate (mirrors how the dtensor workshop was smoke-validated).

- [ ] **Step 1: Write the checklist**

`titan/docs/labs/level1/validation.md` with, for each item, the exact `sbatch` command and expected outcome:

1. **Preflight all-pass** — `singularity exec --nv "$IMAGE" bash -lc 'cd titan && python preflight.py'` → every check PASS (esp. `torchtitan.train import`).
2. **Debug FSDP2 run** — `sbatch slurm/launch_1node.sbatch --training.steps=20` → job COMPLETED; loss trends down in `outputs/`.
3. **Metrics + profiler** — `sbatch slurm/launch_1node.sbatch --training.steps=20 --profiler.enable_profiling` → profiler artifact present; loss/memory/tokens-per-sec/MFU in the log.
4. **Real tokenizer + Llama-3.1-8B taste** — `sbatch slurm/launch_1node.sbatch --training.steps=5 --hf_assets_path=$MODELS/Llama-3.1-8B-Instruct` → an 8B step runs under 1D FSDP2 (short).
5. **Failure lab** — an invalid override fails with a readable error whose root cause is documented.

Note the **8-GPU/user cap** and that jobs use `--account=kempner_dev`.

- [ ] **Step 2: Commit**

```bash
git add titan/docs/labs/level1/validation.md
git commit -m "Add Level 1 GPU validation checklist (rebuilt container)"
```

---

## Self-Review

**1. Spec coverage (Foundation + Level 1 scope):**

| Spec item | Task |
| --- | --- |
| Container rebuild (torch ≥ 2.11 + torchtitan 0.2.2) | Task 1 |
| Offline assets / testbed tokenizer + Llama-3.1-8B | Tasks 4, 5, 6 (Global Constraints) |
| Kempner launchers (account/mem/cpu-bind, torchtitan entry) | Task 3 |
| Preflight (incl. the torchtitan.train gate + fake-backend) | Task 4 (+ lab 3 in Task 5) |
| Config flow (`--module/--config` + overrides, resolve-first) | Tasks 5 (labs 2–3), Global Constraints |
| 1D FSDP2 baseline + metrics + profiler | Task 5 labs 4–5, Task 6 items 2–3 |
| Failure-driven lab (bad config) | Task 5 lab 6 |
| Level 1 capstone | Task 5 |
| Local metrics | Tasks 5/6 (torchtitan built-in logging) |

Deferred to later plans (out of scope here, by design): Level 2 (2D FSDP2+TP, DCP, memory), Level 3 (multi-node, PP/EP/CP, FP8, NCCL debugging), and any custom config-registry entries.

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to Task N". The NGC base tag (`26.01-py3`) is a concrete starting value the builder confirms ships torch ≥ 2.11 (the `%post` import guard fails the build otherwise) — a build-time check, not a placeholder. `<workshop-registered-config>` appears only in the spec's Level 3 example, not this plan.

**3. Type consistency:** Preflight function names are stable (`check_torchtitan_import`, `check_torchtitan_train`, `check_tokenizer`, `check_dir_writable`, `check_gpu_visible`, `run_checks`, `main`, each returning a `(name, ok, detail)` tuple). Launchers, tests, and guides all use the verified `-m torchtitan.train --module llama3 --config llama3_debugmodel` interface and the verified override paths (`--training.steps`, `--parallelism.*`, `--profiler.enable_profiling`), consistent across tasks. Every `[rebuilt]` task states its runtime dependency on Task 1.
