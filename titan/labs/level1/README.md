# Level 1 — Run & Observe (lab scripts)

Launch your first TorchTitan runs and learn to read them: submit a 1D FSDP2 job, inspect
and override a config, dry-run with the fake backend, watch loss/throughput/MFU and a
profiler trace, and break a config on purpose to read the error.

**Full teaching write-up:** [`../../docs/labs/level1/README.md`](../../docs/labs/level1/README.md)
· **Validation checklist:** [`../../docs/labs/level1/validation.md`](../../docs/labs/level1/validation.md)

**Hardware:** training labs use `../../slurm/launch_1node.sbatch` (1 node / **4 GPUs**, `kempner_rtx`).
Labs 1–2 run on the **login node** (no GPU). Job output lands in `../../outputs/<jobname>-<jobid>.out`.

**Run `00_prepare_data.sh` once first** — it generates `../../assets/c4_subset/` (the local C4 subset every training lab reads).

| Script | Teaches | Validated result |
|--------|---------|------------------|
| `00_prepare_data.sh` | one-time: extract a real-C4 subset from the testbed | writes `assets/c4_subset/train.jsonl` (3000 docs) |
| `01_preflight.sh` | environment self-check (login node) | 5/6 PASS on login; `gpu visible` needs a GPU alloc |
| `02_inspect_config.sh` | resolve `--model.*`/`--training.*` overrides without launching | prints the resolved value (e.g. `training.steps=7`) |
| `03_fake_backend.sh` | dry-run the config/model with faked collectives (1 process) | "Applied FSDP to the model" → "Training completed" |
| `04_fsdp2.sh` | first real 1D FSDP2 run across 4 GPUs | loss **8.12 → 3.55** over 20 steps; "Training completed" |
| `05_profiler.sh` | metrics + a profiler trace | `outputs/profile_traces/iteration_10/rank*_trace.json` |
| `06_failure.sh` | **intentional failure** — invalid `tensor_parallel_degree=3` on 4 GPUs | readable `AssertionError: Invalid parallel dims … != WORLD_SIZE(4)` |
| `07_capstone.sh` | full observable run with profiling artifacts | loss decreases, trace written, clean exit |

```bash
bash 00_prepare_data.sh      # once
bash 04_fsdp2.sh             # submit; then: tail -f ../../outputs/titan-l12-<id>.out
```
