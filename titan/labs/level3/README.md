# Level 3 — Production Scaling & Debugging (lab scripts)

Scale to a real 3D mesh across 8 GPUs and add the production tools: pipeline parallelism, a
Mixture-of-Experts model with expert parallelism, FP8 quantization, and NCCL/Flight-Recorder
debugging — the fourth milestone: **debuggable**.

**Full teaching write-up:** [`../../docs/labs/level3/README.md`](../../docs/labs/level3/README.md)
· **Validation checklist:** [`../../docs/labs/level3/validation.md`](../../docs/labs/level3/validation.md)

**Hardware:** all labs use `../../slurm/launch_8gpu.sbatch` (1 node / **8 GPUs**, `kempner_rtx`) —
RTX nodes have 8 GPUs/node, so the 8-GPU 3D mesh fits on one node. Requires the one-time
`../level1/00_prepare_data.sh`. Job output → `../../outputs/<jobname>-<jobid>.out`.

| Script | Teaches | Validated result |
|--------|---------|------------------|
| `01_hsdp_tp.sh` | real 3D mesh: `dp_replicate=2 × dp_shard=2 × tp=2` | loss **6.22 → 3.57**; mesh built |
| `02_pipeline.sh` | pipeline parallel (`pp=2 × dp_shard=4`) | completes; **loss is only meaningful on the last stage** (other ranks log a sentinel) |
| `03_moe_ep.sh` | Mixture-of-Experts (`deepseek_v3`) + expert parallel | loss **4.48 → 3.47** |
| `04_fp8.sh` | FP8 linear via torchao on Blackwell | loss **8.02 → 4.62**; "Float8 tensorwise scaled training active" |
| `05_flight_recorder.sh` | NCCL Flight-Recorder ring buffer for debugging | comm traces under `outputs/<...>/comm_traces/` |
| `06_capstone.sh` | HSDP+TP + FP8 + DCP (Option A) / MoE+EP (Option B) | Option A validated end-to-end (mesh + Float8 + checkpoints step-10/20) |

**Two gotchas the labs teach (Lab 3, MoE):**
- **Expert parallel is orthogonal to the world-size product** — the assert is
  `dp_replicate*dp_shard*cp*tp*pp == world_size` (no `ep`). On 8 GPUs use `dp_shard=8 + ep=2`,
  not `dp_shard=4 × ep=2` (which fails `Invalid parallel dims … != WORLD_SIZE(8)`).
- **The MoE grouped-GEMM Triton kernel needs `libcuda.so`** — `launch_8gpu.sbatch` sets
  `TRITON_LIBCUDA_PATH=/.singularity.d/libs` for you; without it the kernel fails
  `AssertionError: libcuda.so cannot found!`.

Also note: FP8's converter name is **`quantize.linear.float8`** (not `float8`). The `70B/405B`
and `236B/671B` flavors are for topology **planning** only — they don't fit for training here.

```bash
bash ../level1/00_prepare_data.sh    # once, if you haven't
bash 03_moe_ep.sh                    # submit; then: tail -f ../../outputs/titan-l3-<id>.out
```
