#!/bin/bash
# Level 1 — Lab 1: Preflight Checks (verify the environment is ready for TorchTitan runs)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif
# Run as-is on a login node: 5 of 6 checks pass there. A full 6/6 pass (including
# "gpu visible: device_count=4") requires running this inside a kempner_rtx GPU allocation.
singularity exec "$IMAGE" python preflight.py
