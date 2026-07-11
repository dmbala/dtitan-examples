#!/bin/bash
# Level 1 — Lab 2: Inspect a Config and Apply Overrides (verify dotted CLI overrides resolve)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
IMAGE=/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/dtitan/dtitan-torch211.sif
singularity exec "$IMAGE" python - <<'PY'
import torchtitan.config as c
print(c.ConfigManager().parse_args(['--model.name','llama3','--model.flavor','debugmodel','--training.steps','7']).training.steps)
PY
