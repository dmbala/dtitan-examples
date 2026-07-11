#!/bin/bash
# Level 1 — Lab 0: Prepare C4 subset (generates the local training data every training lab needs)
set -euo pipefail
cd "$(dirname "$0")/../.."   # -> titan/
python scripts/prepare_c4_subset.py
