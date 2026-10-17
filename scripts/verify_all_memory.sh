#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for config in "$ROOT"/configs/memory/*.json; do
  name=$(basename "$config")
  "$ROOT/run_artifact.sh" -m flair_artifact.verify \
    --config "$config" \
    --actual "$ROOT/results/reproduced/$name"
done
