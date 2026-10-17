#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset root}"
mkdir -p "$ROOT/results/reproduced"
for config in "$ROOT"/configs/memory/*.json; do
  name=$(basename "$config" .json)
  "$ROOT/run_artifact.sh" -m flair_artifact.run \
    --config "$config" \
    --output "$ROOT/results/reproduced/$name.json"
done
