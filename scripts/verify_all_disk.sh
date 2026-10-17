#!/usr/bin/env bash
set -euo pipefail

export FLAIR_VARIANT=main

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for config in "$ROOT"/configs/disk/*.json; do
  name=$(basename "$config")
  "$ROOT/run_artifact.sh" -m flair_artifact.verify \
    --config "$config" \
    --actual "$ROOT/results/reproduced/disk/$name"
done
