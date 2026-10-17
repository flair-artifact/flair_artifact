#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.11}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: CPython 3.11 is required; set PYTHON_BIN to its executable" >&2
  exit 2
fi
FLAIR_VARIANT=${FLAIR_VARIANT:-main}
case "$FLAIR_VARIANT" in
  main) FLAIR_PACKAGE_ROOT="$ROOT/prebuilt/linux-x86_64-cp311" ;;
  no_label_aware_pruning|no_build_and_query_label_aware_entry_points)
    FLAIR_PACKAGE_ROOT="$ROOT/prebuilt/ablations/linux-x86_64-cp311/$FLAIR_VARIANT" ;;
  *) echo "ERROR: unknown FLAIR_VARIANT: $FLAIR_VARIANT" >&2; exit 2 ;;
esac
if [[ ! -f "$FLAIR_PACKAGE_ROOT/flair/_native.so" ]]; then
  echo "ERROR: selected native package is missing" >&2
  exit 2
fi
export FLAIR_VARIANT FLAIR_PACKAGE_ROOT
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$FLAIR_PACKAGE_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$@"
