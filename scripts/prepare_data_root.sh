#!/usr/bin/env bash
# Link the verified public base-vector files into a DATA_ROOT that already holds
# the bundled derived attributes, predicates, and ground truth.
#
# The derived tree ships everything except the base vectors themselves, which are
# multi-gigabyte redistributions of public datasets.  Point each flag at the
# public file you downloaded and verified with scripts/verify_public_files.py.
#
# Usage:
#   scripts/prepare_data_root.sh --data-root datasets/derived \
#     --gist-vectors /downloads/gist_base.fvecs \
#     [--arxiv-vectors /downloads/database_vectors.fvecs] \
#     [--qdrant-vectors /downloads/qdrant_database_vectors.fvecs]
#
# Qdrant ships vectors.npy upstream; convert it first with
#   python3 datasets/generators/filtered_ann_datasets/qdrant_arxiv/convert_to_fanns.py
# which writes fanns_format/database_vectors.fvecs.
set -euo pipefail

DATA_ROOT=""
declare -A TARGETS=()

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT=${2:?}; shift 2 ;;
    --gist-vectors) TARGETS[gist]=${2:?}; shift 2 ;;
    --arxiv-vectors) TARGETS[arxiv]=${2:?}; shift 2 ;;
    --qdrant-vectors) TARGETS[qdrant]=${2:?}; shift 2 ;;
    -h|--help) usage ;;
    *) echo "ERROR: unknown argument $1" >&2; usage ;;
  esac
done

[[ -n $DATA_ROOT ]] || { echo "ERROR: --data-root is required" >&2; exit 2; }
[[ -d $DATA_ROOT ]] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 2; }
DATA_ROOT=$(cd "$DATA_ROOT" && pwd)

linked=0
for dataset in "${!TARGETS[@]}"; do
  source=${TARGETS[$dataset]}
  [[ -f $source ]] || { echo "ERROR: not a regular file: $source" >&2; exit 1; }
  source=$(cd "$(dirname "$source")" && pwd)/$(basename "$source")
  destination=$DATA_ROOT/$dataset/database_vectors.fvecs
  mkdir -p "$(dirname "$destination")"
  if [[ -e $destination && ! -L $destination ]]; then
    echo "ERROR: refusing to overwrite real file $destination" >&2
    exit 1
  fi
  ln -sfn "$source" "$destination"
  echo "LINKED $dataset/database_vectors.fvecs -> $source"
  linked=$((linked + 1))
done

if [[ $linked -eq 0 ]]; then
  echo "ERROR: no vector files given; nothing to link" >&2
  usage
fi

for dataset in "${!TARGETS[@]}"; do
  for scenario in "$DATA_ROOT/$dataset"/*/; do
    [[ -f ${scenario}groundtruth.ivecs ]] || continue
    echo "READY $dataset/$(basename "$scenario")"
  done
done
echo "DATA ROOT PREPARED root=$DATA_ROOT linked=$linked"
