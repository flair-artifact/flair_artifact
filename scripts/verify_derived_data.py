#!/usr/bin/env python3
"""Verify the bundled derived workload data and its checksums.

Three checks, in order:

1. every file listed in ``datasets/derived/SHA256SUMS`` and
   ``datasets/generators/SHA256SUMS`` matches its recorded SHA-256, and no
   unlisted file is present in those trees;
2. the structural contract in ``datasets/derived/MANIFEST.json`` holds: row
   counts agree across query vectors, predicates, and ground truth, ground-truth
   width matches the recorded top-k, and every ground-truth id is in range;
3. every ``${DATA_ROOT}`` path referenced by ``configs/memory/*.json`` resolves
   inside the derived tree, except the base vector files, which the reviewer
   links from the verified public downloads (see ``scripts/prepare_data_root.sh``).

Usage::

    python3 scripts/verify_derived_data.py
    python3 scripts/verify_derived_data.py --data-root /path/to/DATA_ROOT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest_file(root: Path, manifest: Path) -> int:
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(None, 1)
        listed[relative.strip()] = digest
    for relative, expected in sorted(listed.items()):
        target = root / relative
        if not target.is_file():
            raise SystemExit(f"FAIL: listed file is missing: {relative}")
        actual = sha256(target)
        if actual != expected:
            raise SystemExit(f"FAIL: checksum mismatch for {relative}\n"
                             f"  expected {expected}\n  actual   {actual}")
    # Symlinks are reviewer-created links to the public base vectors
    # (scripts/prepare_data_root.sh); they are not bundled content.
    present = {path.relative_to(root).as_posix() for path in root.rglob("*")
               if path.is_file() and not path.is_symlink()
               and path.name != manifest.name and "__pycache__" not in path.parts}
    unlisted = sorted(present - set(listed))
    if unlisted:
        raise SystemExit(f"FAIL: {len(unlisted)} file(s) not covered by {manifest.name}: "
                         + ", ".join(unlisted[:5]))
    return len(listed)


def fvecs_rows(path: Path) -> tuple[int, int]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        dim = struct.unpack("<i", handle.read(4))[0]
    record = 4 + dim * 4
    if size % record:
        raise SystemExit(f"FAIL: {path} is not a whole number of fvecs records")
    return size // record, dim


def ivecs_shape(path: Path) -> tuple[int, int, int, int]:
    rows = width = 0
    low: int | None = None
    high: int | None = None
    with path.open("rb") as handle:
        while True:
            header = handle.read(4)
            if not header:
                break
            count = struct.unpack("<i", header)[0]
            payload = handle.read(4 * count)
            if len(payload) != 4 * count:
                raise SystemExit(f"FAIL: truncated ivecs row in {path}")
            if rows == 0:
                width = count
            elif count != width:
                raise SystemExit(f"FAIL: ragged ivecs rows in {path}")
            ids = struct.unpack(f"<{count}i", payload)
            low = min(ids) if low is None else min(low, min(ids))
            high = max(ids) if high is None else max(high, max(ids))
            rows += 1
    return rows, width, low if low is not None else 0, high if high is not None else -1


def csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def check_structure(data_root: Path, manifest: dict) -> None:
    for name, spec in sorted(manifest.items()):
        root = data_root / name
        if not root.is_dir():
            raise SystemExit(f"FAIL: dataset directory missing: {root}")
        vectors, queries = spec["vectors"], spec["queries"]
        for relative, expected in sorted(spec["database"].items()):
            rows = csv_rows(root / "database" / relative)
            if rows != vectors or rows != expected["rows"]:
                raise SystemExit(f"FAIL: {name}/database/{relative} has {rows} rows, "
                                 f"expected {vectors}")
        shared = root / "bins_query_vectors.fvecs"
        rows, dim = fvecs_rows(shared)
        if (rows, dim) != (queries, spec["bins_query_vectors"]["dim"]):
            raise SystemExit(f"FAIL: {name}/bins_query_vectors.fvecs shape {(rows, dim)}")
        if sha256(shared) != spec["bins_query_vectors"]["sha256"]:
            raise SystemExit(f"FAIL: {name}/bins_query_vectors.fvecs checksum")
        for scenario, record in sorted(spec["scenarios"].items()):
            base = root / scenario
            rows, dim = fvecs_rows(base / "query_vectors.fvecs")
            if rows != queries or dim != record["dim"]:
                raise SystemExit(f"FAIL: {name}/{scenario}/query_vectors.fvecs shape {(rows, dim)}")
            truth, width, low, high = ivecs_shape(base / "groundtruth.ivecs")
            if truth != queries or width != record["gt_width"]:
                raise SystemExit(f"FAIL: {name}/{scenario}/groundtruth.ivecs shape {(truth, width)}")
            if low < 0 or high >= vectors:
                raise SystemExit(f"FAIL: {name}/{scenario} ground-truth ids out of range [{low}, {high}]")
            if jsonl_rows(base / "selectivity.jsonl") != queries:
                raise SystemExit(f"FAIL: {name}/{scenario}/selectivity.jsonl row count")
            for attribute in record["attributes"]:
                if csv_rows(base / attribute["query"]) != queries:
                    raise SystemExit(f"FAIL: {name}/{scenario}/{attribute['query']} row count")
                if not (root / attribute["database"]).is_file():
                    raise SystemExit(f"FAIL: {name}/{attribute['database']} missing")
            for bin_dir in sorted((base / "bins").iterdir()):
                truth, width, low, high = ivecs_shape(bin_dir / "groundtruth.ivecs")
                if truth != queries or width != record["gt_width"]:
                    raise SystemExit(f"FAIL: {bin_dir} ground truth shape {(truth, width)}")
                if low < 0 or high >= vectors:
                    raise SystemExit(f"FAIL: {bin_dir} ground-truth ids out of range")
                if jsonl_rows(bin_dir / "selectivity.jsonl") != queries:
                    raise SystemExit(f"FAIL: {bin_dir}/selectivity.jsonl row count")
                for attribute in record["attributes"]:
                    if csv_rows(bin_dir / attribute["query"]) != queries:
                        raise SystemExit(f"FAIL: {bin_dir}/{attribute['query']} row count")
        print(f"  {name}: scenarios={len(spec['scenarios'])} vectors={vectors} "
              f"queries={queries} dim={spec['bins_query_vectors']['dim']}")


def check_configs(data_root: Path) -> list[str]:
    referenced: set[str] = set()
    for config_path in sorted((ROOT / "configs" / "memory").glob("*.json")):
        text = config_path.read_text()
        config = json.loads(text)
        dataset = config["dataset"]
        paths = [dataset["base_vectors"], dataset["query_vectors"], dataset["groundtruth"]]
        paths += [attribute["database"] for attribute in dataset["attributes"]]
        paths += [attribute["queries"] for attribute in dataset["attributes"]]
        for value in paths:
            if not value.startswith("${DATA_ROOT}/"):
                raise SystemExit(f"FAIL: {config_path.name} references a non-DATA_ROOT path: {value}")
            referenced.add(value[len("${DATA_ROOT}/"):])
    missing = sorted(item for item in referenced if not (data_root / item).is_file())
    unexpected = [item for item in missing if not item.endswith("database_vectors.fvecs")]
    if unexpected:
        raise SystemExit("FAIL: config inputs missing from the derived tree: " + ", ".join(unexpected))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets" / "derived")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"FAIL: data root not found: {data_root}")

    print("[1/3] checksums")
    derived = check_manifest_file(data_root, data_root / "SHA256SUMS")
    generators = ROOT / "datasets" / "generators"
    gens = check_manifest_file(generators, generators / "SHA256SUMS")
    print(f"  derived files verified: {derived}")
    print(f"  generator files verified: {gens}")

    print("[2/3] structural contract")
    manifest = json.loads((data_root / "MANIFEST.json").read_text())
    check_structure(data_root, manifest)

    print("[3/3] configuration inputs")
    pending = check_configs(data_root)
    for item in pending:
        print(f"  awaiting public vectors: {item} (run scripts/prepare_data_root.sh)")
    print("DERIVED DATA VERIFICATION PASSED")


if __name__ == "__main__":
    sys.exit(main())
