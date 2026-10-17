#!/usr/bin/env python3
"""Verify locally downloaded upstream dataset files against the artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gist", "arxiv", "qdrant"), required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory containing the upstream files named in the manifest")
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "public_files.json",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    failures = []
    for entry in manifest[args.dataset]["files"]:
        path = args.root / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        size = path.stat().st_size
        if size != entry["bytes"]:
            failures.append(f"size: {path} expected={entry['bytes']} actual={size}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            failures.append(f"sha256: {path} expected={entry['sha256']} actual={actual}")
        else:
            print(f"OK {entry['path']} bytes={size} sha256={actual}")
    if failures:
        raise SystemExit("PUBLIC FILE VERIFICATION FAILED\n" + "\n".join(failures))
    print(f"PUBLIC FILE VERIFICATION PASSED dataset={args.dataset}")


if __name__ == "__main__":
    main()
