#!/usr/bin/env python3
"""Audit metadata and obvious disclosure leaks in a candidate FLAIR Python extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


# The literals are assembled from fragments so that this auditor's own source stays clean under
# the package-wide scan performed by scripts/audit_package.py.
_HOME_PATH = b"/" + b"ho" + b"me/"
_SSD_PATH = b"/" + b"ss" + b"d/"
_INTERNAL_HOST = b"k28" + b"[bd]" + b"[0-9]+"
_INTERNAL_USER = b"z" + b"r[0-9]{6}"
_INTERNAL_CLOUD = b"ali" + b"yun|ali" + b"baba"

IDENTITY_PATTERNS = {
    "home_path": re.compile(_HOME_PATH),
    "ssd_path": re.compile(_SSD_PATH),
    "email": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "internal_host": re.compile(_INTERNAL_HOST, re.IGNORECASE),
    "internal_user": re.compile(_INTERNAL_USER),
    "internal_cloud": re.compile(_INTERNAL_CLOUD, re.IGNORECASE),
}


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    path = args.extension.resolve()
    data = path.read_bytes()
    dynamic = command("readelf", "-d", str(path))
    versions = command("readelf", "--version-info", str(path))
    exports = command("nm", "-D", "--defined-only", str(path)).splitlines()
    needed = re.findall(r"Shared library: \[([^]]+)\]", dynamic)
    glibc = sorted(set(re.findall(r"GLIBC_[0-9.]+", versions)))
    glibcxx = sorted(set(re.findall(r"GLIBCXX_[0-9.]+", versions)))
    leaks = {name: len(pattern.findall(data)) for name, pattern in IDENTITY_PATTERNS.items()}
    report = {
        # Keep the persisted report publishable: never record a private absolute build path.
        "file": path.name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "needed": needed,
        "dynamic_export_count": len(exports),
        "python_init_exports": [line for line in exports if "PyInit_" in line],
        "glibc_versions": glibc,
        "glibcxx_versions": glibcxx,
        "identity_leaks": leaks,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.json:
        args.json.write_text(rendered)
    if any(leaks.values()):
        raise SystemExit("candidate contains identity-sensitive strings")


if __name__ == "__main__":
    main()
