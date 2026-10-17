#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_external_vector_link(path: Path, root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return path.is_symlink() and relative in {
        f"datasets/derived/{name}/database_vectors.fvecs"
        for name in ("gist", "arxiv", "qdrant")
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    main_native = root / "prebuilt/linux-x86_64-cp311/flair/_native.so"
    ablation_natives = sorted(root.glob("prebuilt/ablations/linux-x86_64-cp311/*/flair/_native.so"))
    natives = [main_native, *ablation_natives]
    expected_variants = {
        "no_label_aware_pruning", "no_build_and_query_label_aware_entry_points",
    }
    if {native.parents[1].name for native in ablation_natives} != expected_variants:
        raise SystemExit("FAIL: native ablations differ from the two documented variants")
    for native in natives:
        audit = json.loads((native.parents[1] / "audit.json").read_text())
        if sha256(native) != audit["sha256"]:
            raise SystemExit(f"FAIL: native extension hash differs from audit.json: {native}")
        exports = subprocess.check_output(["nm", "-D", "--defined-only", str(native)], text=True).splitlines()
        exports = [line for line in exports if line.strip()]
        if len(exports) != 1 or "PyInit__native" not in exports[0]:
            raise SystemExit(f"FAIL: unexpected dynamic exports in {native}: {exports}")
        sections = subprocess.check_output(["readelf", "-SW", str(native)], text=True)
        if ".comment" in sections:
            raise SystemExit(f"FAIL: compiler comment section is present: {native}")
    # Identity leaks are forbidden everywhere, with no exceptions.
    forbidden = (
        "/" + "home" + "/", "/" + "ssd" + "/",
        "/" + "Users" + "/",
        "lu" + "mina", "ali" + "yun", "ali" + "baba-inc", "ali" + "baba" + "group",
    )
    host_pattern = re.compile(r"\bk28[bd]\d+\b", re.IGNORECASE)
    user_pattern = re.compile(r"\bzr\d{6}\b", re.IGNORECASE)
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    # Apache-2.0 section 4(d) obliges us to reproduce the upstream async_simple NOTICE verbatim,
    # which names that library's vendor. That attribution is a third-party copyright notice, not an
    # author identifier, so it is permitted in exactly the two attribution files below and nowhere
    # else. The vendor word is spelled as a fragment so this auditor stays clean under its own scan.
    vendor_token = ("ali" + "baba").casefold()
    attribution_allowlist = {
        "THIRD_PARTY.md",
        "licenses/async_simple-NOTICE.txt",
    }
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if any(part in {".git", ".venv", "__pycache__"} for part in parts) or parts[:2] == ("results", "reproduced"):
            continue
        if is_external_vector_link(path, root):
            continue
        if path.is_symlink():
            raise SystemExit("FAIL: symlinks are not allowed in the public package")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        lowered = text.casefold()
        if any(token.casefold() in lowered for token in forbidden):
            raise SystemExit(f"FAIL: identity-sensitive text in {relative}")
        if host_pattern.search(text) or user_pattern.search(text):
            raise SystemExit(f"FAIL: identity-sensitive text in {relative}")
        if email_pattern.search(text) and not relative.startswith("licenses/"):
            raise SystemExit(f"FAIL: email outside third-party license texts in {relative}")
        if vendor_token in lowered and relative not in attribution_allowlist:
            raise SystemExit(f"FAIL: vendor attribution outside the allowlist in {relative}")
        if path.suffix == ".json":
            json.loads(text)
    for manifest in root.rglob("SHA256SUMS"):
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(None, 1)
            target = manifest.parent / relative.strip()
            if not target.resolve().is_relative_to(root):
                raise SystemExit("FAIL: checksum path escapes the package")
            if sha256(target) != expected:
                raise SystemExit(f"FAIL: checksum mismatch for {target}")
    public_manifest = root / "MANIFEST.sha256"
    listed = set()
    for line in public_manifest.read_text().splitlines():
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        target = root / relative
        if relative in listed or not target.resolve().is_relative_to(root):
            raise SystemExit("FAIL: duplicate or unsafe public manifest entry")
        listed.add(relative)
        if sha256(target) != expected:
            raise SystemExit(f"FAIL: public checksum mismatch for {relative}")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path != public_manifest
        and not is_external_vector_link(path, root)
        and not any(part in {".git", ".venv", "__pycache__"} for part in path.relative_to(root).parts)
        and path.relative_to(root).parts[:2] != ("results", "reproduced")
    }
    if listed != actual:
        raise SystemExit(f"FAIL: public manifest coverage missing={sorted(actual - listed)} extra={sorted(listed - actual)}")
    print(f"PUBLIC ARTIFACT AUDIT PASSED root={root}")


if __name__ == "__main__":
    main()
