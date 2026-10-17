from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .run import _expanded_path, execute


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a declared FLAIR parameter sweep")
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    declaration = json.loads(args.sweep.read_text())
    base_path = (args.sweep.parent / declaration["base_config"]).resolve()
    base = json.loads(base_path.read_text())
    # Preserve the base configuration's path semantics when moving generated configs.
    for key in ("base_vectors", "query_vectors", "groundtruth"):
        base["dataset"][key] = str(_expanded_path(base["dataset"][key], base_path.parent).resolve())
    for attribute in base["dataset"]["attributes"]:
        for key in ("database", "queries"):
            attribute[key] = str(_expanded_path(attribute[key], base_path.parent).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [v["name"] for v in declaration["variants"]]
    if len(set(names)) != len(names) or any(not n or Path(n).name != n or n in {".", ".."} for n in names):
        raise ValueError("variant names must be unique file names")
    for variant in declaration["variants"]:
        config = copy.deepcopy(base)
        config["name"] = f"{base['name']}/{variant['name']}"
        config["index"].setdefault("build_options", {}).update(variant["build_options"])
        config["index"]["path"] = str((args.output_dir / f"{variant['name']}.idx").resolve())
        generated = args.output_dir / f"{variant['name']}.config.json"
        generated.write_text(json.dumps(config, indent=2) + "\n")
        execute(generated, args.output_dir / f"{variant['name']}.json")
        print(f"PASS variant={variant['name']}")


if __name__ == "__main__":
    main()
