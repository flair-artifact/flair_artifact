"""Generate controlled-selectivity labels-AND-timestamp predicates for Qdrant."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GIST_TOOLS = ROOT.parent / "texmex_gist1m"
sys.path.insert(0, str(GIST_TOOLS))
from generate_controlled_composite_queries import TARGETS, best_enum_range, write_rows  # noqa: E402


def load_labels(path):
    rows = []
    max_value = -1
    with open(path) as handle:
        for line in handle:
            row = [int(value) for value in line.strip().split(",") if value]
            rows.append(row)
            max_value = max(max_value, max(row, default=-1))
    masks = np.zeros((len(rows), max_value + 1), dtype=bool)
    for index, row in enumerate(rows):
        masks[index, row] = True
    return np.arange(max_value + 1), masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-selectivity", type=int, default=1000)
    parser.add_argument("--main-query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "fanns_format/composite_controlled_smoke")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    data = ROOT / "fanns_format"
    values, masks = load_labels(data / "emis_database_attributes.csv")
    timestamps = np.loadtxt(data / "r_database_attributes.csv", dtype=np.int32)

    rows = []
    for target in TARGETS:
        for _ in range(args.queries_per_selectivity):
            enum_values, bounds, actual = best_enum_range(values, masks, timestamps, target, rng)
            rows.append((target, actual, enum_values, bounds))
        print(f"generated target {target:.0%}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    def write_workload(workload, out):
        out.mkdir(parents=True, exist_ok=True)
        write_rows(out / "query_emis.csv", (row[2] for row in workload))
        write_rows(out / "query_range.csv", (row[3] for row in workload))
        with (out / "selectivity.jsonl").open("w") as handle:
            for target, actual, *_ in workload:
                handle.write(json.dumps({"target": target, "actual": actual}) + "\n")

    write_workload(rows, args.output / "all")
    groups = []
    for index, target in enumerate(TARGETS):
        begin = index * args.queries_per_selectivity
        group = rows[begin:begin + args.queries_per_selectivity]
        groups.append(group)
        write_workload(group, args.output / f"selectivity_{int(target * 100)}")
    counts = [args.main_query_count // 6 + (index < args.main_query_count % 6) for index in range(6)]
    mixed, source = [], []
    permutation = rng.permutation(args.queries_per_selectivity)
    cursor = 0
    for group, count in zip(groups, counts):
        if cursor + count <= len(permutation):
            chosen = permutation[cursor:cursor + count]
        else:
            chosen = rng.choice(len(group), count, replace=count > len(group))
        mixed.extend(group[index] for index in chosen)
        source.extend(int(index) for index in chosen)
        cursor = (cursor + count) % len(permutation)
    write_workload(mixed, args.output / "mixed")
    write_rows(args.output / "mixed/query_source_indices.csv", ([index] for index in source))
    errors = [abs(target - actual) for target, actual, *_ in rows]
    print(f"enum_range: {len(rows)} queries, max absolute error={max(errors):.6f}")


if __name__ == "__main__":
    main()
