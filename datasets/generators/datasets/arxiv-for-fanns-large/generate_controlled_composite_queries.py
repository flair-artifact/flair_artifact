"""Generate controlled-selectivity composite predicates for Arxiv-large."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GIST_TOOLS = ROOT.parents[1] / "filtered_ann_datasets" / "texmex_gist1m"
sys.path.insert(0, str(GIST_TOOLS))
from generate_controlled_composite_queries import (  # noqa: E402
    TARGETS, best_enum_pair, best_enum_range, best_range_pair, write_rows,
)


def write_workload(name, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    fields = {
        "enum_enum": (("query_em.csv", 2), ("query_emis.csv", 3)),
        "range_range": (("query_r1.csv", 2), ("query_r2.csv", 3)),
        "enum_range": (("query_emis.csv", 2), ("query_r1.csv", 3)),
    }[name]
    for filename, column in fields:
        write_rows(out / filename, (row[column] for row in rows))
    with (out / "selectivity.jsonl").open("w") as handle:
        for target, actual, *_ in rows:
            handle.write(json.dumps({"target": target, "actual": actual}) + "\n")


def load_enum_memberships(path, scalar=False):
    if scalar:
        column = np.loadtxt(path, dtype=np.int32)
        values = np.unique(column)
        return values, np.column_stack([column == value for value in values])
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


def choose_with_tolerance(factory, target):
    tolerance = 0.002 if target <= 0.10 else 0.005
    best = factory()
    for _ in range(3):
        if abs(best[-1] - target) <= tolerance:
            break
        candidate = factory()
        if abs(candidate[-1] - target) < abs(best[-1] - target):
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-selectivity", type=int, default=1000)
    parser.add_argument("--main-query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "composite_controlled_smoke")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    em_values, em_masks = load_enum_memberships(ROOT / "em_database_attributes.csv", True)
    emis_values, emis_masks = load_enum_memberships(ROOT / "emis_database_attributes.csv")
    range1 = np.loadtxt(ROOT / "r_database_attributes.csv", dtype=np.int32)
    range2 = np.loadtxt(ROOT / "r2_database_attributes.csv", dtype=np.int32)
    range1_sorted, range2_sorted = np.sort(range1), np.sort(range2)
    scenarios = {"enum_enum": [], "range_range": [], "enum_range": []}

    for target in TARGETS:
        for _ in range(args.queries_per_selectivity):
            a, b, actual = choose_with_tolerance(
                lambda: best_enum_pair(em_values, em_masks, emis_values, emis_masks, target, rng), target
            )
            scenarios["enum_enum"].append((target, actual, a, b))
            a, b, actual = choose_with_tolerance(
                lambda: best_range_pair(range1, range2, range1_sorted, range2_sorted, target, rng), target
            )
            scenarios["range_range"].append((target, actual, a, b))
            a, b, actual = choose_with_tolerance(
                lambda: best_enum_range(emis_values, emis_masks, range1, target, rng), target
            )
            scenarios["enum_range"].append((target, actual, a, b))
        print(f"generated target {target:.0%}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in scenarios.items():
        root = args.output / name
        write_workload(name, rows, root / "all")
        groups = []
        for index, target in enumerate(TARGETS):
            begin = index * args.queries_per_selectivity
            group = rows[begin:begin + args.queries_per_selectivity]
            groups.append(group)
            write_workload(name, group, root / f"selectivity_{int(target * 100)}")
        counts = [args.main_query_count // 6 + (index < args.main_query_count % 6) for index in range(6)]
        mixed = []
        source = []
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
        write_workload(name, mixed, root / "mixed")
        write_rows(root / "mixed/query_source_indices.csv", ([index] for index in source))
        errors = [abs(target - actual) for target, actual, *_ in rows]
        print(f"{name}: {len(rows)} queries, max absolute error={max(errors):.6f}")


if __name__ == "__main__":
    main()
