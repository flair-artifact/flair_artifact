"""Generate controlled-selectivity two-attribute workloads for GIST1M.

This first stage writes predicates and measured selectivities only.  Ground
truth is deliberately generated in a separate stage after validation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


TARGETS = (0.01, 0.05, 0.10, 0.20, 0.40, 0.60)
ROOT = Path(__file__).resolve().parent


def enum_membership(column):
    values = np.unique(column)
    values = values[values >= 0]
    masks = np.column_stack([
        column == value if column.ndim == 1 else np.any(column == value, axis=1)
        for value in values
    ])
    return values, masks


def best_enum_pair(left_values, left_masks, right_values, right_masks, target, rng, attempts=4):
    step = max(1, left_masks.shape[0] // 100000)
    sample_left, sample_right = left_masks[::step], right_masks[::step]
    best = None
    for _ in range(attempts):
        lorder, rorder = rng.permutation(left_values), rng.permutation(right_values)
        lmasks, mask = [], np.zeros(sample_left.shape[0], dtype=bool)
        for value in lorder:
            mask |= sample_left[:, np.where(left_values == value)[0][0]]
            lmasks.append(mask.copy())
        rmask = np.zeros(sample_right.shape[0], dtype=bool)
        for rcount, value in enumerate(rorder, 1):
            rmask |= sample_right[:, np.where(right_values == value)[0][0]]
            candidates = sorted(
                range(len(lmasks)), key=lambda i: abs(lmasks[i].mean() * rmask.mean() - target)
            )[:3]
            for index in candidates:
                lcount, lmask = index + 1, lmasks[index]
                actual = np.mean(lmask & rmask)
                error = abs(actual - target)
                if best is None or error < best[0]:
                    best = (error, lorder[:lcount].copy(), rorder[:rcount].copy(), actual)
    _, lvalues, rvalues, _ = best
    full = np.any(left_masks[:, np.isin(left_values, lvalues)], axis=1)
    full &= np.any(right_masks[:, np.isin(right_values, rvalues)], axis=1)
    return lvalues, rvalues, full.mean()


def best_range_pair(left, right, left_sorted, right_sorted, target, rng, attempts=12):
    width = np.sqrt(target)
    best = None
    step = max(1, len(left) // 100000)
    sample_left, sample_right = left[::step], right[::step]
    for _ in range(attempts):
        ql = rng.uniform(0, 1 - width, size=2)
        li = (np.array([ql[0], ql[0] + width]) * (len(left) - 1)).astype(int)
        ri = (np.array([ql[1], ql[1] + width]) * (len(right) - 1)).astype(int)
        llo, lhi = left_sorted[li]
        rlo, rhi = right_sorted[ri]
        actual = np.mean(
            (sample_left >= llo) & (sample_left <= lhi)
            & (sample_right >= rlo) & (sample_right <= rhi)
        )
        error = abs(actual - target)
        if best is None or error < best[0]:
            best = (error, (llo, lhi), (rlo, rhi), actual)
    _, lbounds, rbounds, _ = best
    actual = np.mean(
        (left >= lbounds[0]) & (left <= lbounds[1])
        & (right >= rbounds[0]) & (right <= rbounds[1])
    )
    return lbounds, rbounds, actual


def best_enum_range(values, memberships, ranged, target, rng, attempts=8):
    best = None
    step = max(1, len(ranged) // 100000)
    sample_memberships, sample_range = memberships[::step], ranged[::step]
    sorted_range = np.sort(ranged)
    for _ in range(attempts):
        order = rng.permutation(values)
        emask = np.zeros(sample_memberships.shape[0], dtype=bool)
        for count, value in enumerate(order, 1):
            emask |= sample_memberships[:, np.where(values == value)[0][0]]
            coverage = emask.mean()
            if coverage < target:
                continue
            conditional = min(1.0, target / coverage)
            start = rng.uniform(0, 1 - conditional) if conditional < 1 else 0.0
            indexes = (np.array([start, start + conditional]) * (len(sorted_range) - 1)).astype(int)
            lo, hi = sorted_range[indexes]
            actual = np.mean(emask & (sample_range >= lo) & (sample_range <= hi))
            error = abs(actual - target)
            if best is None or error < best[0]:
                best = (error, order[:count].copy(), (lo, hi), actual)
            # Once the enum predicate is broad enough, the range quantiles
            # already provide the requested joint cardinality.  Trying still
            # larger IN lists only increases query parsing cost.
            break
    _, enum_values, bounds, _ = best
    full = np.any(memberships[:, np.isin(values, enum_values)], axis=1)
    full &= (ranged >= bounds[0]) & (ranged <= bounds[1])
    return enum_values, bounds, full.mean()


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


def ordered_float(value):
    """Match the FLAIR range-label order-preserving uint32 encoding."""
    value = np.float32(0.0 if value == 0.0 else value)
    bits = int(value.view(np.uint32))
    return bits ^ (0xFFFFFFFF if bits & 0x80000000 else 0x80000000)


def encoded_float_range(bounds):
    return ordered_float(bounds[0]), ordered_float(bounds[1])


def write_scenario_rows(name, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    if name == "enum_enum":
        write_rows(out / "query_enum1.csv", (row[2] for row in rows))
        write_rows(out / "query_enum2.csv", (row[3] for row in rows))
    elif name == "range_range":
        write_rows(out / "query_range_int.csv", (row[2] for row in rows))
        write_rows(out / "query_range_float.csv", (encoded_float_range(row[3]) for row in rows))
    else:
        write_rows(out / "query_enum2.csv", (row[2] for row in rows))
        write_rows(out / "query_range_int.csv", (row[3] for row in rows))
    with (out / "selectivity.jsonl").open("w") as handle:
        for target, actual, *_ in rows:
            handle.write(json.dumps({"target": target, "actual": actual}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-per-selectivity", type=int, default=1000)
    parser.add_argument("--main-query-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "composite_controlled_smoke")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    enum1 = np.load(ROOT / "gist_enum1_attributes.npy", mmap_mode="r")
    enum2 = np.load(ROOT / "enum2_k20_candidate/gist_enum2_attributes.npy", mmap_mode="r")
    range1 = np.load(ROOT / "gist_attributes.npy", mmap_mode="r")
    range2 = np.load(ROOT / "gist_float_attributes.npy", mmap_mode="r")
    enum1_values, enum1_masks = enum_membership(enum1)
    enum2_values, enum2_masks = enum_membership(enum2)
    range1_sorted, range2_sorted = np.sort(range1), np.sort(range2)
    scenarios = {"enum_enum": [], "range_range": [], "enum_range": []}

    for target in TARGETS:
        for _ in range(args.queries_per_selectivity):
            a, b, actual = best_enum_pair(
                enum1_values, enum1_masks, enum2_values, enum2_masks, target, rng
            )
            scenarios["enum_enum"].append((target, actual, a, b))
            a, b, actual = best_range_pair(range1, range2, range1_sorted, range2_sorted, target, rng)
            scenarios["range_range"].append((target, actual, a, b))
            a, b, actual = best_enum_range(enum2_values, enum2_masks, range1, target, rng)
            scenarios["enum_range"].append((target, actual, a, b))
        print(f"generated target {target:.0%}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in scenarios.items():
        out = args.output / name
        write_scenario_rows(name, rows, out / "all")
        per_target = []
        offset = 0
        for target in TARGETS:
            group = rows[offset:offset + args.queries_per_selectivity]
            offset += args.queries_per_selectivity
            write_scenario_rows(name, group, out / f"selectivity_{int(target * 100)}")
            per_target.append(group)
        counts = [args.main_query_count // len(TARGETS)] * len(TARGETS)
        for index in range(args.main_query_count % len(TARGETS)):
            counts[index] += 1
        mixed = []
        source_indices = []
        permutation = rng.permutation(args.queries_per_selectivity)
        cursor = 0
        for group, count in zip(per_target, counts):
            chosen = permutation[cursor:cursor + count] if cursor + count <= len(permutation) else rng.choice(len(group), count, replace=False)
            mixed.extend(group[index] for index in chosen)
            source_indices.extend(int(index) for index in chosen)
            cursor = (cursor + count) % len(permutation)
        write_scenario_rows(name, mixed, out / "mixed")
        write_rows(out / "mixed/query_source_indices.csv", ([index] for index in source_indices))
        errors = [abs(target - actual) for target, actual, *_ in rows]
        print(f"{name}: {len(rows)} queries, max absolute error={max(errors):.6f}")

    # Database columns shared by all workloads.  The float encoding must be
    # identical to the query-bound encoding so integer comparison preserves
    # the original numeric order.
    write_rows(args.output / "database_enum1.csv", ([int(v)] for v in enum1))
    write_rows(args.output / "database_enum2.csv", (row[row >= 0].tolist() for row in enum2))
    write_rows(args.output / "database_range_int.csv", ([int(v)] for v in range1))
    write_rows(args.output / "database_range_float.csv", ([ordered_float(v)] for v in range2))


if __name__ == "__main__":
    main()
