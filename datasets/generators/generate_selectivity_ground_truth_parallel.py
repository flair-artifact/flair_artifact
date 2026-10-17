"""Exact, checkpointable controlled-bin GT with shared vector scoring.

Two workers handle disjoint query ranges. Each database block is filtered to
the union of candidates for the current queries before computing distances;
each score is reused across all relevant selectivity bins and scenarios.
"""
import argparse
import os
from pathlib import Path

# Set before importing numpy (and thus before loading the BLAS runtime).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from generate_controlled_ground_truth import (TARGETS, dataset_config, read_rows,
                                              write_fvecs, write_ivecs)


def workloads(root, configs, count):
    items = []
    for scenario, specs in configs.items():
        base = root if scenario == "." else root / scenario
        for target in TARGETS:
            directory = base / f"selectivity_{target}"
            rows = {spec["query"]: read_rows(directory / spec["query"])
                    for spec in specs}
            if any(len(row) != count for row in rows.values()):
                raise ValueError(f"Unexpected query count in {directory}")
            items.append((scenario, target, directory, specs, rows))
    return items


def candidate_mask(specs, rows, query_id, start, end):
    mask = np.ones(end - start, dtype=bool)
    for spec in specs:
        values = rows[spec["query"]][query_id]
        attr = spec["database"][start:end]
        if spec["kind"] == "enum":
            if attr.ndim == 1:
                mask &= np.isin(attr, values)
            else:
                mask &= np.any(attr[:, values], axis=1)
        else:
            low, high = values
            mask &= (attr >= low) & (attr <= high)
    return mask


def exact_batch(database, vectors, items, k, metric, begin, end, block_size):
    """Compute all workload top-k for one query batch in a single DB pass."""
    width = end - begin
    best_scores = np.full((len(items), width, k), -np.inf, dtype=np.float32)
    best_ids = np.full((len(items), width, k), -1, dtype=np.int32)
    queries = np.asarray(vectors[begin:end], dtype=np.float32)
    for start in range(0, len(database), block_size):
        stop = min(start + block_size, len(database))
        masks = np.empty((len(items), width, stop - start), dtype=bool)
        union = np.zeros(stop - start, dtype=bool)
        for item_id, (_, _, _, specs, rows) in enumerate(items):
            for offset, query_id in enumerate(range(begin, end)):
                mask = candidate_mask(specs, rows, query_id, start, stop)
                masks[item_id, offset] = mask
                union |= mask
        if not np.any(union):
            continue
        local_ids = np.flatnonzero(union)
        selected = np.ascontiguousarray(database.block(start, stop)[local_ids])
        scores = selected @ queries.T
        if metric == "l2":
            scores = 2 * scores - np.einsum("ij,ij->i", selected, selected)[:, None]
        for item_id in range(len(items)):
            for offset in range(width):
                valid = masks[item_id, offset, local_ids]
                ids = local_ids[valid] + start
                if len(ids) == 0:
                    continue
                values = scores[valid, offset]
                take = min(k, len(ids))
                ix = np.argpartition(values, -take)[-take:]
                combined_scores = np.concatenate((best_scores[item_id, offset], values[ix]))
                combined_ids = np.concatenate((best_ids[item_id, offset], ids[ix]))
                keep = np.argpartition(combined_scores, -k)[-k:]
                best_scores[item_id, offset] = combined_scores[keep]
                best_ids[item_id, offset] = combined_ids[keep]
    order = np.argsort(-best_scores, axis=2)
    result = np.take_along_axis(best_ids, order, axis=2)
    if np.any(result < 0):
        raise RuntimeError(f"Fewer than {k} matches in query range [{begin}, {end})")
    return result


def output_path(directory, worker):
    return directory / f"ground_truth.worker_{worker}.part"


def run_worker(args, database, queries, items, k, metric):
    count = len(queries)
    lo = count * args.worker // args.workers
    hi = count * (args.worker + 1) // args.workers
    paths = [output_path(item[2], args.worker) for item in items]
    lengths = [path.stat().st_size if path.exists() else 0 for path in paths]
    # The process may have stopped between flushing two workload files. Keep
    # only rows present in *every* part (all of these are our own checkpoints).
    done = min(lengths) // (4 * k)
    if len(set(lengths)) != 1 or lengths[0] != done * 4 * k:
        print(f"Recovering partial checkpoint to {done} complete rows", flush=True)
        for path in paths:
            if path.exists():
                with path.open("r+b") as handle:
                    handle.truncate(done * 4 * k)
    if done > hi - lo:
        raise RuntimeError("Partial output longer than worker's range")
    print(f"{args.dataset} worker {args.worker}: queries [{lo + done}, {hi}), "
          f"{len(items)} workloads, block={args.database_block}, batch={args.query_batch}", flush=True)
    handles = [path.open("ab") for path in paths]
    try:
        for begin in range(lo + done, hi, args.query_batch):
            end = min(begin + args.query_batch, hi)
            result = exact_batch(database, queries, items, k, metric,
                                 begin, end, args.database_block)
            for item_id, handle in enumerate(handles):
                handle.write(result[item_id].astype("<i4", copy=False).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            print(f"{args.dataset} worker {args.worker}: {end}/{hi}", flush=True)
    finally:
        for handle in handles:
            handle.close()


def merge(args, queries, items, k):
    count = len(queries)
    for scenario, target, directory, _, _ in items:
        final = directory / "ground_truth.ivecs"
        vectors = directory / "query_vectors.fvecs"
        if final.exists() or vectors.exists():
            raise RuntimeError(f"Refusing to overwrite existing output in {directory}")
        parts = []
        for worker in range(args.workers):
            lo = count * worker // args.workers
            hi = count * (worker + 1) // args.workers
            path = output_path(directory, worker)
            if not path.is_file() or path.stat().st_size != (hi - lo) * k * 4:
                raise RuntimeError(f"Missing or incomplete output {path}")
            parts.append(np.fromfile(path, dtype="<i4").reshape(-1, k))
        merged = np.concatenate(parts)
        if merged.shape != (count, k) or np.any(merged < 0):
            raise RuntimeError(f"Invalid merged ground truth in {directory}")
        # Write using the established FANNS format, with atomic publication.
        vector_tmp = directory / "query_vectors.fvecs.tmp"
        gt_tmp = directory / "ground_truth.ivecs.tmp"
        write_fvecs(vector_tmp, queries)
        write_ivecs(gt_tmp, merged)
        os.replace(vector_tmp, vectors)
        os.replace(gt_tmp, final)
        print(f"DONE {args.dataset}/{scenario}/{target}%: {count} queries x {k}", flush=True)


def validate(database, queries, items, k):
    count, dim = queries.shape
    for scenario, target, directory, specs, rows in items:
        raw = np.fromfile(directory / "query_vectors.fvecs", dtype="<i4")
        if raw.shape != (count * (dim + 1),):
            raise RuntimeError(f"Invalid query vector length in {directory}")
        raw = raw.reshape(count, dim + 1)
        if not np.all(raw[:, 0] == dim) or not np.array_equal(
                raw[:, 1:].view("<f4"), queries):
            raise RuntimeError(f"Incorrect query vector mapping in {directory}")
        raw_gt = np.fromfile(directory / "ground_truth.ivecs", dtype="<i4")
        if raw_gt.shape != (count * (k + 1),):
            raise RuntimeError(f"Invalid ground truth length in {directory}")
        raw_gt = raw_gt.reshape(count, k + 1)
        if not np.all(raw_gt[:, 0] == k):
            raise RuntimeError(f"Invalid ground truth width in {directory}")
        for query_id, ids in enumerate(raw_gt[:, 1:]):
            if np.any(ids < 0) or np.any(ids >= len(database)) or len(np.unique(ids)) != k:
                raise RuntimeError(f"Invalid or duplicate ID in {directory}: {query_id}")
            for spec in specs:
                attr = spec["database"][ids]
                values = rows[spec["query"]][query_id]
                if spec["kind"] == "enum":
                    match = np.isin(attr, values) if attr.ndim == 1 else attr[:, values].any(axis=1)
                else:
                    match = (attr >= values[0]) & (attr <= values[1])
                if not np.all(match):
                    raise RuntimeError(f"Result violates filter in {directory}: {query_id}")
        print(f"VALIDATED {scenario}/{target}%: {count} vectors, {count} GT rows, "
              f"all {k} IDs distinct and filter-matching", flush=True)


def verify(database, queries, items, k, metric):
    from generate_controlled_ground_truth import exact_topk
    class PrefixDatabase:
        def __len__(self):
            return min(300000, len(database))
        def block(self, begin, end):
            return database.block(begin, end)
    prefix = PrefixDatabase()
    for scenario, target, directory, specs, rows in items:
        if target not in (1, 60):
            continue
        short_specs = [{**spec, "database": spec["database"][:len(prefix)]}
                       for spec in specs]
        indices = []
        for query_id in range(min(len(queries), 40)):
            if candidate_mask(short_specs, rows, query_id, 0, len(prefix)).sum() >= k:
                indices.append(query_id)
                if len(indices) == 2:
                    break
        if len(indices) != 2:
            raise RuntimeError(f"Not enough representative queries in {directory}")
        test_queries = np.asarray(queries[indices], dtype=np.float32)
        test_rows = {key: [value[i] for i in indices] for key, value in rows.items()}
        reference = exact_topk(prefix, test_queries, short_specs,
                               test_rows,
                               k, metric, 2, 4096)
        proposed = exact_batch(prefix, test_queries, [(scenario, target, directory,
                                                        short_specs, test_rows)], k, metric,
                               0, 2, 4096)[0]
        if not np.array_equal(reference, proposed):
            diff = np.argwhere(reference != proposed)
            print(f"DIAGNOSTIC {scenario}/{target}%: first differences "
                  f"{[(tuple(pos), int(reference[tuple(pos)]), int(proposed[tuple(pos)])) for pos in diff[:6]]}",
                  flush=True)
            if any(set(reference[q, :10]) != set(proposed[q, :10])
                   for q in range(len(reference))):
                raise RuntimeError(f"Recall@10 differs: {scenario}/{target}%")
            if any(set(reference[q]) != set(proposed[q]) for q in range(len(reference))):
                raise RuntimeError(f"Top-k candidate set differs: {scenario}/{target}%")
        print(f"VERIFIED {scenario}/{target}%", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("arxiv", "qdrant"))
    parser.add_argument("--mode", choices=("verify", "work", "merge", "validate"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--query-batch", type=int, default=8)
    parser.add_argument("--database-block", type=int, default=4096)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers:
        parser.error("invalid worker count or index")
    database, queries, root, configs, k, metric = dataset_config(args.dataset)
    items = workloads(root, configs, len(queries))
    if args.mode == "verify":
        verify(database, queries, items, k, metric)
    elif args.mode == "work":
        run_worker(args, database, queries, items, k, metric)
    elif args.mode == "merge":
        merge(args, queries, items, k)
    else:
        validate(database, queries, items, k)


if __name__ == "__main__":
    main()
