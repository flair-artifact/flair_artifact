"""Materialize query vectors and exact GT for controlled composite workloads."""
import argparse
import csv
import json
import struct
from pathlib import Path

import h5py
import numpy as np


TARGETS = (1, 5, 10, 20, 40, 60)


def read_rows(path):
    with open(path, newline="") as handle:
        return [[int(value) for value in row] for row in csv.reader(handle)]


def write_fvecs(path, vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    dims = np.full((len(vectors), 1), vectors.shape[1], dtype=np.int32)
    with open(path, "wb") as handle:
        for dim, row in zip(dims, vectors):
            handle.write(dim.tobytes())
            handle.write(row.tobytes())


def write_ivecs(path, values):
    values = np.asarray(values, dtype=np.int32)
    with open(path, "wb") as handle:
        for row in values:
            handle.write(struct.pack("<i", len(row)))
            handle.write(row.tobytes())


def load_multivalue_csv(path, value_count=None):
    rows = read_rows(path)
    if value_count is None:
        value_count = max(max(row) for row in rows) + 1
    matrix = np.zeros((len(rows), value_count), dtype=bool)
    for index, row in enumerate(rows):
        matrix[index, row] = True
    return matrix


class NpyDatabase:
    def __init__(self, path):
        self.data = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.data)

    def block(self, begin, end):
        return np.asarray(self.data[begin:end], dtype=np.float32)


class Hdf5Database:
    def __init__(self, path, key):
        self.handle = h5py.File(path, "r")
        self.data = self.handle[key]

    def __len__(self):
        return len(self.data)

    def block(self, begin, end):
        return np.asarray(self.data[begin:end], dtype=np.float32)


def enum_mask(database, values):
    if database.ndim == 1:
        return np.isin(database, values)
    return database[:, values].any(axis=1)


def make_masks(specs, query_rows, begin, end):
    masks = np.ones((end - begin, len(specs[0]["database"])), dtype=bool)
    for spec in specs:
        database = spec["database"]
        rows = query_rows[spec["query"]][begin:end]
        if spec["kind"] == "enum":
            for offset, values in enumerate(rows):
                masks[offset] &= enum_mask(database, values)
        else:
            for offset, (low, high) in enumerate(rows):
                masks[offset] &= (database >= low) & (database <= high)
    return masks


def exact_topk(database, queries, specs, query_rows, k, metric, query_batch, database_block):
    count = len(queries)
    result = np.full((count, k), -1, dtype=np.int32)
    for qbegin in range(0, count, query_batch):
        qend = min(count, qbegin + query_batch)
        query = np.asarray(queries[qbegin:qend], dtype=np.float32)
        masks = make_masks(specs, query_rows, qbegin, qend)
        width = qend - qbegin
        best_scores = np.full((width, k), -np.inf, dtype=np.float32)
        best_ids = np.full((width, k), -1, dtype=np.int32)
        for dbegin in range(0, len(database), database_block):
            dend = min(len(database), dbegin + database_block)
            vectors = database.block(dbegin, dend)
            scores = vectors @ query.T
            if metric == "l2":
                scores = 2.0 * scores - np.einsum("ij,ij->i", vectors, vectors)[:, None]
            valid = masks[:, dbegin:dend].T
            scores[~valid] = -np.inf
            take = min(k, len(vectors))
            local = np.argpartition(scores, -take, axis=0)[-take:]
            local_scores = np.take_along_axis(scores, local, axis=0).T
            local_ids = (local + dbegin).T.astype(np.int32)
            combined_scores = np.concatenate((best_scores, local_scores), axis=1)
            combined_ids = np.concatenate((best_ids, local_ids), axis=1)
            keep = np.argpartition(combined_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(combined_scores, keep, axis=1)
            best_ids = np.take_along_axis(combined_ids, keep, axis=1)
        order = np.argsort(-best_scores, axis=1)
        result[qbegin:qend] = np.take_along_axis(best_ids, order, axis=1)
        print(f"  queries {qend}/{count}", flush=True)
    return result


def load_query_rows(directory):
    return {path.name: read_rows(path) for path in directory.glob("query_*.csv")}


def materialize_workload(database, source_queries, specs, directory, k, metric, query_batch, database_block,
                         source_indices=None):
    query_rows = load_query_rows(directory)
    count = len(next(iter(query_rows.values())))
    if source_indices is None:
        source_indices = np.arange(count) % len(source_queries)
    queries = np.asarray(source_queries[source_indices], dtype=np.float32)
    write_fvecs(directory / "query_vectors.fvecs", queries)
    gt = exact_topk(database, queries, specs, query_rows, k, metric, query_batch, database_block)
    if np.any(gt < 0):
        raise RuntimeError(f"Not enough candidates for top-{k}: {directory}")
    write_ivecs(directory / "ground_truth.ivecs", gt)
    return gt


def dataset_config(name):
    if name == "gist":
        source = Path("filtered_ann_datasets/texmex_gist1m")
        root = source / "composite_controlled"
        with h5py.File(source / "gist-960-euclidean.hdf5", "r") as handle:
            queries = handle["test"][:]
        enum1 = np.load(source / "gist_enum1_attributes.npy", mmap_mode="r")
        raw_enum2 = np.load(source / "enum2_k20_candidate/gist_enum2_attributes.npy", mmap_mode="r")
        enum2 = np.zeros((len(raw_enum2), 20), dtype=bool)
        for value in range(20):
            enum2[:, value] = np.any(raw_enum2 == value, axis=1)
        r1 = np.load(source / "gist_attributes.npy", mmap_mode="r")
        r2 = np.asarray(read_rows(root / "database_range_float.csv"), dtype=np.uint32).reshape(-1)
        configs = {
            "enum_enum": [
                {"kind": "enum", "database": enum1, "query": "query_enum1.csv"},
                {"kind": "enum", "database": enum2, "query": "query_enum2.csv"}],
            "range_range": [
                {"kind": "range", "database": r1, "query": "query_range_int.csv"},
                {"kind": "range", "database": r2, "query": "query_range_float.csv"}],
            "enum_range": [
                {"kind": "enum", "database": enum2, "query": "query_enum2.csv"},
                {"kind": "range", "database": r1, "query": "query_range_int.csv"}],
        }
        return Hdf5Database(source / "gist-960-euclidean.hdf5", "train"), queries, root, configs, 10, "l2"
    if name == "arxiv":
        source = Path("datasets/arxiv-for-fanns-large")
        root = source / "composite_controlled"
        with h5py.File(source / "hdf5/query.hdf5", "r") as handle:
            queries = handle["vector"][:1000]
        em = np.loadtxt(source / "em_database_attributes.csv", dtype=np.int32)
        emis = load_multivalue_csv(source / "emis_database_attributes.csv", 38)
        r1 = np.loadtxt(source / "r_database_attributes.csv", dtype=np.int32)
        r2 = np.loadtxt(source / "r2_database_attributes.csv", dtype=np.int32)
        configs = {
            "enum_enum": [
                {"kind": "enum", "database": em, "query": "query_em.csv"},
                {"kind": "enum", "database": emis, "query": "query_emis.csv"}],
            "range_range": [
                {"kind": "range", "database": r1, "query": "query_r1.csv"},
                {"kind": "range", "database": r2, "query": "query_r2.csv"}],
            "enum_range": [
                {"kind": "enum", "database": emis, "query": "query_emis.csv"},
                {"kind": "range", "database": r1, "query": "query_r1.csv"}],
        }
        return NpyDatabase(source / "database_vectors.npy"), queries, root, configs, 100, "cosine"
    source = Path("filtered_ann_datasets/qdrant_arxiv")
    data = source / "fanns_format"
    root = data / "composite_controlled"
    queries = []
    with open(source / "tests.jsonl") as handle:
        for index, line in enumerate(handle):
            if index == 1000:
                break
            queries.append(json.loads(line)["query"])
    queries = np.asarray(queries, dtype=np.float32)
    emis = load_multivalue_csv(data / "emis_database_attributes.csv", 176)
    ranges = np.loadtxt(data / "r_database_attributes.csv", dtype=np.int32)
    configs = {".": [
        {"kind": "enum", "database": emis, "query": "query_emis.csv"},
        {"kind": "range", "database": ranges, "query": "query_range.csv"}]}
    return NpyDatabase(source / "vectors.npy"), queries, root, configs, 25, "cosine"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("gist", "arxiv", "qdrant"))
    parser.add_argument("--scope", choices=("mixed", "bins", "all"), default="mixed")
    parser.add_argument("--query-batch", type=int, default=128)
    parser.add_argument("--database-block", type=int, default=4096)
    args = parser.parse_args()
    database, queries, root, configs, k, metric = dataset_config(args.dataset)
    for scenario, specs in configs.items():
        base = root if scenario == "." else root / scenario
        bin_truth = {}
        if args.scope in ("bins", "all"):
            for target in TARGETS:
                directory = base / f"selectivity_{target}"
                print(f"{args.dataset}/{scenario}/{target}%", flush=True)
                bin_truth[target] = materialize_workload(
                    database, queries, specs, directory, k, metric, args.query_batch, args.database_block)
        if args.scope in ("mixed", "all"):
            directory = base / "mixed"
            source = np.asarray([row[0] for row in read_rows(directory / "query_source_indices.csv")])
            print(f"{args.dataset}/{scenario}/mixed", flush=True)
            materialize_workload(database, queries, specs, directory, k, metric,
                                 args.query_batch, args.database_block, source)


if __name__ == "__main__":
    main()
