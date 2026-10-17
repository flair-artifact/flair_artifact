from __future__ import annotations

import argparse
import importlib
import tempfile
from pathlib import Path

import numpy as np


def run_backend(package: object, backend: str) -> list[int]:
    experimental = importlib.import_module(f"{package.__name__}.experimental")
    rng = np.random.default_rng(42)
    count, dimension = 500, 16
    vectors = rng.normal(size=(count, dimension)).astype(np.float32)
    ids = np.arange(count, dtype=np.uint64)
    schema = '[{"key_name":"category","type":"enum","value_type":"int32"},{"key_name":"score","type":"range","value_type":"float"}]'
    options: dict[str, object] = {"extension.build.tag.tag_schema": schema}
    if backend == "flatnsw":
        options.update({"flatnsw.build.thread_count": 2, "flatnsw.build.ef_construction": 64, "flatnsw.build.neighbor_count": 32, "extension.build.tag.min_range_label_ratio": 0.02, "extension.build.tag.max_range_label_ratio": 0.5})
    else:
        options.update({"diskann.build.thread_count": 2, "diskann.build.ef_construction": 64, "diskann.build.neighbor_count": 32, "encoding.pq.m": 8, "encoding.pq.thread_count": 2})
    index_path = Path(tempfile.mkdtemp(prefix="flair-artifact-smoke-")) / f"{backend}.idx"
    builder = package.Builder(package.BuilderConfig(dimension=dimension, index_type=backend, distance_metric="l2", extra_options=options))
    if backend == "diskann":
        builder.pretrain(vectors)
    extension = experimental.BuildWithTagExtension().attach(builder)
    extension.insert_batch_with_tag(vectors, ids, [
        {"tagk_name": "category", "data": [int(i % 4) for i in range(count)], "data_type": "int32"},
        {"tagk_name": "score", "data": [float(i) for i in range(count)], "data_type": "float"},
    ])
    builder.dump(str(index_path))
    searcher = package.Searcher(package.SearcherConfig(dimension=dimension, index_type=backend, extra_options={"extension.build.tag.tag_schema": schema}))
    searcher.open(str(index_path))
    tagged = experimental.SearchWithTagExtension().attach(searcher)
    result = tagged.search(vectors[60], {"$and": [{"category": {"$eq": np.int32(0)}}, {"score": {"$gte": np.float32(40)}}, {"score": {"$lte": np.float32(100)}}]}, package.SearchParams(topk=5, extra_options={"flatnsw.search.ef_search": 100} if backend == "flatnsw" else {"diskann.search.list_size": 64, "diskann.search.io_limit": 64, "search.parallel_number": 1}))
    returned = [int(value) for value in result.ids]
    if len(returned) != 5 or len(set(returned)) != 5 or not all(value % 4 == 0 and 40 <= value <= 100 for value in returned):
        searcher.close()
        raise ValueError(f"smoke returned invalid filtered IDs: {returned}")
    searcher.close()
    return returned


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-contained FLAIR smoke test")
    parser.add_argument("--backend", choices=("flatnsw", "diskann", "all"), default="all")
    args = parser.parse_args()
    package = importlib.import_module("flair")
    backends = ("flatnsw", "diskann") if args.backend == "all" else (args.backend,)
    for backend in backends:
        print(f"PASS backend={backend} ids={run_backend(package, backend)}")


if __name__ == "__main__":
    main()
