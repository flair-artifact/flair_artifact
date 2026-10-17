from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import flair
from flair.experimental import BuildWithTagExtension, SearchWithTagExtension

from .io import read_attribute_rows, read_fvecs, read_ivecs
from .validation import (check_ids, config_digest, native_identity, package_root, positive_int, query_plan)


def _expanded_path(value: str, base: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    return path if path.is_absolute() else base / path


def _schema(attributes: list[dict[str, Any]]) -> str:
    return json.dumps([
        {"key_name": a["name"], "type": a["type"], "value_type": a.get("value_type", "int32")}
        for a in attributes
    ], separators=(",", ":"))


def _build_filter(attributes: list[dict[str, Any]], rows: list[list[list[int | float]]], qid: int) -> dict[str, Any]:
    predicates: list[dict[str, Any]] = []
    for attribute, column in zip(attributes, rows):
        values = column[qid]
        name = attribute["name"]
        value_type = attribute.get("value_type", "int32")
        cast = np.float32 if value_type == "float" else np.float64 if value_type == "double" else np.int32 if value_type == "int32" else np.int64
        if attribute["type"] == "enum":
            typed = [cast(value) for value in values]
            predicates.append({name: {"$eq": typed[0]}} if len(typed) == 1 else {name: {"$in": typed}})
        else:
            if len(values) != 2:
                raise ValueError(f"range query {name} must contain low,high")
            predicates.extend(({name: {"$gte": cast(values[0])}}, {name: {"$lte": cast(values[1])}}))
    return {"$and": predicates}


def _recall(found: np.ndarray, truth: np.ndarray, topk: int) -> float:
    expected = {int(value) for value in truth[:topk] if int(value) >= 0}
    actual = {int(value) for value in found[:topk] if int(value) >= 0}
    return len(expected.intersection(actual)) / topk


def _execute(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    base = config_path.parent
    dataset = config["dataset"]
    vectors = read_fvecs(_expanded_path(dataset["base_vectors"], base))
    queries = read_fvecs(_expanded_path(dataset["query_vectors"], base))
    groundtruth = read_ivecs(_expanded_path(dataset["groundtruth"], base))
    attributes = dataset["attributes"]
    if not attributes or len({a["name"] for a in attributes}) != len(attributes):
        raise ValueError("attribute names must be nonempty and unique")
    if vectors.shape[1] != queries.shape[1] or not len(queries):
        raise ValueError("vector dimensions differ or query set is empty")
    for matrix in (vectors, queries):
        for start in range(0, len(matrix), 4096):
            if not np.isfinite(matrix[start:start + 4096]).all():
                raise ValueError("vector inputs must be finite")
    for attribute in attributes:
        if attribute["type"] not in {"enum", "range"}:
            raise ValueError("unsupported attribute type")
        if attribute.get("value_type", "int32") not in {"int32", "int64", "float", "double"}:
            raise ValueError("unsupported attribute value type")
    database_columns = [
        read_attribute_rows(_expanded_path(a["database"], base), a.get("value_type", "int32"))
        for a in attributes
    ]
    query_columns = [
        read_attribute_rows(_expanded_path(a["queries"], base), a.get("value_type", "int32"))
        for a in attributes
    ]
    if any(len(column) != len(vectors) for column in database_columns):
        raise ValueError("database vector and attribute row counts differ")
    if any(len(column) != len(queries) for column in query_columns) or len(groundtruth) != len(queries):
        raise ValueError("query vector, attribute, and ground-truth row counts differ")

    topk = positive_int(config["query"].get("topk", 10), "topk")
    plan = query_plan(config)
    if groundtruth.shape[1] < topk:
        raise ValueError("ground truth width is smaller than topk")
    # Match the scalar representation passed to the native implementation.
    for attribute, db, qs in zip(attributes, database_columns, query_columns):
        value_type = attribute.get("value_type", "int32")
        cast = {"float": np.float32, "double": np.float64, "int32": np.int32, "int64": np.int64}[value_type]
        for column, is_query in ((db, False), (qs, True)):
            for row in column:
                if attribute["type"] == "range" and len(row) != (2 if is_query else 1):
                    raise ValueError("range rows require one database value or two query endpoints")
                if value_type in {"int32", "int64"}:
                    bounds = np.iinfo(cast)
                    if any(not bounds.min <= value <= bounds.max for value in row):
                        raise ValueError("attribute integer is outside its declared type")
                row[:] = [cast(value).item() for value in row]
                if not all(np.isfinite(value) for value in row):
                    raise ValueError("attribute values must be finite")
                if attribute["type"] == "range" and is_query and row[0] > row[1]:
                    raise ValueError("range query endpoints are reversed")
    for qid, truth in enumerate(groundtruth):
        check_ids(truth, count=len(vectors), topk=topk, attributes=attributes,
                  database=database_columns, queries=query_columns, qid=qid, truth=True)

    variant = os.environ.get("FLAIR_VARIANT", "main")
    expected_package = package_root(variant) / "flair"
    if Path(flair.__file__).resolve().parent != expected_package.resolve():
        raise ValueError("loaded flair package differs from selected variant; use run_artifact.sh")
    provenance = native_identity(variant)
    print(f"PACKAGE variant={variant} native_sha256={provenance['native_sha256']}")
    backend = config["index"]["backend"]
    if backend not in {"flatnsw", "diskann"}:
        raise ValueError("unsupported backend")
    schema = _schema(attributes)
    build_options = dict(config["index"].get("build_options", {}))
    build_options["extension.build.tag.tag_schema"] = schema
    index_path = _expanded_path(config["index"].get("path", "output/index"), base)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    builder = flair.Builder(flair.BuilderConfig(
        dimension=vectors.shape[1], index_type=backend, distance_metric="l2",
        extra_options=build_options,
    ))
    if backend == "diskann":
        sample_count = min(len(vectors), int(config["index"].get("pretrain_vectors", 100000)))
        builder.pretrain(np.ascontiguousarray(vectors[:sample_count], dtype=np.float32))
    tagged_builder = BuildWithTagExtension().attach(builder)
    tag_data = []
    for attribute, column in zip(attributes, database_columns):
        values = [row[0] if len(row) == 1 else row for row in column]
        tag_data.append({"tagk_name": attribute["name"], "data": values, "data_type": attribute.get("value_type", "int32")})
    tagged_builder.insert_batch_with_tag(
        np.ascontiguousarray(vectors, dtype=np.float32),
        np.arange(len(vectors), dtype=np.uint64), tag_data,
    )
    builder.dump(str(index_path))
    build_seconds = time.perf_counter() - started
    del tagged_builder, builder

    searcher_options = dict(config["index"].get("searcher_options", {}))
    searcher_options["extension.build.tag.tag_schema"] = schema
    searcher = flair.Searcher(flair.SearcherConfig(
        dimension=vectors.shape[1], index_type=backend,
        extra_options=searcher_options,
    ))
    searcher.open(str(index_path))
    try:
        tagged_searcher = SearchWithTagExtension().attach(searcher)
        curve = []
        for parallel, width, options in plan:
            recalls = []
            elapsed = 0.0
            returned_count = 0
            for qid, query in enumerate(queries):
                started = time.perf_counter()
                result = tagged_searcher.search(
                    np.ascontiguousarray(query, dtype=np.float32),
                    _build_filter(attributes, query_columns, qid),
                    flair.SearchParams(topk=topk, extra_options=options),
                )
                elapsed += time.perf_counter() - started
                returned_count += check_ids(
                    result.ids, count=len(vectors), topk=topk, attributes=attributes,
                    database=database_columns, queries=query_columns, qid=qid,
                )
                if len(result.distances) != len(result.ids) or not np.isfinite(result.distances).all():
                    raise ValueError(f"query {qid}: invalid distances")
                recalls.append(_recall(result.ids, groundtruth[qid], topk))
            row = {"search_width": int(width), "recall": float(np.mean(recalls)), "qps": len(queries) / elapsed, "query_seconds": elapsed, "checked_queries": len(queries), "returned_ids": returned_count}
            if backend == "diskann":
                row["parallel_number"] = parallel
                row["effective_options"] = options
            curve.append(row)
    finally:
        searcher.close()
    result = {
        "format_version": 2,
        "config_sha256": config_digest(config),
        "provenance": provenance,
        "topk": topk,
        "query_count": len(queries),
        "database_count": len(vectors),
        "validation": {"ids_and_predicates_checked": True, "groundtruth_ids_checked": True},
        "dataset": config.get("name", config_path.stem),
        "backend": backend,
        "measurement_driver": "python-single-query",
        "status": "complete",
        "build": {"wall_seconds": build_seconds, "index_bytes": index_path.stat().st_size},
        "query": curve,
    }
    if backend == "diskann":
        result["cache_policy"] = "operating-system-managed; not cleared"
        result["searcher_options"] = dict(config["index"].get("searcher_options", {}))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def execute(config_path: Path, output_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    inputs = [config_path]
    dataset = config["dataset"]
    inputs.extend(_expanded_path(dataset[k], config_path.parent)
                  for k in ("base_vectors", "query_vectors", "groundtruth"))
    for attribute in dataset["attributes"]:
        inputs.extend(_expanded_path(attribute[k], config_path.parent) for k in ("database", "queries"))
    if output_path.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("output must not overwrite an input file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"status": "running"}) + "\n")
    try:
        return _execute(config_path, output_path)
    except Exception as error:
        output_path.write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a FLAIR composite-filter experiment")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/reproduced/results.json"))
    args = parser.parse_args()
    result = execute(args.config, args.output)
    print(f"PASS output={args.output} curve_points={len(result['query'])}")


if __name__ == "__main__":
    main()
