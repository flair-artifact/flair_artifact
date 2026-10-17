"""Functional checks and result provenance; no paper metric thresholds."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
from pathlib import Path
from typing import Any

VARIANTS = (
    'main', 'no_label_aware_pruning', 'no_query_time_label_aware_entry_points',
    'no_build_and_query_label_aware_entry_points',
)
ROOT = Path(__file__).resolve().parents[1]


def config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def package_root(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant: {variant}')
    if variant == 'main':
        return ROOT / 'prebuilt/linux-x86_64-cp311'
    return ROOT / 'prebuilt/ablations/linux-x86_64-cp311' / variant


def native_identity(variant: str) -> dict[str, str]:
    native = package_root(variant) / 'flair/_native.so'
    return {'variant': variant, 'native_sha256': file_digest(native)}


def matches(vector_id: int, attributes: list[dict[str, Any]], database: list,
            queries: list, qid: int) -> bool:
    for attribute, db, qs in zip(attributes, database, queries):
        values, predicate = db[vector_id], qs[qid]
        if attribute['type'] == 'enum':
            if not set(values).intersection(predicate):
                return False
        elif not predicate[0] <= values[0] <= predicate[1]:
            return False
    return True


def check_ids(ids: Any, *, count: int, topk: int, attributes: list[dict[str, Any]],
              database: list, queries: list, qid: int, truth: bool = False) -> int:
    # Returned arrays must be flat integer ID lists. Ground truth may use -1 padding.
    if getattr(ids, 'ndim', 1) != 1:
        raise ValueError(f'query {qid}: IDs must be one-dimensional')
    values = list(ids)
    if not truth and len(values) > topk:
        raise ValueError(f'query {qid}: more than topk IDs returned')
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f'query {qid}: ID is not an integer')
        value = int(value)
        if truth and value == -1:
            continue
        if not 0 <= value < count:
            raise ValueError(f'query {qid}: ID outside database range: {value}')
        if value in seen:
            raise ValueError(f'query {qid}: duplicate ID: {value}')
        seen.add(value)
        if not matches(value, attributes, database, queries, qid):
            raise ValueError(f'query {qid}: ID violates the filter: {value}')
    return len(seen)


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f'{name} must be finite' + (' and positive' if positive else ''))
    return number


def query_plan(config: dict[str, Any]) -> list[tuple[int | None, int, dict[str, Any]]]:
    """Resolve the options actually passed for every series and search width."""
    query = config['query']
    backend = config['index']['backend']
    widths = query['widths']
    if not widths or len(set(widths)) != len(widths):
        raise ValueError('search widths must be nonempty and unique')
    for width in widths:
        positive_int(width, 'search width')
    parallels = query.get('parallel_numbers', [None])
    if not isinstance(parallels, list) or not parallels or len(set(parallels)) != len(parallels):
        raise ValueError('parallel numbers must be nonempty and unique')
    if 'parallel_numbers' in query:
        if backend != 'diskann':
            raise ValueError('parallel_numbers requires diskann')
        for parallel in parallels:
            positive_int(parallel, 'parallel number')
    multiplier = query.get('io_limit_multiplier')
    if multiplier is not None:
        if backend != 'diskann':
            raise ValueError('io_limit_multiplier requires diskann')
        positive_int(multiplier, 'io limit multiplier')
    width_key = query.get('width_option', 'flatnsw.search.ef_search' if backend == 'flatnsw' else 'diskann.search.list_size')
    plan = []
    for parallel in parallels:
        for width in widths:
            options = dict(query.get('options', {}))
            options[width_key] = width
            if parallel is not None:
                options['search.parallel_number'] = parallel
            if multiplier is not None:
                options['diskann.search.io_limit'] = width * multiplier
            plan.append((parallel, width, options))
    return plan
