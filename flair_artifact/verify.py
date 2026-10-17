from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .validation import VARIANTS, config_digest, finite, native_identity, positive_int, query_plan


def check(actual: Path, config_path: Path | None, variant: str | None = None) -> int:
    result = json.loads(actual.read_text())
    if result.get('format_version') != 2:
        raise ValueError('unsupported result format; rerun with the current driver')
    if result.get('status') != 'complete':
        raise ValueError(f"experiment status={result.get('status')!r}")
    variant = variant or os.environ.get('FLAIR_VARIANT', 'main')
    if result.get('provenance') != native_identity(variant):
        raise ValueError('result variant or binary hash differs from selected package')
    topk = positive_int(result['topk'], 'topk')
    nq = positive_int(result['query_count'], 'query_count')
    positive_int(result['database_count'], 'database_count')
    checks = result.get('validation', {})
    if checks.get('ids_and_predicates_checked') is not True or checks.get('groundtruth_ids_checked') is not True:
        raise ValueError('missing driver correctness checks')
    finite(result['build']['wall_seconds'], 'build.wall_seconds', positive=True)
    positive_int(result['build']['index_bytes'], 'build.index_bytes')
    rows = result.get('query')
    if not isinstance(rows, list) or not rows:
        raise ValueError('query curve is empty')
    widths = []
    series_widths = []
    for i, row in enumerate(rows):
        width = positive_int(row['search_width'], 'search_width')
        key = (row.get('parallel_number'), width)
        if key in series_widths:
            raise ValueError('duplicate series/search width')
        series_widths.append(key)
        widths.append(width)
        recall = finite(row['recall'], f'query[{i}].recall')
        if not 0 <= recall <= 1:
            raise ValueError('recall outside [0,1]')
        finite(row['qps'], f'query[{i}].qps', positive=True)
        finite(row['query_seconds'], f'query[{i}].query_seconds', positive=True)
        if type(row.get('checked_queries')) is not int or row['checked_queries'] != nq:
            raise ValueError('not all queries were checked')
        returned = row.get('returned_ids')
        if type(returned) is not int or not 0 <= returned <= nq * topk:
            raise ValueError('invalid returned ID count')
    if config_path is not None:
        config = json.loads(config_path.read_text())
        if result.get('config_sha256') != config_digest(config):
            raise ValueError('result configuration hash mismatch')
        if result.get('dataset') != config.get('name', config_path.stem):
            raise ValueError('result dataset/scenario mismatch')
        if result.get('backend') != config['index']['backend']:
            raise ValueError('result backend mismatch')
        plan = query_plan(config)
        if series_widths != [(p, w) for p, w, _ in plan] or topk != config['query'].get('topk', 10):
            raise ValueError('search widths, series or topk mismatch')
        if config['index']['backend'] == 'diskann':
            if result.get('searcher_options') != config['index'].get('searcher_options', {}):
                raise ValueError('searcher options mismatch')
            if result.get('cache_policy') != 'operating-system-managed; not cleared':
                raise ValueError('unexpected cache policy')
            for row, (_, _, options) in zip(rows, plan):
                if row.get('effective_options') != options:
                    raise ValueError('effective query options mismatch')
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Check correctness records and output integrity without paper metric targets')
    parser.add_argument('--actual', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default=None,
                        help='expected package; defaults to FLAIR_VARIANT or main')
    args = parser.parse_args()
    try:
        points = check(args.actual, args.config, args.variant)
    except (ValueError, TypeError, KeyError, OSError, OverflowError, AttributeError) as error:
        raise SystemExit(f'FAIL result validation: {error}') from error
    print(f'FUNCTIONAL VALIDATION PASSED curve_points={points} metric_values_recorded_but_not_compared')


if __name__ == '__main__':
    main()
