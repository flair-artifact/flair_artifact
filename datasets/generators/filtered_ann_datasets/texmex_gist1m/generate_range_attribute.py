#!/usr/bin/env python3
"""Regenerate the synthetic sorted integer range attribute used as GIST1M range-1.

Definition: 1,000,000 integers drawn uniformly from [0, 9999992] using the legacy
NumPy ``RandomState(123)`` stream, cast to int32 and sorted ascending.  Sorting is
a monotone transform, so predicate semantics are identical to the unsorted draw.
The composite workloads use this array as the integer range dimension (range-1);
``generate_attributes.py`` reads it to derive the integer range query files.

Output: ``gist_attributes.npy`` next to this script.
"""
import os

import numpy as np

SEED = 123
N = 1_000_000
HIGH = 9_999_993  # exclusive upper bound


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gist_attributes.npy")
    values = np.random.RandomState(SEED).randint(0, HIGH, N).astype(np.int32)
    values.sort()
    np.save(out, values)
    print(f"wrote {out}: n={N} dtype=int32 min={values[0]} max={values[-1]} "
          f"unique={np.unique(values).size}")


if __name__ == "__main__":
    main()
