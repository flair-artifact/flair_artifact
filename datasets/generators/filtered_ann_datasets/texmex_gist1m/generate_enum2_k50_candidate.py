#!/usr/bin/env python3
"""Generate a 50-label candidate GIST1M EMIS workload without ground truth."""

import argparse
from pathlib import Path

import numpy as np


N = 1_000_000
NQ = 1_000
ZIPF_S = 1.2
MAX_M = 10
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument("--num-labels", type=int, default=50)
args = parser.parse_args()
K_LABELS = args.num_labels
OUT = Path(__file__).resolve().parent / f"enum2_k{K_LABELS}_candidate"

rng = np.random.RandomState(SEED)
ranks = np.arange(1, K_LABELS + 1)
label_probs = 1.0 / ranks**ZIPF_S
label_probs /= label_probs.sum()

ms = np.arange(1, MAX_M + 1)
count_probs = 0.5 ** (ms - 1)
count_probs /= count_probs.sum()
counts = rng.choice(ms, size=N, p=count_probs)

labels = np.full((N, MAX_M), -1, dtype=np.int32)
for m in range(1, MAX_M + 1):
    for row_id in np.flatnonzero(counts == m):
        labels[row_id, :m] = rng.choice(
            K_LABELS, size=m, replace=False, p=label_probs
        )

sizes = np.array([(labels == label).any(axis=1).sum() for label in range(K_LABELS)])
present = np.flatnonzero(sizes)
queries = rng.choice(present, size=NQ).astype(np.int32)

OUT.mkdir(exist_ok=True)
np.save(OUT / "gist_enum2_attributes.npy", labels)
np.save(OUT / "gist_enum2_queries.npy", queries)
with (OUT / "enum2_database_attributes.csv").open("w") as output:
    for row, count in zip(labels, counts):
        output.write(",".join(map(str, row[:count])) + "\n")
with (OUT / "enum2_query_attributes.csv").open("w") as output:
    output.write("\n".join(map(str, queries)) + "\n")

query_selectivities = sizes[queries] / N
print(f"mean labels/vector: {counts.mean():.6f}")
print(f"multi-label vectors: {(counts > 1).mean():.6%}")
print("label selectivity percentiles (%):", np.percentile(sizes / N * 100, [0, 10, 25, 50, 75, 90, 100]))
print("query selectivity percentiles (%):", np.percentile(query_selectivities * 100, [0, 10, 25, 50, 75, 90, 100]))
print(f"query mean selectivity: {query_selectivities.mean():.6%}")
print(f"queries below 1%: {(query_selectivities < .01).mean():.2%}")
print(f"queries below 0.5%: {(query_selectivities < .005).mean():.2%}")
print(f"unique queried labels: {np.unique(queries).size}/{K_LABELS}")
print(f"output: {OUT}")
