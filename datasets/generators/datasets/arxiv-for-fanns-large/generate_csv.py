"""
Generate CSV attribute files for arxiv-for-fanns-large dataset.
Applies the same logic as FANNS benchmark dataset_loader.py to the local raw/ files,
plus adds number_of_authors as a second range attribute (r2).

Input:  raw/database_attributes.jsonl, raw/*_query_attributes.jsonl
Output: em_database_attributes.csv, em_query_attributes.csv
        emis_database_attributes.csv, emis_query_attributes.csv
        r_database_attributes.csv, r_query_attributes.csv
        r2_database_attributes.csv, r2_query_attributes.csv
"""
import json
import os
import sys
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

attribute_by_filter = {
    "EM": "number_of_sub_categories",
    "R": "update_date",
    "EMIS": "main_categories",
    "R2": "number_of_authors",
}

# ============================================================
# Load database attributes
# ============================================================
print("Loading database attributes...")
database_attributes = []
with open(os.path.join(RAW_DIR, "database_attributes.jsonl")) as f:
    for line in f:
        database_attributes.append(json.loads(line))
n_items = len(database_attributes)
print(f"  Loaded {n_items} records")

# ============================================================
# EM: number_of_sub_categories -> integer encoding
# ============================================================
print("Processing EM (number_of_sub_categories)...")
em_encoding_map = {}
em_db_path = os.path.join(OUT_DIR, "em_database_attributes.csv")
with open(em_db_path, "w") as f:
    for row in database_attributes:
        val = row[attribute_by_filter["EM"]]
        if val not in em_encoding_map:
            em_encoding_map[val] = len(em_encoding_map)
        f.write(str(em_encoding_map[val]) + "\n")
print(f"  EM encoding: {len(em_encoding_map)} unique values")

# EM query attributes
em_query_path = os.path.join(OUT_DIR, "em_query_attributes.csv")
with open(os.path.join(RAW_DIR, "em_query_attributes.jsonl")) as fin, \
     open(em_query_path, "w") as fout:
    for line in fin:
        row = json.loads(line)
        val = row["label"]
        if val not in em_encoding_map:
            print(f"WARNING: EM query value {val} not in database")
            em_encoding_map[val] = len(em_encoding_map)
        fout.write(str(em_encoding_map[val]) + "\n")
print(f"  EM query attributes written")

# ============================================================
# EMIS: main_categories -> integer encoding (multi-value)
# ============================================================
print("Processing EMIS (main_categories)...")
emis_encoding_map = {}
emis_db_path = os.path.join(OUT_DIR, "emis_database_attributes.csv")
with open(emis_db_path, "w") as f:
    for row in database_attributes:
        values = row[attribute_by_filter["EMIS"]]
        values_encoded = []
        for val in values:
            if val not in emis_encoding_map:
                emis_encoding_map[val] = len(emis_encoding_map)
            values_encoded.append(str(emis_encoding_map[val]))
        f.write(",".join(values_encoded) + "\n")
print(f"  EMIS encoding: {len(emis_encoding_map)} unique categories")

# EMIS query attributes
emis_query_path = os.path.join(OUT_DIR, "emis_query_attributes.csv")
with open(os.path.join(RAW_DIR, "emis_query_attributes.jsonl")) as fin, \
     open(emis_query_path, "w") as fout:
    for line in fin:
        row = json.loads(line)
        val = row["label"]
        if val not in emis_encoding_map:
            print(f"WARNING: EMIS query value {val} not in database")
            emis_encoding_map[val] = len(emis_encoding_map)
        fout.write(str(emis_encoding_map[val]) + "\n")
print(f"  EMIS query attributes written")

# ============================================================
# R (Range 1): update_date -> sort-and-remap to 0~N-1
# ============================================================
print("Processing R (update_date)...")
raw_values = [row[attribute_by_filter["R"]] for row in database_attributes]
old_idx_val_pairs = list(enumerate(raw_values))
old_idx_val_pairs_sorted = sorted(old_idx_val_pairs, key=lambda x: x[1])
old_idx_val_new_idx = [(old_idx, val, new_idx) for new_idx, (old_idx, val) in enumerate(old_idx_val_pairs_sorted)]
new_idx_by_old = [0] * n_items
for old_idx, val, new_idx in old_idx_val_new_idx:
    new_idx_by_old[old_idx] = new_idx

r_db_path = os.path.join(OUT_DIR, "r_database_attributes.csv")
with open(r_db_path, "w") as f:
    for idx in new_idx_by_old:
        f.write(str(idx) + "\n")

val_min = old_idx_val_pairs_sorted[0][1]
val_max = old_idx_val_pairs_sorted[-1][1]
print(f"  R1 raw range: [{val_min}, {val_max}], remapped to [0, {n_items-1}]")

# R query attributes: remap range_start/range_end
r_query_path = os.path.join(OUT_DIR, "r_query_attributes.csv")
with open(os.path.join(RAW_DIR, "r_query_attributes.jsonl")) as fin, \
     open(r_query_path, "w") as fout:
    for line in fin:
        row = json.loads(line)
        range_start = row["range_start"]
        range_end = row["range_end"]
        # Find smallest new index with value >= range_start
        lo = 0
        hi = n_items - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if old_idx_val_pairs_sorted[mid][1] < range_start:
                lo = mid + 1
            else:
                hi = mid
        new_start = lo
        # Find largest new index with value <= range_end
        lo = 0
        hi = n_items - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if old_idx_val_pairs_sorted[mid][1] > range_end:
                hi = mid - 1
            else:
                lo = mid
        new_end = lo
        fout.write(f"{new_start}-{new_end}\n")
print(f"  R1 query attributes written")

# ============================================================
# R2 (Range 2): number_of_authors -> sort-and-remap to 0~N-1
# ============================================================
print("Processing R2 (number_of_authors)...")
raw_values_r2 = [row[attribute_by_filter["R2"]] for row in database_attributes]
old_idx_val_pairs_r2 = list(enumerate(raw_values_r2))
old_idx_val_pairs_sorted_r2 = sorted(old_idx_val_pairs_r2, key=lambda x: x[1])
old_idx_val_new_idx_r2 = [(old_idx, val, new_idx) for new_idx, (old_idx, val) in enumerate(old_idx_val_pairs_sorted_r2)]
new_idx_by_old_r2 = [0] * n_items
for old_idx, val, new_idx in old_idx_val_new_idx_r2:
    new_idx_by_old_r2[old_idx] = new_idx

r2_db_path = os.path.join(OUT_DIR, "r2_database_attributes.csv")
with open(r2_db_path, "w") as f:
    for idx in new_idx_by_old_r2:
        f.write(str(idx) + "\n")

val_min_r2 = old_idx_val_pairs_sorted_r2[0][1]
val_max_r2 = old_idx_val_pairs_sorted_r2[-1][1]
print(f"  R2 raw range: [{val_min_r2}, {val_max_r2}], remapped to [0, {n_items-1}]")

# R2 query attributes: generate range queries with multiple selectivities
# We generate queries similar to the existing R queries' selectivity distribution
# Sample from a range of selectivities (1%, 5%, 10%, 25%, 50%)
print("  Generating R2 range queries...")
np.random.seed(42)
n_queries = 10000

selectivities = [0.01, 0.05, 0.10, 0.25, 0.50]
queries_per_sel = n_queries // len(selectivities)

r2_queries = []
for sel in selectivities:
    width = max(1, int(n_items * sel))
    for _ in range(queries_per_sel):
        start = np.random.randint(0, n_items - width + 1)
        end = start + width - 1
        r2_queries.append((start, end))

r2_query_path = os.path.join(OUT_DIR, "r2_query_attributes.csv")
with open(r2_query_path, "w") as f:
    for start, end in r2_queries:
        f.write(f"{start}-{end}\n")

# Also save the original (un-remapped) number_of_authors as npy for reference
authors_arr = np.array(raw_values_r2, dtype=np.int32)
np.save(os.path.join(OUT_DIR, "number_of_authors_raw.npy"), authors_arr)

print(f"  R2: {n_queries} queries across selectivities {selectivities}")
print(f"  Saved number_of_authors_raw.npy for reference")

# ============================================================
# Verify
# ============================================================
print("\n=== Verification ===")
for name, path in [
    ("em_database_attributes.csv", em_db_path),
    ("em_query_attributes.csv", em_query_path),
    ("emis_database_attributes.csv", emis_db_path),
    ("emis_query_attributes.csv", emis_query_path),
    ("r_database_attributes.csv", r_db_path),
    ("r_query_attributes.csv", r_query_path),
    ("r2_database_attributes.csv", r2_db_path),
    ("r2_query_attributes.csv", r2_query_path),
]:
    with open(path) as f:
        lines = sum(1 for _ in f)
    print(f"  {name}: {lines} lines")

print("\nDone!")
