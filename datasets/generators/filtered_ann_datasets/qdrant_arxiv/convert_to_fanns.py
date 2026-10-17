"""
Convert Qdrant ArXiv dataset to FANNS benchmark format.

Attributes:
  - labels (EMIS): ArXiv categories, multi-value, 176 unique
  - submitter (EM): submitter name, single-value, 11954 unique
  - update_date_ts (Range): Unix timestamp, int32

Input:  vectors.npy, payloads.jsonl, tests.jsonl
Output: database_vectors.fvecs, query_vectors_{emis,em,r}.fvecs
        {emis,em,r}_database_attributes.csv
        {emis,em,r}_query_attributes.csv
        ground_truth_{emis,em,r}.ivecs
"""
import json
import os
import struct
import sys
import numpy as np

DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DIR, "fanns_format")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Helper functions
# ============================================================
def write_fvecs(path, data):
    """Write float32 vectors in fvecs format."""
    n, d = data.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(data[i].tobytes())

def write_ivecs(path, data):
    """Write int32 vectors in ivecs format. data is list of 1D arrays or 2D array."""
    if isinstance(data, np.ndarray) and data.ndim == 2:
        n, d = data.shape
        with open(path, 'wb') as f:
            for i in range(n):
                f.write(struct.pack('<i', d))
                f.write(data[i].astype(np.int32).tobytes())
    else:
        with open(path, 'wb') as f:
            for row in data:
                arr = np.array(row, dtype=np.int32)
                f.write(struct.pack('<i', len(arr)))
                f.write(arr.tobytes())

# ============================================================
# 1. Load vectors
# ============================================================
print("Loading vectors...")
vectors = np.load(os.path.join(DIR, "vectors.npy"))
n_db, dim = vectors.shape
print(f"  Database vectors: {n_db} x {dim}, dtype={vectors.dtype}")

# Convert to float32 if needed
if vectors.dtype != np.float32:
    vectors = vectors.astype(np.float32)

# Write database vectors
print("Writing database_vectors.fvecs...")
write_fvecs(os.path.join(OUT_DIR, "database_vectors.fvecs"), vectors)

# ============================================================
# 2. Load and encode database attributes
# ============================================================
print("Loading payloads...")
labels_list = []
submitters = []
timestamps = []

labels_encoding = {}
submitter_encoding = {}

with open(os.path.join(DIR, "payloads.jsonl")) as f:
    for line in f:
        d = json.loads(line)

        # labels: multi-value EMIS
        labs = d.get("labels", [])
        encoded = []
        for lab in labs:
            if lab not in labels_encoding:
                labels_encoding[lab] = len(labels_encoding)
            encoded.append(labels_encoding[lab])
        labels_list.append(encoded)

        # submitter: single-value EM
        sub = d.get("submitter", "")
        if sub not in submitter_encoding:
            submitter_encoding[sub] = len(submitter_encoding)
        submitters.append(submitter_encoding[sub])

        # update_date_ts: range (int)
        ts = d.get("update_date_ts", 0)
        timestamps.append(int(ts))

print(f"  labels: {len(labels_encoding)} unique categories")
print(f"  submitters: {len(submitter_encoding)} unique submitters")
ts_arr = np.array(timestamps, dtype=np.int64)
print(f"  timestamps: [{ts_arr.min()}, {ts_arr.max()}]")

# Write EMIS database attributes
print("Writing database attribute CSVs...")
with open(os.path.join(OUT_DIR, "emis_database_attributes.csv"), "w") as f:
    for encoded in labels_list:
        f.write(",".join(str(e) for e in encoded) + "\n")

# Write EM database attributes
with open(os.path.join(OUT_DIR, "em_database_attributes.csv"), "w") as f:
    for sub in submitters:
        f.write(str(sub) + "\n")

# Write R database attributes (sort-and-remap to 0~N-1)
print("Remapping timestamps to 0~N-1...")
old_idx_val = list(enumerate(timestamps))
old_idx_val_sorted = sorted(old_idx_val, key=lambda x: x[1])
new_idx_by_old = [0] * n_db
val_by_new_idx = [0] * n_db
for new_idx, (old_idx, val) in enumerate(old_idx_val_sorted):
    new_idx_by_old[old_idx] = new_idx
    val_by_new_idx[new_idx] = val

with open(os.path.join(OUT_DIR, "r_database_attributes.csv"), "w") as f:
    for idx in new_idx_by_old:
        f.write(str(idx) + "\n")

print(f"  Timestamps remapped to [0, {n_db-1}]")

# ============================================================
# 3. Process queries - split by filter type
# ============================================================
print("Processing queries...")
emis_queries = []  # (query_vec, label_encoded, closest_ids)
em_queries = []
r_queries = []

with open(os.path.join(DIR, "tests.jsonl")) as f:
    for line in f:
        d = json.loads(line)
        query_vec = np.array(d["query"], dtype=np.float32)
        closest_ids = d["closest_ids"]
        conds = d["conditions"]["and"]
        c = conds[0]

        if "labels" in c:
            val = c["labels"]["match"]["value"]
            if val not in labels_encoding:
                labels_encoding[val] = len(labels_encoding)
            emis_queries.append((query_vec, labels_encoding[val], closest_ids))
        elif "submitter" in c:
            val = c["submitter"]["match"]["value"]
            if val not in submitter_encoding:
                submitter_encoding[val] = len(submitter_encoding)
            em_queries.append((query_vec, submitter_encoding[val], closest_ids))
        elif "update_date_ts" in c:
            rng = c["update_date_ts"]["range"]
            gt_val = rng.get("gt")
            lt_val = rng.get("lt")
            # Convert to remapped indices using binary search
            if gt_val is not None:
                lo = 0
                hi = n_db - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if val_by_new_idx[mid] <= gt_val:
                        lo = mid + 1
                    else:
                        hi = mid
                new_start = lo
            else:
                new_start = 0

            if lt_val is not None:
                lo = 0
                hi = n_db - 1
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if val_by_new_idx[mid] >= lt_val:
                        hi = mid - 1
                    else:
                        lo = mid
                new_end = lo
            else:
                new_end = n_db - 1

            r_queries.append((query_vec, new_start, new_end, closest_ids))

print(f"  EMIS (labels) queries: {len(emis_queries)}")
print(f"  EM (submitter) queries: {len(em_queries)}")
print(f"  R (timestamp) queries: {len(r_queries)}")

# ============================================================
# 4. Write query vectors, attributes, and ground truth per type
# ============================================================
K = 25  # GT top-K

# --- EMIS ---
print("Writing EMIS query files...")
emis_vecs = np.array([q[0] for q in emis_queries], dtype=np.float32)
write_fvecs(os.path.join(OUT_DIR, "query_vectors_emis.fvecs"), emis_vecs)
with open(os.path.join(OUT_DIR, "emis_query_attributes.csv"), "w") as f:
    for _, label, _ in emis_queries:
        f.write(str(label) + "\n")
gt_emis = np.array([q[2][:K] for q in emis_queries], dtype=np.int32)
write_ivecs(os.path.join(OUT_DIR, "ground_truth_emis.ivecs"), gt_emis)

# --- EM ---
print("Writing EM query files...")
em_vecs = np.array([q[0] for q in em_queries], dtype=np.float32)
write_fvecs(os.path.join(OUT_DIR, "query_vectors_em.fvecs"), em_vecs)
with open(os.path.join(OUT_DIR, "em_query_attributes.csv"), "w") as f:
    for _, label, _ in em_queries:
        f.write(str(label) + "\n")
gt_em = np.array([q[2][:K] for q in em_queries], dtype=np.int32)
write_ivecs(os.path.join(OUT_DIR, "ground_truth_em.ivecs"), gt_em)

# --- R ---
print("Writing R query files...")
r_vecs = np.array([q[0] for q in r_queries], dtype=np.float32)
write_fvecs(os.path.join(OUT_DIR, "query_vectors_r.fvecs"), r_vecs)
with open(os.path.join(OUT_DIR, "r_query_attributes.csv"), "w") as f:
    for _, start, end, _ in r_queries:
        f.write(f"{start}-{end}\n")
gt_r = np.array([q[3][:K] for q in r_queries], dtype=np.int32)
write_ivecs(os.path.join(OUT_DIR, "ground_truth_r.ivecs"), gt_r)

# ============================================================
# 5. Save encoding maps for reference
# ============================================================
with open(os.path.join(OUT_DIR, "labels_encoding.json"), "w") as f:
    json.dump(labels_encoding, f, indent=2)
with open(os.path.join(OUT_DIR, "submitter_encoding.json"), "w") as f:
    json.dump(submitter_encoding, f, indent=2)

# ============================================================
# Verification
# ============================================================
print("\n=== Verification ===")
for name in sorted(os.listdir(OUT_DIR)):
    path = os.path.join(OUT_DIR, name)
    size = os.path.getsize(path)
    if size > 1024 * 1024:
        print(f"  {name}: {size / 1024 / 1024:.1f} MB")
    else:
        print(f"  {name}: {size / 1024:.1f} KB")

print("\nDone!")
