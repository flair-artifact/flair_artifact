"""
Generate synthetic attributes for GIST1M dataset:
- 1 float32 range attribute (uniform [0,1))
- 2 enum attributes (Zipf distribution, 20 classes and 200 classes)
- Regenerate range query files for 1000 queries (matching gist_query.fvecs)
"""
import numpy as np
import struct
import os

SEED = 42
N = 1_000_000
NQ = 1000
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(SEED)

# ============================================================
# 1. Float range attribute
# ============================================================
print("Generating float range attribute...")
float_attr = np.random.uniform(size=N).astype(np.float32)
np.save(os.path.join(OUTPUT_DIR, "gist_float_attributes.npy"), float_attr)
print(f"  float_attr: dtype={float_attr.dtype}, shape={float_attr.shape}, "
      f"min={float_attr.min():.6f}, max={float_attr.max():.6f}")

# ============================================================
# 2. Enum attributes via Zipf distribution
# ============================================================
def generate_zipf_enum(n, k, s, seed):
    """Generate enum attribute with Zipf(s) distribution over k classes."""
    rng = np.random.RandomState(seed)
    ranks = np.arange(1, k + 1)
    weights = 1.0 / (ranks ** s)
    probs = weights / weights.sum()
    labels = rng.choice(k, size=n, p=probs)
    return labels.astype(np.int32)

print("Generating enum-1 (Zipf s=1.0, K=20)...")
enum1 = generate_zipf_enum(N, k=20, s=1.0, seed=SEED + 1)
np.save(os.path.join(OUTPUT_DIR, "gist_enum1_attributes.npy"), enum1)
unique1, counts1 = np.unique(enum1, return_counts=True)
print(f"  enum1: {len(unique1)} classes, top-5 counts: {sorted(counts1, reverse=True)[:5]}")

print("Generating enum-2 (Zipf s=1.2, K=200)...")
enum2 = generate_zipf_enum(N, k=200, s=1.2, seed=SEED + 2)
np.save(os.path.join(OUTPUT_DIR, "gist_enum2_attributes.npy"), enum2)
unique2, counts2 = np.unique(enum2, return_counts=True)
print(f"  enum2: {len(unique2)} classes, top-5 counts: {sorted(counts2, reverse=True)[:5]}")

# ============================================================
# 3. Regenerate int32 range query files for 1000 queries
# ============================================================
print("\nRegenerating int32 range query files for 1000 queries...")
int_attr = np.load(os.path.join(OUTPUT_DIR, "gist_attributes.npy"))
attr_min, attr_max = int(int_attr.min()), int(int_attr.max())
attr_range = attr_max - attr_min + 1

ranges_dir = os.path.join(OUTPUT_DIR, "ranges")
os.makedirs(ranges_dir, exist_ok=True)

rng_range = np.random.RandomState(SEED + 100)

for f_level in range(18):
    if f_level == 17:
        # Mixed: cycle through levels 0~10
        queries = []
        for qi in range(NQ):
            sub_level = qi % 11
            width = max(1, int(attr_range * (2.0 ** (-sub_level))))
            lo = rng_range.randint(attr_min, attr_max - width + 2)
            hi = lo + width - 1
            queries.append((lo, hi))
    else:
        selectivity = 2.0 ** (-f_level)
        width = max(1, int(attr_range * selectivity))
        queries = []
        for _ in range(NQ):
            lo = rng_range.randint(attr_min, attr_max - width + 2)
            hi = lo + width - 1
            queries.append((lo, hi))

    outpath = os.path.join(ranges_dir, f"{f_level}.bin")
    with open(outpath, 'wb') as fp:
        for lo, hi in queries:
            fp.write(struct.pack('<ii', lo, hi))

    if f_level <= 5 or f_level == 17:
        sel = selectivity if f_level < 17 else "mixed"
        print(f"  level {f_level:2d}: selectivity={sel}, width={width if f_level<17 else 'varies'}, "
              f"sample [{queries[0][0]}, {queries[0][1]}]")

# ============================================================
# 4. Generate float range query files for 1000 queries
# ============================================================
print("\nGenerating float range query files...")
float_ranges_dir = os.path.join(OUTPUT_DIR, "float_ranges")
os.makedirs(float_ranges_dir, exist_ok=True)

rng_float = np.random.RandomState(SEED + 200)

for f_level in range(17):  # 0~16, no mixed level for float
    selectivity = 2.0 ** (-f_level)
    width = selectivity  # in [0,1) uniform, width = selectivity

    queries = []
    for _ in range(NQ):
        lo = rng_float.uniform(0.0, 1.0 - width)
        hi = lo + width
        queries.append((np.float32(lo), np.float32(hi)))

    outpath = os.path.join(float_ranges_dir, f"{f_level}.npy")
    arr = np.array(queries, dtype=np.float32)  # shape (NQ, 2)
    np.save(outpath, arr)

    if f_level <= 5 or f_level == 16:
        print(f"  level {f_level:2d}: selectivity={selectivity:.6f}, "
              f"sample [{queries[0][0]:.4f}, {queries[0][1]:.4f}]")

# ============================================================
# 5. Generate enum query files for 1000 queries
# ============================================================
print("\nGenerating enum query files...")

def generate_enum_queries(attr, n_queries, seed):
    """Generate single-value exact-match queries. Sample from the attribute's actual distribution."""
    rng = np.random.RandomState(seed)
    unique_vals = np.unique(attr)
    query_labels = rng.choice(unique_vals, size=n_queries)
    return query_labels.astype(np.int32)

enum1_queries = generate_enum_queries(enum1, NQ, seed=SEED + 300)
np.save(os.path.join(OUTPUT_DIR, "gist_enum1_queries.npy"), enum1_queries)
print(f"  enum1 queries: {NQ} queries, {len(np.unique(enum1_queries))} unique values")

enum2_queries = generate_enum_queries(enum2, NQ, seed=SEED + 400)
np.save(os.path.join(OUTPUT_DIR, "gist_enum2_queries.npy"), enum2_queries)
print(f"  enum2 queries: {NQ} queries, {len(np.unique(enum2_queries))} unique values")

# ============================================================
# Summary
# ============================================================
print("\n=== Summary ===")
print(f"Dataset: GIST1M, N={N}, NQ={NQ}")
print(f"Range-1 (int32): gist_attributes.npy (existing), ranges/*.bin (regenerated for {NQ}q)")
print(f"Range-2 (float32): gist_float_attributes.npy, float_ranges/*.npy")
print(f"Enum-1 (Zipf 1.0, K=20): gist_enum1_attributes.npy, gist_enum1_queries.npy")
print(f"Enum-2 (Zipf 1.2, K=200): gist_enum2_attributes.npy, gist_enum2_queries.npy")
print("Done!")
