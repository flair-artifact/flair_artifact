# Derived workload data and the generators that produced it

This directory makes the composite workloads reproducible end to end. It contains
the generation scripts (`generators/`), the data those scripts produced
(`derived/`), and the checksums of the upstream public inputs
(`public_files.json`).

Base vectors are **not** redistributed here: they are multi-gigabyte copies of
public datasets. Everything derived from them — attribute columns, predicates,
query vectors, and exhaustive ground truth — is bundled.

## Verify what is bundled

```bash
python3 scripts/verify_derived_data.py
```

This checks three things and prints `DERIVED DATA VERIFICATION PASSED`:

1. every file under `datasets/derived/` and `datasets/generators/` matches its
   recorded SHA-256, and no file is present that the manifests do not cover;
2. the structural contract holds — predicate rows, query-vector rows, and
   ground-truth rows agree per workload; ground-truth width equals the recorded
   top-k; every ground-truth id is within the database size;
3. every `${DATA_ROOT}` path referenced by `configs/memory/*.json` exists, except
   `database_vectors.fvecs`, which you link in the next step.

## Run the experiments against the bundled data

```bash
python3 scripts/verify_public_files.py --dataset gist --root /downloads/gist
scripts/prepare_data_root.sh --data-root datasets/derived \
  --gist-vectors /downloads/gist/gist_base.fvecs
export DATA_ROOT=$PWD/datasets/derived
./run_artifact.sh -m flair_artifact.run \
  --config configs/memory/gist_enum_range.json \
  --output results/reproduced/gist_enum_range.json
```

`prepare_data_root.sh` only creates symlinks; it never copies or modifies the
public files. Add `--arxiv-vectors` and `--qdrant-vectors` as those datasets
become available. Run `bash scripts/run_all_memory.sh` only after all three database-vector files
are linked. Qdrant ships `vectors.npy`, so convert it first with
`generators/filtered_ann_datasets/qdrant_arxiv/convert_to_fanns.py`.

## Layout of `derived/`

```text
derived/
  MANIFEST.json                  machine-readable contract used by the verifier
  SHA256SUMS                     checksums for every file below
  DATASET/
    database_vectors.fvecs       NOT bundled; created by prepare_data_root.sh
    bins_query_vectors.fvecs     the 1,000 public query vectors in source order
    database/                    attribute columns for the whole database
    attributes_source/           intermediate arrays the generators read/write
    SCENARIO/                    the mixed workload used by configs/memory/
      query_vectors.fvecs        1,000 queries in workload order
      groundtruth.ivecs          exhaustive top-k under both predicates
      attribute_0.query.csv      predicate for the first attribute dimension
      attribute_1.query.csv      predicate for the second attribute dimension
      selectivity.jsonl          target vs measured joint selectivity per query
      query_source_indices.csv   which public query vector each row uses
      bins/selectivity_P/        per-selectivity breakdown workload (P in 1..60)
```

`SCENARIO` is `enum_enum`, `range_range`, or `enum_range`. Every selectivity bin
reuses `bins_query_vectors.fvecs`: the bins query the 1,000 public vectors in
source order, so all 18 bins of a dataset share one identical query-vector file
and the artifact stores it once. The `mixed` workload uses a fixed 1,000-query
draw across the six selectivity targets, so its `query_vectors.fvecs` is
scenario-specific.

## Attribute mapping

| Dataset | Scenario | `attribute_0` | `attribute_1` | top-k |
|---|---|---|---|---|
| gist | enum_enum | `database/enum1.csv` (single label, Zipf 1.0, 20 labels) | `database/enum2.csv` (1--10 labels, Zipf 1.2, 20 labels) | 10 |
| gist | range_range | `database/range_int.csv` (sorted synthetic int32) | `database/range_float.csv` (uniform float32, order-preserving uint32) | 10 |
| gist | enum_range | `database/enum2.csv` | `database/range_int.csv` | 10 |
| arxiv | enum_enum | `database/em.csv` (sub-categories) | `database/emis.csv` (multi-valued main categories) | 100 |
| arxiv | range_range | `database/r.csv` (update date rank) | `database/r2.csv` (author count rank) | 100 |
| arxiv | enum_range | `database/emis.csv` | `database/r.csv` | 100 |
| qdrant | enum_range | `database/emis.csv` (176 labels) | `database/r.csv` (timestamp rank) | 25 |

Ground-truth width is the recorded top-k above. The bundled configurations ask
for top-10, which the harness takes as the first ten columns; the wider files are
kept so the same ground truth also serves larger-k evaluations.

Enum rows are OR-combined within a dimension; dimensions are AND-combined. Range
query rows are inclusive `low,high` pairs. Float range values are stored in the
order-preserving uint32 encoding, so integer comparison reproduces float ordering.

## Regenerating from the public inputs

The runtime requires NumPy only. Regeneration also requires `h5py`:

```bash
python -m pip install h5py
```

The generators keep their original relative layout, so run them from
`datasets/generators/` with the public inputs placed at these paths:

| Path under `generators/` | Public input |
|---|---|
| `filtered_ann_datasets/texmex_gist1m/gist_base.fvecs`, `gist_query.fvecs` | TexMex GIST1M |
| `filtered_ann_datasets/texmex_gist1m/gist-960-euclidean.hdf5` | ANN-Benchmarks GIST mirror |
| `datasets/arxiv-for-fanns-large/raw/*.jsonl` | SPCL/arxiv-for-fanns-large |
| `datasets/arxiv-for-fanns-large/hdf5/query.hdf5`, `database_vectors.npy` | SPCL/arxiv-for-fanns-large |
| `filtered_ann_datasets/qdrant_arxiv/{vectors.npy,payloads.jsonl,tests.jsonl}` | qdrant/ann-filtering-benchmark-datasets |

```bash
cd datasets/generators

# GIST1M attributes: synthetic range-1, float range-2, enum-1, multi-valued enum-2
python3 filtered_ann_datasets/texmex_gist1m/generate_range_attribute.py
python3 filtered_ann_datasets/texmex_gist1m/generate_attributes.py
python3 filtered_ann_datasets/texmex_gist1m/generate_enum2_k50_candidate.py --num-labels 20

# Arxiv attribute encoding from the upstream JSONL, including the derived range-2
python3 datasets/arxiv-for-fanns-large/generate_csv.py

# Qdrant conversion from the upstream npy/jsonl payload
python3 filtered_ann_datasets/qdrant_arxiv/convert_to_fanns.py

# Controlled composite predicates, seed 42, six selectivity targets, 1,000 queries each
python3 filtered_ann_datasets/texmex_gist1m/generate_controlled_composite_queries.py \
  --output filtered_ann_datasets/texmex_gist1m/composite_controlled
python3 datasets/arxiv-for-fanns-large/generate_controlled_composite_queries.py \
  --output datasets/arxiv-for-fanns-large/composite_controlled
python3 filtered_ann_datasets/qdrant_arxiv/generate_controlled_composite_queries.py \
  --output filtered_ann_datasets/qdrant_arxiv/fanns_format/composite_controlled

# Exhaustive ground truth over the full database
python3 generate_controlled_ground_truth.py gist --scope all
python3 generate_controlled_ground_truth.py arxiv --scope all
python3 generate_controlled_ground_truth.py qdrant --scope all
```

The default `--output` of each predicate generator is a `*_smoke` directory, so
the explicit paths above matter. `generate_selectivity_ground_truth_parallel.py`
is the sharded variant of the ground-truth stage; it produces the same rows, and
its `--workers` value only changes how the query range is split.

Regenerated bytes are **not** guaranteed to be identical to the bundled files.
`docs/DATA.md` states exactly which stages are reproducible bit-for-bit, which
depend on a pinned NumPy version, and which depend on the BLAS implementation.
The bundled files plus `SHA256SUMS` are the authoritative inputs for the reported
measurements; use the generators to inspect or extend the workload definition,
not to replace the bundled ground truth.
