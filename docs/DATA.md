# Datasets and reproducibility contract

The artifact does not redistribute any full dataset. Download each dataset from its upstream
publisher, verify the public files, and use the bundled composite workloads described below.
The SHA-256 values in `datasets/public_files.json` were computed from the exact upstream files
used for the paper experiments.

## Public sources and license status

| Dataset | Upstream source | License status checked 2026-10-17 |
|---|---|---|
| GIST1M | [TexMex corpus](http://corpus-texmex.irisa.fr/) (also mirrored by [ANN-Benchmarks](http://ann-benchmarks.com/gist-960-euclidean.hdf5)) | The upstream dataset page does not state an explicit data license. Obtain and use the data under the upstream publisher's terms. |
| Arxiv-large | [SPCL/arxiv-for-fanns-large](https://huggingface.co/datasets/SPCL/arxiv-for-fanns-large) | The dataset repository does not declare an explicit data license. Obtain and use the data under the upstream publisher's terms. |
| Qdrant Arxiv | [qdrant/ann-filtering-benchmark-datasets](https://github.com/qdrant/ann-filtering-benchmark-datasets) | The repository has no root `LICENSE`, and the GitHub repository metadata reports no SPDX license. Obtain and use the data under the upstream publisher's terms. |

These statements concern the datasets, not the licenses of downloader or benchmark code. The
artifact's own eventual repository license will not grant rights to the upstream data.

## Verify the exact public inputs

The verifier checks both byte count and SHA-256 and does not modify its input:

```bash
python3 scripts/verify_public_files.py --dataset gist --root /path/to/gist/files
python3 scripts/verify_public_files.py --dataset arxiv --root /path/to/arxiv/files
python3 scripts/verify_public_files.py --dataset qdrant --root /path/to/qdrant/files
```

GIST verification covers `gist_base.fvecs` and `gist_query.fvecs`. Arxiv verification covers the
database/query vectors, database/query attributes, and upstream single-filter ground truths.
Qdrant verification covers `vectors.npy`, `payloads.jsonl`, and `tests.jsonl`.

## Exact derivation used by the experiments

Every random stage is seeded and single-threaded. The controlled composite predicates use seed 42
through NumPy's `default_rng`; the GIST attribute generators use seed 42 with fixed per-attribute
offsets through the legacy `RandomState`; the GIST range-1 attribute uses seed 123 through the
legacy `RandomState`. Composite ground truth is recomputed by exhaustive scan over the full
database after applying both predicates; it is not copied from either upstream single-filter
workload.

### GIST1M

The paper uses the one-million 960-dimensional base vectors and 1,000 query vectors from GIST1M.
The attributes are synthetic because GIST1M contains vectors only:

- enum 1: one label per vector, Zipf exponent 1.0 over 20 labels;
- enum 2: 1--10 distinct labels per vector, Zipf exponent 1.2 over 20 labels; the label-count
  distribution is proportional to `0.5 ** (m - 1)`;
- range 1: a synthetic sorted integer attribute;
- range 2: independent uniform float32 values in `[0, 1)`, encoded order-preservingly as uint32.

The controlled workload targets joint selectivities 1%, 5%, 10%, 20%, 40%, and 60%, then draws a
fixed 1,000-query mixed set. The experiments use the 20-label enum-2 candidate, not the older
200-label intermediate file.

### Arxiv-large

The upstream files are used as follows:

- enum 1 (`EM`): `number_of_sub_categories`, deterministically integer encoded;
- enum 2 (`EMIS`): the multi-valued `main_categories`, deterministically integer encoded;
- range 1: `update_date`, stable sort-and-remap to integer rank;
- range 2: `number_of_authors`, stable sort-and-remap to integer rank.

Range 2 and every two-attribute predicate/ground-truth pair are artifact-derived additions. The
controlled workload uses the same six joint-selectivity targets and fixed 1,000-query mixture as
GIST1M.

### Qdrant Arxiv

`vectors.npy`, `payloads.jsonl`, and `tests.jsonl` are the upstream inputs. The paper's composite
scenario uses only:

- enum: multi-valued `labels` (176 observed categories, deterministic integer encoding);
- range: `update_date_ts`, stable sort-and-remap to integer rank.

The upstream `submitter` field is not used in the reported composite experiment. Composite
predicates and exhaustive ground truth use the same fixed controlled-workload procedure.

## Bundled derived data

The derived attributes, predicates, query vectors, and exhaustive ground truth are bundled under
`datasets/derived/`, together with the generator scripts under `datasets/generators/`.
`datasets/README.md` documents the layout, the per-scenario attribute mapping, and the regeneration
command sequence. Only the base vectors are left out, because they are multi-gigabyte copies of the
public datasets listed above.

Set `DATA_ROOT` to the bundled tree after linking the base vectors:

```bash
python3 scripts/verify_derived_data.py
scripts/prepare_data_root.sh --data-root datasets/derived \
  --gist-vectors /downloads/gist/gist_base.fvecs
export DATA_ROOT=$PWD/datasets/derived
```

The layout is:

```text
DATA_ROOT/
  DATASET/
    database_vectors.fvecs         linked from the verified public download
    bins_query_vectors.fvecs       1,000 public queries in source order
    database/                      attribute columns for the whole database
    attributes_source/             intermediate arrays used by the generators
    SCENARIO/
      query_vectors.fvecs          1,000 queries in workload order
      groundtruth.ivecs
      attribute_0.query.csv
      attribute_1.query.csv
      selectivity.jsonl
      query_source_indices.csv
      bins/selectivity_P/          P in 1, 5, 10, 20, 40, 60
```

`DATASET` is `gist`, `arxiv`, or `qdrant`; `SCENARIO` is `enum_enum`, `range_range`, or
`enum_range` where present in `configs/memory/`.

Query vectors live **inside each scenario directory**, not at the dataset level. The mixed workload
draws a fixed 1,000-query mixture per scenario, and the three scenarios of a dataset draw
different query orders, so a dataset-level query file would misalign two of the three ground-truth
files. Ground-truth row `i` always corresponds to `query_vectors.fvecs` row `i` of the same
scenario directory.

Vector files use standard little-endian `fvecs`: each row begins with an int32 dimension followed
by float32 values. Ground truth uses `ivecs`: each row begins with an int32 width followed by int32
IDs. Attribute CSV files have no header. An enum row contains one or more integer labels; a range
database row contains one scalar; a range query row contains inclusive `low,high` endpoints.
Predicates are OR within a multi-valued enum dimension and AND across dimensions.

## Determinism of each generation stage

Regenerating the data is supported and documented, but bit-for-bit identity is only guaranteed for
some stages. The bundled files and their SHA-256 values are the authoritative inputs behind the
reported measurements.

| Stage | Source of variation | Reproducibility |
|---|---|---|
| GIST attributes, enum-2 candidate | legacy `RandomState`, whose stream NumPy keeps frozen across releases | byte-identical on any NumPy version |
| Controlled composite predicates, all three datasets | NumPy `default_rng`, whose stream NumPy does **not** guarantee across releases | byte-identical on the pinned NumPy; may differ on another major version |
| Attribute encoding from upstream metadata | no randomness; pure integer encoding and stable sorting | byte-identical |
| Exhaustive ground truth | BLAS accumulation order in the distance matrix, plus non-stable `argpartition`/`argsort` tie-breaking | same rows in almost all cases; near-ties can flip |

The ground-truth stage computes L2 distances as a matrix product followed by norm correction, so the
floating-point accumulation order depends on the BLAS build, the thread count, and the instruction
set. `--query-batch` and `--database-block` also change the accumulation order. When two candidates
are within a few float32 ulps of each other at the top-k boundary, the selection and the ordering of
tied ids can differ. This shows up as recall differences of a few thousandths at most, and it is
expected rather than a defect: it is why the artifact ships the ground truth it measured against
instead of asking a reviewer to recompute it.

Pinned environment used for the bundled files: CPython 3.11, NumPy 2.x, single-threaded BLAS
(`OMP_NUM_THREADS=1`). The verifier does not depend on these versions, because it checks the
bundled bytes rather than regenerated ones.
