# Reproducing the experiments

## What reproduction means

The artifact validates executable behavior, not equality with measurements obtained on the paper
machines. CPU model, compiler, SIMD selection, thread scheduling, storage, and system load can
change throughput and build time. Parallel graph construction can also change the resulting graph,
recall curve, and parameter-search winner.

Accordingly, the artifact does not bundle paper curves as pass/fail references. A run passes when
the package imports, the dataset contract holds, the index builds/dumps/opens, filtered queries
return only predicate-matching IDs, and the requested search-width curve is complete with finite
metrics. The paper figures and tables are the authoritative reported results.

## Smoke mode

This runs self-contained 500-vector enum+range checks for the in-memory and disk backends. It
checks index lifecycle and filtered-result correctness and requires no external dataset:

```bash
./run_artifact.sh -m flair_artifact.smoke
```

## Representative mode

Verify the bundled attributes, predicates, query vectors, and ground truth, then link a verified
GIST base-vector file. The full database vectors are not bundled:

```bash
python3 scripts/verify_derived_data.py
scripts/prepare_data_root.sh --data-root datasets/derived \
  --gist-vectors /path/to/gist_base.fvecs
```

Set DATA_ROOT to the bundled derived tree:

```bash
export DATA_ROOT=$PWD/datasets/derived
./run_artifact.sh -m flair_artifact.run \
  --config configs/memory/gist_enum_range.json \
  --output results/reproduced/gist_enum_range.json
./run_artifact.sh -m flair_artifact.verify \
  --config configs/memory/gist_enum_range.json \
  --actual results/reproduced/gist_enum_range.json
```

The driver checks ground-truth IDs, returned ID ranges and uniqueness, predicate matching, and
finite distances. Ground-truth padding may use -1; returned IDs must be valid database IDs. Empty
or fewer-than-topk results are allowed, without imposing a recall threshold. The verifier checks
completion, dataset/scenario, exact configuration and binary hashes, selected variant, topk,
search-width coverage, per-width query-check counts, and numeric validity. It deliberately does not impose recall, QPS, build-time, index-size, or winner
equivalence.

The public Python driver performs one filtered call per query, so its QPS includes Python-loop and
filter-construction overhead. Predicate validation and recall computation are outside the query
timer. Build time includes construction and index serialization, after input checks. It must not be treated as numerically interchangeable with the
paper's native batch-oriented harness.

## Full in-memory mode

```bash
export DATA_ROOT=$PWD/datasets/derived
scripts/run_all_memory.sh
scripts/verify_all_memory.sh
```

All three datasets must have their verified database vectors linked before this batch command.
For one available dataset, use its individual configuration as shown above.
This builds seven indexes and evaluates 11 search widths per index. Outputs are written under
`results/reproduced/`, which is ignored by version control and excluded from the release archive.

## Parameter search and sensitivity

`configs/paper_memory_parameters.json` records the parameters frozen for the paper environment.
`configs/parameter_candidates.json` records the paper parameter-search candidate space and
selection policy. The separate one-factor sensitivity runs under `configs/sensitivity/` do not
implement the full coordinate-search winner-selection algorithm. Neither workflow requires
another machine to select the same winner.

For example:

```bash
./run_artifact.sh -m flair_artifact.sweep \
  --sweep configs/sensitivity/gist_enum_range.json \
  --output-dir results/reproduced/sensitivity/gist_enum_range
```

Each candidate produces its own result JSON. Users should interpret the metrics in the context of
their own machine rather than compare them against a bundled paper curve.

## Disk mode

The seven final disk configurations are under `configs/disk/`. Each configuration builds one
index, then evaluates both query parallel settings (1 and 4) at 11 search widths. The parallel
setting maps to `search.parallel_number`; query calls are issued sequentially by one Python
thread. For every width, `diskann.search.io_limit` equals `diskann.search.list_size` (1x budget).
The driver explicitly disables layout reordering, sector-aligned reads, and node caching. PQ
chunk counts are 240 for GIST, 256 for Arxiv, and 96 for Qdrant; build thread count is 30.
The 256-entry construction lock pool is compiled into the supplied binary and is not a runtime
option. `configs/paper_disk_parameters.json` records the final choices; the candidate sets and
selection policy are in `configs/disk_parameter_candidates.json`.

After preparing DATA_ROOT as above, run one available scenario:

```bash
export DATA_ROOT=$PWD/datasets/derived
./run_artifact.sh -m flair_artifact.run \
  --config configs/disk/gist_enum_range.json \
  --output results/reproduced/disk/gist_enum_range.json
./run_artifact.sh -m flair_artifact.verify \
  --config configs/disk/gist_enum_range.json \
  --actual results/reproduced/disk/gist_enum_range.json
```

To run all seven scenarios, with all three database-vector files available:

```bash
scripts/run_all_disk.sh
scripts/verify_all_disk.sh
```

Results contain 22 rows per scenario, identified by `(parallel_number, search_width)`. The
verifier checks both complete series and the recorded effective query/searcher options, in
addition to the same ID, predicate, provenance, and numeric checks as memory mode. This does
not impose a recall or throughput target. Disk results produced before this driver update
must be regenerated to include the effective options and cache policy.

The paper harness used cold-cache measurements. This functional Python driver does not clear
the operating-system page cache; it records that policy explicitly in each result. Disabling
the index node cache does not disable the operating-system cache. It trains PQ using up to the
first 100,000 database vectors, builds and serializes through the Python API, and includes
Python call/filter overhead in query timing. Its measurements are therefore not numerically
interchangeable with the paper harness. Neither global cache-clearing privileges nor repeated
full-dataset measurements are required for functional validation. The public package provides
the frozen parameters and candidate sets, without the original private search harness, raw
performance results, or a requirement to reproduce its winner.

The two ablation variants modify the in-memory FlatNSW algorithm. Disk configurations should
use the main package (`FLAIR_VARIANT=main`, the default).

## Ablations

The two native variants are under `prebuilt/ablations/linux-x86_64-cp311/`; see the adjacent
README for their semantics and selection instructions. Use the same selected variant for a run
and its verification. Build and result output paths should be distinct for each variant. Full
ablation metrics can be generated with the experiment driver; no reference curves are bundled.

## Result identity and limitations

Results use format version 2. Old result JSON files must be regenerated with the current driver.
The checks are consistency and functionality checks, not proof that a result file cannot be edited.
Configuration hashes do not hash the contents of external input files; verify the upstream inputs
and keep prepared data unchanged between running and verifying. Ground-truth checking validates
IDs and predicates, not exhaustive nearest-neighbor ordering. No metric-equivalence thresholds
are applied. The paper parameter-search recall target is a selection rule, not a validation gate.

The release platform is described in INSTALL.md. Resource requirements for the full workloads
have not yet been measured for this driver; provision space for the inputs and generated indexes.
