# Native ablation modules

The two Linux x86-64 / CPython 3.11 packages are under
`linux-x86_64-cp311/VARIANT/flair/`. Each is a complete drop-in replacement for the main
`flair` package. Select a variant explicitly through the launcher:

```bash
export FLAIR_VARIANT=no_build_and_query_label_aware_entry_points
./run_artifact.sh -m flair_artifact.smoke
./run_artifact.sh -m flair_artifact.run \
  --config configs/memory/gist_enum_range.json \
  --output results/reproduced/no_build_and_query_label_aware_entry_points.json
./run_artifact.sh -m flair_artifact.verify \
  --config configs/memory/gist_enum_range.json \
  --actual results/reproduced/no_build_and_query_label_aware_entry_points.json
```

Set `FLAIR_VARIANT=main` for the main package. Each completed result records the selected variant
and native binary SHA-256. The example reuses the index path from the configuration; run variants
sequentially or assign separate index paths in separate configurations for concurrent runs.

| Variant | Native change |
|---|---|
| `no_label_aware_pruning` | Disable label-compatibility pruning during graph construction. |
| `no_build_and_query_label_aware_entry_points` | Use ordinary entry points during both graph construction and querying. |

The entry-point variant enforces the query predicate on every returned result.
Ordinary entry points may seed graph traversal, but cannot enter the result heap unless they match
the predicate. The supplied ablation patches affect the FlatNSW backend; DiskANN is included in
these packages for basic functionality checks, not as a disk ablation implementation.

Each variant contains `audit.json`, `dynamic_exports.txt`, and `SHA256SUMS`. The extension exports
only `PyInit__native`. Run the smoke command to check the selected package on your system.
