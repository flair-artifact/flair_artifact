# Paper-to-artifact map

This map covers the provided FLAIR composite-filter workflows. The artifact does not include
complete execution workflows for baseline methods or single-attribute experiments.

| Paper component | Artifact entry point | Frozen methodology metadata |
|---|---|---|
| Composite in-memory experiments | `scripts/run_all_memory.sh` | `configs/memory/` and `configs/paper_memory_parameters.json` |
| Representative GIST enum–range | `flair_artifact.run` | `configs/memory/gist_enum_range.json` |
| FLAIR build time and index size | Fields emitted by the same run | Same configuration files |
| Composite disk experiments | `scripts/run_all_disk.sh` | `configs/disk/`, `configs/paper_disk_parameters.json`, and `configs/disk_parameter_candidates.json` |
| Entry-point and pruning ablations | Packages under `prebuilt/ablations/linux-x86_64-cp311/` | `prebuilt/ablations/README.md` |
| FLAIR parameter sensitivity | `flair_artifact.sweep` | `configs/sensitivity/` |

No bundled curve is treated as a canonical pass/fail target. The paper figures and tables are the
authoritative measurements from the disclosed paper environment. Artifact validation covers
functionality and output integrity only.
