# FLAIR Artifact

This repository provides the anonymous FLAIR artifact for SIGMOD 2027.

It intentionally contains no implementation source. The staged Linux x86-64 / CPython 3.11
release is under `prebuilt/linux-x86_64-cp311/`; it exposes the public `flair` Python package and
a stripped native extension. Alongside it are the experiment drivers (`flair_artifact/`), the frozen
configurations (`configs/`), the two precompiled ablation variants (`prebuilt/ablations/`), and the
reproduction documentation (`docs/`).

Use is restricted by `LICENSE` to artifact evaluation and non-commercial academic reproduction.
The precompiled native binaries may not be redistributed, modified, or reverse engineered.

Public interface:

```bash
./run_artifact.sh -m flair_artifact.smoke
./run_artifact.sh -m flair_artifact.run \
  --config configs/memory/gist_enum_range.json \
  --output results/reproduced/gist_enum_range.json
./run_artifact.sh -m flair_artifact.verify \
  --config configs/memory/gist_enum_range.json \
  --actual results/reproduced/gist_enum_range.json
```

Validation is functional, not a comparison with paper measurements. Hardware-dependent metrics
and parameter-search winners are not expected to match across systems; the paper figures and
tables remain the authoritative reported results.

Start with `docs/INSTALL.md`, `docs/DATA.md`, and `docs/REPRODUCING.md`. `docs/DATA.md` records
the exact upstream sources, upstream license status, derivation, and file-verification commands.
`docs/PAPER_MAP.md` maps each paper component to its artifact command. The current distribution
includes prepared composite workloads under `datasets/derived/`; original database vectors must
be obtained separately and linked using `scripts/prepare_data_root.sh`. Final disk configurations
are under `configs/disk/`; see the disk commands and measurement scope in `docs/REPRODUCING.md`.

The repository is provided under the limited Artifact Evaluation and Academic Reproducibility
License in `LICENSE`. It is not an open-source license and does not provide FLAIR implementation
source code.
