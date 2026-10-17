# Installation

## Supported platform

The supplied extension targets Linux x86-64 and CPython 3.11. It was built with GCC 10.2 and
requires glibc 2.29 or newer and a `libstdc++.so.6` providing `GLIBCXX_3.4.28`. The extension has
no network dependency at runtime.

Create an environment and install NumPy:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
./run_artifact.sh -m flair_artifact.smoke
```

`run_artifact.sh` adds the precompiled package and reproduction driver to `PYTHONPATH`. If the
Python executable is named differently, set `PYTHON_BIN=/path/to/python3.11`.

The smoke test needs no downloaded dataset. The native extension is intentionally distributed
without implementation source. Its SHA-256, dynamic dependencies, exported symbol list, ABI
versions, and anonymity audit are under `prebuilt/linux-x86_64-cp311/`.

Set `FLAIR_VARIANT` to select an ablation, as described in `prebuilt/ablations/README.md`. The
launcher defaults to `main` and gives the selected package priority over an existing PYTHONPATH.

For a package-integrity and known-identity-string audit (GNU binutils `nm` and `readelf` required):

```bash
python3 scripts/audit_package.py .
```

`MANIFEST.sha256` covers all published files except itself. The auditor ignores local `.git`,
`.venv`, Python caches, `results/reproduced/`, and the three external base-vector symlinks created by
`prepare_data_root.sh`; these are not release content. Any intentional
change to published files requires regenerating the manifest. Passing this audit does not prove
absolute anonymity or replace functional testing.
