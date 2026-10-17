# Generated results

Experiment outputs are written under `results/reproduced/` and are intentionally excluded from
the distributed artifact. The paper figures and tables are the authoritative reported results.

The artifact does not compare recall, throughput, build time, index size, or parameter-search
winners against a bundled reference curve. These values can change with CPU model, compiler,
instruction set, thread scheduling, storage, and system load. `flair_artifact.verify` checks only
that a run completed with driver correctness checks, matches its configuration and selected
binary, and produced a structurally valid, finite, complete curve for the configured search widths.
