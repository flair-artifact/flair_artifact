# Configurations

`paper_memory_parameters.json` freezes the seven in-memory FLAIR configurations used by the
paper. `parameter_candidates.json` records the declared candidate space, conditional refinements,
selection objective, and search-width sweep. Each file under `memory/` is directly executable; set
`DATA_ROOT` to a prepared dataset directory (see `docs/DATA.md`) before running one. Paths may be
absolute, relative to the JSON file, or contain environment variables.

These are frozen paper-environment choices, not expected winners on every machine. Candidate
parameter declarations under `sensitivity/` document the search methodology; rerunning them on
different hardware may select a different winner.

The seven final disk configurations are provided under `disk/`. The selected build parameters
are recorded in `paper_disk_parameters.json`, and the candidate space and selection policy are
in `disk_parameter_candidates.json`. `disk_status.json` summarizes the fixed policy and execution
commands. Each disk configuration runs with `search.parallel_number=4` over 11 search widths,
with the I/O limit equal to the search list size. See `docs/REPRODUCING.md` for the commands and
measurement scope. Paper result curves are not used as artifact validation targets.
