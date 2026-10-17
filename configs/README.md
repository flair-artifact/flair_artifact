# Configurations

`paper_memory_parameters.json` freezes the seven in-memory FLAIR configurations used by the
paper. `parameter_candidates.json` records the declared candidate space, conditional refinements,
selection objective, and search-width sweep. Each file under `memory/` is directly executable; set
`DATA_ROOT` to a prepared dataset directory (see `docs/DATA.md`) before running one. Paths may be
absolute, relative to the JSON file, or contain environment variables.

These are frozen paper-environment choices, not expected winners on every machine. Candidate
parameter declarations under `sensitivity/` document the search methodology; rerunning them on
different hardware may select a different winner.

The disk experiment is still being finalized. `disk_status.json` records the fixed fairness
policy without publishing a superseded winner. It will be replaced by executable disk configs
and final selected parameters once the outstanding run has completed. Paper result curves are not
used as artifact validation targets.
