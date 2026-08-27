# Registered rare-scenario taxonomy

## Scope

scRareBench can attach fixed rare-population scenario metadata to registered datasets without modifying their downloaded expression matrices or source H5AD files.

The six scenario labels combine a distribution class with a topology class:

```text
GR-DL  GR-RM  LE-DL  LE-RM  SR-DL  SR-RM
```

- **GR**: globally rare
- **LE**: locally enriched
- **SR**: sample-restricted
- **DL**: distinct lineage / relatively distinct identity
- **RM**: related manifold / state, subtype, activation or differentiation program

DL/RM assignments are annotation-driven and should not be inferred automatically from abundance alone. Ambiguous populations may be retained in provenance while excluded from the six-state analysis.

## Registered coverage

- **mBDRC renal cortex**: 10 assigned populations covering `GR-DL`, `GR-RM`, `LE-DL`, and `LE-RM`; one additional topology-ambiguous population is retained as provenance.
- **Wu breast-cancer atlas**: 17 assigned populations covering all six scenarios.
- **COVID-19 autoimmunity PBMC**: 12 assigned populations covering `GR-DL`, `GR-RM`, `LE-DL`, and `SR-DL`.

The dashboard preserves all six scenario slots even when a dataset has no registered populations in a given slot.

## Source-integrity rule

`download_dataset()` downloads/caches the published H5AD without scenario edits. `load_dataset()` attaches registered scenario metadata **in memory** and validates that the registered source labels still exist. If a source annotation revision removes or renames a registered label, the loader fails loudly instead of silently remapping it.

## Custom datasets

For user datasets, rare populations may be supplied either as a simple `rare_types=[...]` list or as an explicit scenario table through `register_dataset()`. If no defensible DL/RM topology exists, use distribution-only `GR`/`LE`/`SR` metadata with unassigned topology rather than inventing a six-state assignment.
