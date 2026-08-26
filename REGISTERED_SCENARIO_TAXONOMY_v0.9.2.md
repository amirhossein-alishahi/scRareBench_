# Registered scenario taxonomy - v0.9.2

## Scope

This release adds fixed benchmark metadata for registry datasets 2, 3, and 4. It does not modify expression matrices or downloaded source H5AD files.

DL/RM assignments are provisional and annotation-driven, following the supplied Rare Cell Benchmark Report:
- DL (Distinct Lineage): annotation denotes a relatively distinct lineage/cell identity.
- RM (Related Manifold): annotation denotes a state, subtype, activation/differentiation program, or otherwise related manifold with a plausible abundant parent.
- Ambiguous: retained in provenance but excluded from the six-state scenario analysis.

## Coverage

- Dataset 2 - mBDRC renal cortex: 10 assigned, 1 topology-ambiguous; GR-DL, GR-RM, LE-DL, LE-RM.
- Dataset 3 - Wu breast-cancer atlas: 17 assigned; all six scenarios.
- Dataset 4 - COVID-19 autoimmunity PBMC: 12 assigned; GR-DL, GR-RM, LE-DL, SR-DL.

The dashboard intentionally displays all six scenario slots even when a dataset has zero registered cell types for a scenario.

## Reproducibility / source-integrity rule

`download_dataset()` still downloads/caches published H5AD data without scenario edits. `load_dataset()` attaches scenario metadata in memory and validates that registered cell-type labels still exist. A source annotation revision that removes/renames registered labels fails loudly rather than silently remapping them.
