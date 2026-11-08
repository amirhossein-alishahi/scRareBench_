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
- **NYGC / Seurat v4 CITE-seq PBMC**: 9 assigned populations covering `GR-DL`, `GR-RM`, `SR-DL`, and `SR-RM` after the fixed official scvi-tools PBMC QC filter.

The dashboard preserves all six scenario slots even when a dataset has no registered populations in a given slot.

## Source-integrity rule

`download_dataset()` downloads/caches the published H5AD without scenario edits. `load_dataset()` attaches registered scenario metadata **in memory** and validates that the registered source labels still exist. If a source annotation revision removes or renames a registered label, the loader fails loudly instead of silently remapping it.

## Custom datasets

For user datasets, rare populations may be supplied either as a simple `rare_types=[...]` list or as an explicit scenario table through `register_dataset()`. If no defensible DL/RM topology exists, use distribution-only `GR`/`LE`/`SR` metadata with unassigned topology rather than inventing a six-state assignment.

## Reviewed topology assignments (November 2026)

The following **three topology-only corrections** were made after targeted
marker-expression and source-annotation review. Source H5AD cell-type labels,
rarity/distribution classes, and the underlying cells are **not** changed.

| Dataset | Original H5AD benchmark label | Previous | Reviewed | Evidence |
| --- | --- | --- | --- | --- |
| Dataset 3 — Wu breast-cancer atlas | `Myeloid_c4_DCs_pDC_IRF7` | GR-RM | **GR-DL** | The pDC lineage is distinct from cDC1/cDC2/LAMP3 DC; LILRA4, CLEC4C, TCF4 and IRF7 support the pDC identity. |
| Dataset 4 — COVID PBMC | `B cell` | GR-DL | **GR-RM** | The 510 benchmark-labeled cells match the `Exhausted B` source annotation (510/510); exhaustion is a B-lineage state. |
| Dataset 4 — COVID PBMC | `natural killer cell` | GR-DL | **GR-RM** | The 226 benchmark-labeled cells match the `Prolif NK` source annotation (226/226); proliferation markers MKI67, TOP2A and TYMS support an NK-cell state. |

These are explicit curated metadata decisions, not an automatic redefinition of
DL/RM or of any cell type. The original benchmark labels remain literal keys
in `celltype_subset` (Dataset 3) and `cell_type` (Dataset 4). Existing
`low_confidence`, `threshold_sensitive` and group-restriction flags are
retained pending wider taxonomy review.

Registered per-scenario counts now are:

| Dataset | GR-DL | GR-RM | LE-DL | LE-RM | SR-DL | SR-RM | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset 3 | 3 | 7 | 1 | 2 | 1 | 3 | 17 |
| Dataset 4 | 4 | 6 | 1 | 0 | 1 | 0 | 12 |

The **method outputs, integration embeddings, cluster assignments, true cell-type
labels, and per-cell-type recovery metrics are unaffected** by these changes.
Scenario-level summaries and per-scenario views, however, must be recomputed
or consistently re-aggregated when reusing older results. Old standalone HTML
reports do **not** automatically adopt the new registry. Their embedded
scenario assignments remain historical until explicitly regenerated or audited
as post-hoc corrections.

Because the dataset/evaluation contract hashes include ordered scenario labels,
post-change reports must not be mixed with pre-change runs as if they shared
the same evaluation contract. Preserve the original reports for reproducibility.
