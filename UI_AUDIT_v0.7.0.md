# scRareBench v0.7.0 interactive dashboard UI audit

The v0.7.0 dashboard was reviewed as an end-to-end stateful report rather than as isolated figures.

## Shared Rare-cell Explorer state

The following controls now feed one shared filter state:

- Scenario
- Failure archetype
- Rare population name search
- Other-cell display mode for the Rare UMAP
- Focused rare population

The same filtered population set is used by:

- match count and matching-population chips
- rare-population details
- rare-cell metric table
- Rare UMAP
- recovery-quality heatmap
- within-type batch-NMI heatmap

Rare population search intentionally matches only `cell_type`; it no longer searches serialized metric/failure rows. A single search match auto-focuses its detail panel. A no-match state is explicit and does not leave stale plots visible.

## Rare scenario analysis

Scenario comparison and Rare Explorer filtering are intentionally distinct actions. Scenario cards and six scenario chips select the scenario-analysis table; the explicit **Apply ... to Rare Explorer** control transfers that scenario to the shared Rare filter state. Clicking a cell type from a scenario table/card focuses that population and synchronizes filters so it remains visible in Rare UMAP/heatmaps.

## Heatmap interpretation

Recovery metrics are shown together because higher is better:

- precision
- recall
- F1
- dominant-cluster capture (`inverse_purity` internally)

`within_type_batch_nmi` is shown separately because lower values mean less batch-associated cluster structure. Both heatmaps follow the exact same active Rare filters.

## Tables and export

- Every dashboard data table supports column sorting.
- Sort direction is tracked per selected column and surfaced with an arrow/ARIA state.
- `Export view` saves only currently visible rows.
- `Export all` saves the complete source table passed to that table; for the Rare population table this is the full curated rare-population table, not only the current filter result.
- Reproducibility text artifacts and scIB reference config remain directly downloadable.

## Figures and plots

- Plot exports are fail-safe: attempting to export an empty/unrendered plot produces a user message rather than an uncaught error.
- Static figures open in a high-resolution lightbox with fit-to-view, scrollable zoom, previous/next navigation, keyboard close/navigation, and original embedded-image download.
- Figure gallery supports title/filename search and figure-group filtering.

## Main UMAP and Sankey

- Main UMAP controls adapt when Sankey is excluded from the report.
- Highlight controls include select-all and clear-selection actions.
- Gray background traces skip expensive hover-string generation for better large-dataset responsiveness.
- Sankey flow-range maximum is derived from the current flow mode.
- Thresholds producing no links show an explicit empty state.
- Sankey-to-UMAP linking is conditional on the UMAP tab actually being included.

## Defensive behavior

- Tabs render lazily and rendering failures are caught per tab so one section cannot blank the entire report.
- Optional Rare subfeatures can be omitted independently from the HTML payload:
  - `include_rare_umap`
  - `include_rare_heatmaps`
  - `include_rare_scenario_analysis`
- Keyboard navigation and ARIA metadata were added to tabs and scenario selectors.

## Validation performed

- Python package compile check.
- Full pytest suite.
- Node.js syntax validation of the generated dashboard runtime.
- Dashboard generation using a compact synthetic benchmark result covering scIB, rare metrics, scenario metrics, figures, UMAP metadata and Sankey data.
- Dashboard generation using the previously produced real 89,199-cell scVI report payload, including 21 curated rare populations and all six rare scenarios.
- Both Colab notebooks parsed cell-by-cell with Python `ast` after the v0.7.0 API/flag update.

A full browser interaction harness was attempted in the build container, but Chromium navigation/execution is restricted by the container policy; therefore the automated validation relies on package tests, runtime JavaScript syntax validation, generated-payload checks, and defensive runtime guards rather than claiming browser-automation coverage that was not available in this environment.
