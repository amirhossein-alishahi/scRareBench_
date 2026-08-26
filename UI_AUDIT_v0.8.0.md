# scRareBench v0.8.0 dashboard validation

This release closes the issues identified in the v0.7.0 interaction audit.

Validated fixes:
- Main UMAP click-to-focus selection and selection details.
- Rare-cell detail → Main UMAP state synchronization.
- Sankey link → already-rendered UMAP refresh.
- Rare outcome (Preserved / Not preserved) separated from failure mode.
- Scenario outcome stack click applies scenario and failure-mode filters.
- Rare recovery and batch-NMI heatmaps select the clicked rare population.
- Sankey numeric threshold and slider clamp to the same value.
- Figure Previous/Next navigation respects active gallery filters.
- Lightbox has Fit and actual 100% controls.
- Very small numeric metrics use scientific notation.
- Overview separates scIB aggregate visualization from rare-cell summary visualization.

Validation performed:
- Full Python test suite: 30/30 passed.
- Generated dashboard JavaScript checked with Node.
- Playwright/Chromium interaction smoke test passed for Rare search, heatmap click, Rare→UMAP, Main UMAP click, Sankey threshold, Sankey→UMAP, outcome/failure filters, and figure navigation/zoom, with zero captured browser page/console errors.
- The prior 89,199-cell / 21-rare-population payload was regenerated successfully with the v0.8.0 template and its runtime JavaScript syntax-checked.
