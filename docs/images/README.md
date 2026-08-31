# Documentation screenshots

This directory is reserved for screenshots used by the repository documentation. The screenshots are **documentation-only**: scRareBench, its benchmark logic, notebooks, reports, and Comparator do not depend on these files at runtime.

## Expected image files

Add the following files with these exact names:

- `report_overview.webp` — the main **Overview** page of a scRareBench interactive report, showing the report navigation and high-level benchmark summary.
- `report_rare_explorer.webp` — the **Rare-cell Explorer** view, showing per-cell-type rare-recovery diagnostics and the interactive rare-cell analysis workflow.
- `report_seed_stability.webp` — the native **multi-seed / Seed Stability** view, illustrating seed-aware inspection and stability analysis across repeated method runs.
- `comparator_methods_v10.webp` — **Multi-Report Comparator v10**, preferably the **Methods on Dataset** view after loading multiple compatible reports so visitors can immediately understand cross-method comparison.

## Where they are used

The root `README.md` and `INTERACTIVE_REPORTS.md` currently describe these screenshots with placeholders. After the files are uploaded here, replace the corresponding placeholder text with Markdown image references such as:

```markdown
![scRareBench interactive report overview](docs/images/report_overview.webp)
```

For links written from `INTERACTIVE_REPORTS.md` at the repository root, the same `docs/images/...` paths can be used.

## Recommended screenshot format

- Use WebP for compact GitHub rendering.
- Keep approximately a 5:3 aspect ratio; the prepared reference captures are 1600×960.
- Preserve readable metric labels, tab names, seed controls, and table headers.
- Avoid cropping away the report/comparator navigation because the goal is to show that the outputs are interactive rather than static figures.
- Do not include private paths, tokens, credentials, or unrelated browser chrome.

If an image filename changes, update every Markdown reference that points to it.
