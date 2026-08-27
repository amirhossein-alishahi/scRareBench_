from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
version = project["project"]["version"]
release_ref = f"scRareBench_.git@v{version}"

notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
if len(notebooks) != 8:
    raise SystemExit(f"Expected 8 release notebooks, found {len(notebooks)}")

persian = re.compile(r"[\u0600-\u06FF]")
stale_name = re.compile(r"(?:_v\d|FIXED)", re.IGNORECASE)
semver_ref = re.compile(r"v\d+\.\d+\.\d+")

cells = 0
for path in notebooks:
    if stale_name.search(path.name):
        raise SystemExit(f"Versioned/internal notebook filename in release: {path.name}")

    nb = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    if persian.search(text):
        raise SystemExit(f"Non-English Persian text remains in release notebook: {path}")
    if "@main" in text or "GitHub main" in text:
        raise SystemExit(f"Release notebook uses/describes moving main branch: {path}")
    if release_ref not in text:
        raise SystemExit(f"Release notebook is not pinned to v{version}: {path}")
    stale_refs = sorted(set(semver_ref.findall(text)) - {f"v{version}"})
    if stale_refs:
        raise SystemExit(f"Stale release version reference(s) in {path}: {stale_refs}")
    if "_EXPECTED_COLAB_ANCHORS" not in text or "Google Colab runtime 2026.07" not in text:
        raise SystemExit(f"Validated Colab runtime preflight is missing: {path}")

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path}:{i}", "exec")
        cells += 1

print(f"Notebook validation passed: {len(notebooks)} notebooks, {cells} code cells, release v{version}")
