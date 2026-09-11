from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
ADVANCED_ROOT = NOTEBOOK_ROOT / "advanced_configs"

version_text = (ROOT / "src" / "scrarebench" / "_version.py").read_text(encoding="utf-8")
match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_text)
if match is None:
    raise SystemExit("Could not resolve scRareBench version from _version.py")
version = match.group(1)
release_ref = f"scRareBench_.git@v{version}"

# The root notebooks are the two primary, user-facing Dataset 0 examples.
featured_names = {
    "scRareBench_Seurat_HighLevel_Dataset0_Colab.ipynb",
    "scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb",
}

# The previous configurable/release notebooks are intentionally retained under
# notebooks/advanced_configs/.
release_names = {
    "scRareBench_Harmony_Colab.ipynb",
    "scRareBench_Harmony_Dataset2_mBDRC_Colab.ipynb",
    "scRareBench_MrVI_Dataset0_GSE194122_Colab.ipynb",
    "scRareBench_MrVI_Dataset2_mBDRC_Colab.ipynb",
    "scRareBench_scVI_Colab.ipynb",
    "scRareBench_scVI_Dataset2_mBDRC_Colab.ipynb",
}
advanced_developer_names = {
    "scRareBench_CustomMethod_HighLevel_Colab.ipynb",
    "scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb",
    "scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb",
    "scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb",
}
advanced_names = release_names | advanced_developer_names

featured_notebooks = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
advanced_notebooks = sorted(ADVANCED_ROOT.glob("*.ipynb"))

observed_featured = {p.name for p in featured_notebooks}
if observed_featured != featured_names:
    missing = sorted(featured_names - observed_featured)
    extra = sorted(observed_featured - featured_names)
    raise SystemExit(f"Featured notebook set mismatch. Missing={missing}; extra={extra}")

observed_advanced = {p.name for p in advanced_notebooks}
if observed_advanced != advanced_names:
    missing = sorted(advanced_names - observed_advanced)
    extra = sorted(observed_advanced - advanced_names)
    raise SystemExit(f"Advanced notebook set mismatch. Missing={missing}; extra={extra}")

notebooks = featured_notebooks + advanced_notebooks

persian = re.compile(r"[\u0600-\u06FF]")
stale_name = re.compile(r"(?:_v\d|FIXED)", re.IGNORECASE)
semver_ref = re.compile(r"v\d+\.\d+\.\d+")
repo_pin = re.compile(r"scRareBench_\.git@(v\d+\.\d+\.\d+)")

cells = 0
for path in notebooks:
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    if persian.search(text):
        raise SystemExit(f"Non-English Persian text remains in notebook: {path}")

    if stale_name.search(path.name):
        raise SystemExit(f"Versioned/internal notebook filename: {path.name}")

    if path.parent == NOTEBOOK_ROOT:
        # Featured examples must be reproducibly pinned, but they are allowed to
        # demonstrate a previously released package version independently of the
        # current package patch version.
        pins = sorted(set(repo_pin.findall(text)))
        if not pins:
            raise SystemExit(f"Featured notebook is not pinned to a released scRareBench version: {path}")
        if len(pins) != 1:
            raise SystemExit(f"Featured notebook contains multiple scRareBench version pins: {path}: {pins}")
        if "@main" in text or "GitHub main" in text:
            raise SystemExit(f"Featured notebook uses/describes moving main branch: {path}")
        if "scrarebench.methods" in text:
            raise SystemExit(f"Featured notebook must remain method-agnostic: {path}")

    elif path.name in release_names:
        if "@main" in text or "GitHub main" in text:
            raise SystemExit(f"Release notebook uses/describes moving main branch: {path}")
        if release_ref not in text:
            raise SystemExit(f"Release notebook is not pinned to v{version}: {path}")
        stale_refs = sorted(set(semver_ref.findall(text)) - {f"v{version}"})
        if stale_refs:
            raise SystemExit(f"Stale release version reference(s) in {path}: {stale_refs}")
        if "_EXPECTED_COLAB_ANCHORS" not in text or "Google Colab runtime 2026.07" not in text:
            raise SystemExit(f"Validated Colab runtime preflight is missing: {path}")

    else:
        if release_ref not in text:
            raise SystemExit(f"Advanced developer notebook is not pinned to v{version}: {path}")
        stale_refs = sorted(set(semver_ref.findall(text)) - {f"v{version}"})
        if stale_refs:
            raise SystemExit(f"Stale release version reference(s) in {path}: {stale_refs}")
        if "scrarebench.methods" in text:
            raise SystemExit(f"Advanced developer notebook must remain method-agnostic: {path}")

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path}:{i}", "exec")
        cells += 1

print(
    f"Notebook validation passed: {len(featured_names)} featured + "
    f"{len(release_names)} release + {len(advanced_developer_names)} advanced developer "
    f"notebooks, {cells} code cells, package {version}"
)
