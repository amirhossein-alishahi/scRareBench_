from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
version_text = (ROOT / "src" / "scrarebench" / "_version.py").read_text(encoding="utf-8")
match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', version_text)
if match is None:
    raise SystemExit("Could not resolve scRareBench version from _version.py")
version = match.group(1)
release_ref = f"scRareBench_.git@v{version}"
main_ref = "scRareBench_.git@main"

release_names = {
    "scRareBench_Harmony_Colab.ipynb",
    "scRareBench_Harmony_Dataset2_mBDRC_Colab.ipynb",
    "scRareBench_MrVI_Dataset0_GSE194122_Colab.ipynb",
    "scRareBench_MrVI_Dataset2_mBDRC_Colab.ipynb",
    "scRareBench_scVI_Colab.ipynb",
    "scRareBench_scVI_Dataset2_mBDRC_Colab.ipynb",
}
developer_names = {
    "scRareBench_CustomMethod_HighLevel_Colab.ipynb",
    "scRareBench_MultiSeed_LowLevel_Template_Colab.ipynb",
    "scRareBench_scVI_HighLevel_Dataset0_Colab.ipynb",
    "scRareBench_scVI_HighLevel_Dataset2_mBDRC_Colab.ipynb",
}

notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
observed = {p.name for p in notebooks}
expected = release_names | developer_names
if observed != expected:
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    raise SystemExit(f"Notebook set mismatch. Missing={missing}; extra={extra}")

persian = re.compile(r"[\u0600-\u06FF]")
stale_name = re.compile(r"(?:_v\d|FIXED)", re.IGNORECASE)
semver_ref = re.compile(r"v\d+\.\d+\.\d+")

cells = 0
for path in notebooks:
    nb = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")

    if persian.search(text):
        raise SystemExit(f"Non-English Persian text remains in notebook: {path}")

    if path.name in release_names:
        if stale_name.search(path.name):
            raise SystemExit(f"Versioned/internal notebook filename in release: {path.name}")
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
        if main_ref not in text:
            raise SystemExit(f"Developer notebook must install the integrated main branch: {path}")
        if "scrarebench.methods" in text:
            raise SystemExit(f"Developer notebook must remain method-agnostic: {path}")

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        compile(source, f"{path}:{i}", "exec")
        cells += 1

print(
    f"Notebook validation passed: {len(release_names)} release + "
    f"{len(developer_names)} developer notebooks, {cells} code cells, package {version}"
)
