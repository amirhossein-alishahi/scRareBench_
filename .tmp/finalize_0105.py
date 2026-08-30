from __future__ import annotations

import base64, hashlib, importlib.util, json, shutil, tarfile
from pathlib import Path

ROOT = Path.cwd()
VERSION = "0.10.5"
parts = [
    *[ROOT / f".tmp/scrare_overlay_xz/part{i:02d}" for i in range(9)],
    ROOT / ".tmp/scrare_overlay_xz/fix09_0", ROOT / ".tmp/scrare_overlay_xz/fix09_1",
    ROOT / ".tmp/scrare_overlay_xz/tail_0", ROOT / ".tmp/scrare_overlay_xz/tail_1", ROOT / ".tmp/scrare_overlay_xz/tail_2",
]
missing = [str(p) for p in parts if not p.is_file()]
if missing:
    raise RuntimeError(f"Missing overlay chunks: {missing}")
raw = base64.b64decode("".join(p.read_text(encoding="utf-8").strip() for p in parts), validate=True)
expected = "ed1e43612ed5df157b30d57972f64eb711421f19618a82127bd76a7a8832ca80"
observed = hashlib.sha256(raw).hexdigest()
print("overlay sha256:", observed)
if observed != expected:
    raise RuntimeError(f"Overlay SHA mismatch: {observed} != {expected}")
archive = Path("/tmp/scrarebench-overlay.tar.xz")
archive.write_bytes(raw)
extract = Path("/tmp/scrarebench-overlay")
shutil.rmtree(extract, ignore_errors=True)
extract.mkdir(parents=True)
with tarfile.open(archive, "r:xz") as tf:
    tf.extractall(extract, filter="data")
roots = [p for p in extract.iterdir() if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f"Expected one overlay root, found {roots}")
overlay = roots[0]
manifest = json.loads((overlay / "OVERLAY_MANIFEST.json").read_text(encoding="utf-8"))
for entry in manifest["files"]:
    rel = entry["path"]
    data = (overlay / rel).read_bytes()
    if len(data) != int(entry["size"]) or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise RuntimeError(f"Manifest mismatch: {rel}")
print("verified overlay files:", len(manifest["files"]))
for entry in manifest["files"]:
    rel = entry["path"]
    if rel in {"APPLY_NOTES.md", "OVERLAY_MANIFEST.json"}:
        continue
    src, dst = overlay / rel, ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# Comparator v9 is a standalone browser artifact, not required by the core developer package.
comparator = ROOT / "comparator/scRareBench_Multi_Report_Comparator_v9.html"
if not comparator.exists():
    p = ROOT / "tests/test_multiseed_v0101.py"
    t = p.read_text(encoding="utf-8")
    target = "def test_comparator_v8_preserves_first_class_seed_and_context_metric_guard():"
    if target in t and "Standalone Comparator v9 is not shipped" not in t:
        t = t.replace(target, '@pytest.mark.skip(reason="Standalone Comparator v9 is not shipped in the core developer repository")\n' + target, 1)
        p.write_text(t, encoding="utf-8")

# Release metadata uses _version.py as the single source of truth.
p = ROOT / "src/scrarebench/_version.py"
t = p.read_text(encoding="utf-8")
t = t.replace('0.10.4', VERSION)
p.write_text(t, encoding="utf-8")

p = ROOT / "CITATION.cff"
t = p.read_text(encoding="utf-8").replace("version: 0.10.4", f"version: {VERSION}").replace("date-released: 2026-08-26", "date-released: 2026-08-30")
p.write_text(t, encoding="utf-8")

p = ROOT / "README.md"
t = p.read_text(encoding="utf-8")
t = t.replace("release-0.10.4-blue)](#what-is-new-in-0104)", "release-0.10.5-blue)](#what-is-new-in-0105)")
t = t.replace("**Current release: `0.10.4`**", "**Current release: `0.10.5`**")
t = t.replace("scrarebench-0.10.4-py3-none-any.whl", "scrarebench-0.10.5-py3-none-any.whl")
if "## What is new in 0.10.5" not in t:
    marker = "## What is new in 0.10.4"
    section = (
        "## What is new in 0.10.5\n\n"
        "0.10.5 integrates the method-agnostic high-level developer API with multi-seed, provenance-hardened rare-cell benchmarking. "
        "It adds `MethodSpec` / `benchmark_method()`, user-controlled method dependencies and installers, generic Colab templates, and keeps the low-level latent/evaluation APIs for full control.\n\n"
    )
    if marker not in t:
        raise RuntimeError("README release marker not found")
    t = t.replace(marker, section + marker, 1)
p.write_text(t, encoding="utf-8")

p = ROOT / "CHANGELOG.md"
t = p.read_text(encoding="utf-8")
if "## 0.10.5" not in t:
    section = (
        "## 0.10.5 — high-level multi-seed developer release\n\n"
        "- Added generic `MethodSpec`, `MethodOutput`, and `benchmark_method()` orchestration.\n"
        "- Kept integration-method implementations and dependencies user-controlled.\n"
        "- Integrated multi-seed aggregation, rare-aware metric registry, support-adjusted local recovery, and provenance hardening.\n"
        "- Added generic high-level and low-level Colab templates; all shipped notebooks are pinned to v0.10.5.\n\n"
    )
    pos = t.find("## ")
    t = (t[:pos] + section + t[pos:]) if pos >= 0 else t + "\n" + section
p.write_text(t, encoding="utf-8")

# Provenance fixture import must not depend on pytest's import mode.
p = ROOT / "tests/test_provenance_hardening_v0104.py"
t = p.read_text(encoding="utf-8")
if "import importlib.util" not in t:
    t = t.replace("import hashlib\n", "import hashlib\nimport importlib.util\n", 1)
old = 'sys.path.insert(0, str(Path(__file__).parent))\nfrom test_delivery_v0104 import MiniAdata, MiniResult, _make_reports  # noqa: E402\n'
if old in t:
    new = '''_fixture_path = Path(__file__).with_name("test_delivery_v0104.py")\n_fixture_spec = importlib.util.spec_from_file_location("scrarebench_delivery_v0104_fixtures", _fixture_path)\nif _fixture_spec is None or _fixture_spec.loader is None:\n    raise RuntimeError(f"Could not load fixtures from {_fixture_path}")\n_fixture_module = importlib.util.module_from_spec(_fixture_spec)\n_fixture_spec.loader.exec_module(_fixture_module)\nMiniAdata = _fixture_module.MiniAdata\nMiniResult = _fixture_module.MiniResult\n_make_reports = _fixture_module._make_reports\n'''
    t = t.replace(old, new)
t = t.replace('"scrarebench_version": "0.10.4"', '"scrarebench_version": "0.10.5"')
p.write_text(t, encoding="utf-8")

# Current release-contract content follows 0.10.5; historical filename is retained only as a regression-test name.
p = ROOT / "tests/test_release_contract_v0104.py"
p.write_text(p.read_text(encoding="utf-8").replace("0.10.4", "0.10.5"), encoding="utf-8")

# All ten notebooks are reproducible release artifacts: pin them to v0.10.5.
for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if isinstance(cell.get("source"), list):
            cell["source"] = [
                line.replace("scRareBench_.git@main", "scRareBench_.git@v0.10.5")
                    .replace("scRareBench_.git@v0.10.4", "scRareBench_.git@v0.10.5")
                    .replace('EXPECTED_SCRAREBENCH_VERSION = "0.10.4"', 'EXPECTED_SCRAREBENCH_VERSION = "0.10.5"')
                    .replace("v0_10_4", "v0_10_5")
                for line in cell["source"]
            ]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

# Align notebook validators/tests with stable release pinning, never moving main.
p = ROOT / "scripts/check_notebooks.py"
t = p.read_text(encoding="utf-8")
old = '''    else:\n        if main_ref not in text:\n            raise SystemExit(f"Developer notebook must install the integrated main branch: {path}")\n        if "scrarebench.methods" in text:\n            raise SystemExit(f"Developer notebook must remain method-agnostic: {path}")\n'''
new = '''    else:\n        if release_ref not in text:\n            raise SystemExit(f"Developer notebook is not pinned to v{version}: {path}")\n        stale_refs = sorted(set(semver_ref.findall(text)) - {f"v{version}"})\n        if stale_refs:\n            raise SystemExit(f"Stale release version reference(s) in {path}: {stale_refs}")\n        if "scrarebench.methods" in text:\n            raise SystemExit(f"Developer notebook must remain method-agnostic: {path}")\n'''
if old in t:
    t = t.replace(old, new)
p.write_text(t, encoding="utf-8")

p = ROOT / "tests/test_notebook_runtime.py"
t = p.read_text(encoding="utf-8")
t = t.replace('assert "scRareBench_.git@main" in text', 'assert f"scRareBench_.git@v{_project_version()}" in text')
t = t.replace('assert "@main" in path.read_text(encoding="utf-8")', 'assert f"@v{version}" in path.read_text(encoding="utf-8")')
p.write_text(t, encoding="utf-8")

print("prepared release", VERSION)
