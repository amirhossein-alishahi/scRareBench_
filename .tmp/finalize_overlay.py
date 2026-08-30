from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import tarfile
from pathlib import Path

VERSION = "0.10.5"
ROOT = Path.cwd()

parts = [
    *[ROOT / f".tmp/scrare_overlay_xz/part{i:02d}" for i in range(9)],
    ROOT / ".tmp/scrare_overlay_xz/fix09_0",
    ROOT / ".tmp/scrare_overlay_xz/fix09_1",
    ROOT / ".tmp/scrare_overlay_xz/tail_0",
    ROOT / ".tmp/scrare_overlay_xz/tail_1",
    ROOT / ".tmp/scrare_overlay_xz/tail_2",
]
missing = [str(p) for p in parts if not p.is_file()]
if missing:
    raise RuntimeError(f"Missing staged overlay chunks: {missing}")
raw = base64.b64decode("".join(p.read_text(encoding="utf-8").strip() for p in parts), validate=True)
expected = "ed1e43612ed5df157b30d57972f64eb711421f19618a82127bd76a7a8832ca80"
observed = hashlib.sha256(raw).hexdigest()
print("overlay sha256:", observed)
if observed != expected:
    raise RuntimeError(f"Overlay SHA256 mismatch: {observed} != {expected}")

archive = Path("/tmp/scrarebench-overlay.tar.xz")
archive.write_bytes(raw)
extract = Path("/tmp/scrarebench-overlay")
shutil.rmtree(extract, ignore_errors=True)
extract.mkdir(parents=True)
with tarfile.open(archive, "r:xz") as tf:
    tf.extractall(extract, filter="data")
roots = [p for p in extract.iterdir() if p.is_dir()]
if len(roots) != 1:
    raise RuntimeError(f"Expected one overlay root, got: {roots}")
overlay = roots[0]
manifest = json.loads((overlay / "OVERLAY_MANIFEST.json").read_text(encoding="utf-8"))
entries = manifest["files"]
for entry in entries:
    rel = entry["path"]
    data = (overlay / rel).read_bytes()
    if len(data) != int(entry["size"]):
        raise RuntimeError(f"Size mismatch: {rel}")
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {rel}")
print("manifest verified files:", len(entries))

for entry in entries:
    rel = entry["path"]
    if rel in {"APPLY_NOTES.md", "OVERLAY_MANIFEST.json"}:
        continue
    src = overlay / rel
    dst = ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# Keep the standalone Comparator browser artifact optional in the core developer repo.
comparator = ROOT / "comparator/scRareBench_Multi_Report_Comparator_v9.html"
if not comparator.exists():
    p = ROOT / "tests/test_multiseed_v0101.py"
    text = p.read_text(encoding="utf-8")
    target = "def test_comparator_v8_preserves_first_class_seed_and_context_metric_guard():"
    if target in text and "Standalone Comparator v9 is not shipped" not in text:
        text = text.replace(
            target,
            '@pytest.mark.skip(reason="Standalone Comparator v9 is not shipped in the core developer repository")\n' + target,
            1,
        )
        p.write_text(text, encoding="utf-8")

# Release metadata.
p = ROOT / "src/scrarebench/_version.py"
p.write_text(p.read_text(encoding="utf-8").replace('__version__ = "0.10.4"', f'__version__ = "{VERSION}"'), encoding="utf-8")
p = ROOT / "pyproject.toml"
p.write_text(p.read_text(encoding="utf-8").replace('version = "0.10.4"', f'version = "{VERSION}"'), encoding="utf-8")
p = ROOT / "CITATION.cff"
t = p.read_text(encoding="utf-8").replace("version: 0.10.4", f"version: {VERSION}").replace("date-released: 2026-08-26", "date-released: 2026-08-30")
p.write_text(t, encoding="utf-8")

p = ROOT / "README.md"
t = p.read_text(encoding="utf-8")
t = t.replace("release-0.10.4-blue)](#what-is-new-in-0104)", "release-0.10.5-blue)](#what-is-new-in-0105)")
t = t.replace("**Current release: `0.10.4`**", "**Current release: `0.10.5`**")
t = t.replace("pip install dist/scrarebench-0.10.4-py3-none-any.whl", "pip install dist/scrarebench-0.10.5-py3-none-any.whl")
if "## What is new in 0.10.5" not in t:
    marker = "## What is new in 0.10.4"
    note = (
        "## What is new in 0.10.5\n\n"
        "0.10.5 integrates the method-agnostic high-level developer API with the multi-seed, provenance-hardened rare-cell benchmark. "
        "It adds generic `MethodSpec`/`benchmark_method()` orchestration, user-controlled method dependencies/installers, high-level Colab templates, "
        "and retains the low-level latent/evaluation APIs for full control.\n\n"
    )
    if marker not in t:
        raise RuntimeError("README release-history marker is missing")
    t = t.replace(marker, note + marker, 1)
p.write_text(t, encoding="utf-8")

p = ROOT / "DESIGN_NOTES.md"
p.write_text(p.read_text(encoding="utf-8").replace("# scRareBench 0.10.4 design decisions", "# scRareBench 0.10.5 design decisions"), encoding="utf-8")

p = ROOT / "CHANGELOG.md"
t = p.read_text(encoding="utf-8")
if "## 0.10.5" not in t:
    section = (
        "## 0.10.5 — method-agnostic high-level API and integrated multi-seed benchmark\n\n"
        "- Added generic `MethodSpec`, `MethodOutput`, and `benchmark_method()` orchestration.\n"
        "- Kept method implementation and dependencies user-controlled; no method registry is embedded.\n"
        "- Integrated multi-seed evaluation, rare-aware metric registry, support-adjusted local recovery, and provenance hardening.\n"
        "- Added high-level and low-level Colab templates and pinned all shipped notebooks to v0.10.5.\n\n"
    )
    pos = t.find("## ")
    t = (t[:pos] + section + t[pos:]) if pos >= 0 else t + "\n" + section
p.write_text(t, encoding="utf-8")

# Make the provenance fixture import independent of pytest import mode.
p = ROOT / "tests/test_provenance_hardening_v0104.py"
t = p.read_text(encoding="utf-8")
old = 'sys.path.insert(0, str(Path(__file__).parent))\nfrom test_delivery_v0104 import MiniAdata, MiniResult, _make_reports  # noqa: E402\n'
new = '''_fixture_path = Path(__file__).with_name("test_delivery_v0104.py")\n_fixture_spec = importlib.util.spec_from_file_location("scrarebench_delivery_v0104_fixtures", _fixture_path)\nif _fixture_spec is None or _fixture_spec.loader is None:\n    raise RuntimeError(f"Could not load test fixtures from {_fixture_path}")\n_fixture_module = importlib.util.module_from_spec(_fixture_spec)\n_fixture_spec.loader.exec_module(_fixture_module)\nMiniAdata = _fixture_module.MiniAdata\nMiniResult = _fixture_module.MiniResult\n_make_reports = _fixture_module._make_reports\n'''
if old not in t:
    raise RuntimeError("Provenance sibling import block is missing")
t = t.replace(old, new).replace('"scrarebench_version": "0.10.4"', '"scrarebench_version": "0.10.5"')
p.write_text(t, encoding="utf-8")

# Current release contract should validate 0.10.5. Filename remains historical for continuity.
p = ROOT / "tests/test_release_contract_v0104.py"
p.write_text(p.read_text(encoding="utf-8").replace("0.10.4", "0.10.5"), encoding="utf-8")

# Pin every shipped notebook to the release tag, including developer-oriented notebooks.
for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            cell["source"] = [
                line.replace("scRareBench_.git@main", "scRareBench_.git@v0.10.5")
                    .replace("scRareBench_.git@v0.10.4", "scRareBench_.git@v0.10.5")
                    .replace('EXPECTED_SCRAREBENCH_VERSION = "0.10.4"', 'EXPECTED_SCRAREBENCH_VERSION = "0.10.5"')
                    .replace("v0_10_4", "v0_10_5")
                for line in src
            ]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

# Developer notebooks are release artifacts now: validator must require the tag, not moving main.
p = ROOT / "scripts/check_notebooks.py"
t = p.read_text(encoding="utf-8")
old = '''    else:\n        if main_ref not in text:\n            raise SystemExit(f"Developer notebook must install the integrated main branch: {path}")\n        if "scrarebench.methods" in text:\n            raise SystemExit(f"Developer notebook must remain method-agnostic: {path}")\n'''
new = '''    else:\n        if release_ref not in text:\n            raise SystemExit(f"Developer notebook is not pinned to v{version}: {path}")\n        stale_refs = sorted(set(semver_ref.findall(text)) - {f"v{version}"})\n        if stale_refs:\n            raise SystemExit(f"Stale release version reference(s) in {path}: {stale_refs}")\n        if "scrarebench.methods" in text:\n            raise SystemExit(f"Developer notebook must remain method-agnostic: {path}")\n'''
if old not in t:
    raise RuntimeError("Notebook validator developer block is missing")
p.write_text(t.replace(old, new), encoding="utf-8")

print("Final overlay prepared for release", VERSION)
