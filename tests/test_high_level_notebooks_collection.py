"""Static, non-training regression checks for the complete high-level notebook collection.

Keeping this separate from method execution avoids downloading ~100k-cell data in CI.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks" / "high_level"
SHA = "cfdb720f5488b9ceef9a225b35fc7558bcb0251c"
DATASETS = (0, 2, 3, 4, 5)
METHODS = ("Harmony", "MrVI", "scDisInFact", "scDML", "scVI", "Seurat")
EXPECTED = {(m, d) for m in METHODS if m != "Seurat" for d in DATASETS} | {
    ("Seurat", 0), ("Seurat", 2),
}
CANON = re.compile(r"^scRareBench_HighLevel_(Harmony|MrVI|scDisInFact|scDML|scVI|Seurat)_Dataset(0|2|3|4|5)_Colab\.ipynb$")


def _source(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else str(src)


def test_manifest_matches_exact_high_level_matrix_and_hashes():
    manifest = json.loads((BASE / "NOTEBOOK_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["benchmark_package_git_revision"] == SHA
    rows = manifest["notebooks"]
    actual = {(r["method"], r["dataset"]) for r in rows}
    assert len(rows) == 27
    assert actual == EXPECTED
    assert len(actual) == len(rows)
    paths = {r["output"] for r in rows}
    expected_paths = {f"notebooks/high_level/scRareBench_HighLevel_{m}_Dataset{d}_Colab.ipynb" for m, d in EXPECTED}
    assert paths == expected_paths
    assert {str(p.relative_to(ROOT)) for p in BASE.glob("*.ipynb")} == paths
    for row in rows:
        p = ROOT / row["output"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"]
        assert p.stat().st_size == row["size"]


@pytest.mark.parametrize("method,dataset", sorted(EXPECTED))
def test_notebook_is_structurally_valid_and_safely_pinned(method, dataset):
    p = BASE / f"scRareBench_HighLevel_{method}_Dataset{dataset}_Colab.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert nb["cells"]
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert f"Dataset {dataset}" in _source(nb["cells"][0])
    assert method in _source(nb["cells"][0])
    assert nb["metadata"]["colab"]["name"] == p.name
    codes = []
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        source = _source(cell)
        assert source.strip()
        ast.parse(source, filename=str(p))
        codes.append(source)
    combined = "\n".join(codes)
    assert SHA in combined  # never accidentally install old registered scenarios
    assert "scRareBench_.git@v0.10.5" not in combined
    assert "scRareBench_.git@v0.10.6" not in combined
    assert "12c4e287f0367f2dc67945b45e5e79e707977f73" not in combined
    assert re.search(r"\bload_dataset\s*\(\s*" + str(dataset) + r"\s*\)", combined)
    assert "MethodSpec(" in combined
    assert "benchmark_method(" in combined
    assert "METHOD_SEEDS" in combined
    assert "BENCHMARK_SEED" in combined


def test_companion_seurat_scripts_are_parseable_and_log_free():
    root = ROOT / "scripts" / "seurat_server"
    for d in (0, 3, 4, 5):
        folder = root / f"dataset{d}"
        program = folder / f"run_seurat_dataset{d}.py"
        assert program.is_file()
        ast.parse(program.read_text(encoding="utf-8"), filename=str(program))
        assert (folder / "generated_seurat_reference.R").is_file()
        assert not list(folder.rglob("*.log"))
        assert not list(folder.rglob("*.backup"))
    assert (root / "README.md").is_file()


def test_no_large_execution_artifacts_embedded_in_notebooks():
    # An accidental committed output can inflate every notebook by hundreds of KB.
    for p in BASE.glob("*.ipynb"):
        nb = json.loads(p.read_text(encoding="utf-8"))
        assert all(c.get("execution_count") is None and not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
