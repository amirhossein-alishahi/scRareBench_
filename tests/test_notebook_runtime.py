from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import re
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "notebooks").glob("*.ipynb")))
DETAILED_NOTEBOOKS = tuple(p for p in NOTEBOOKS if "HighLevel" not in p.name)
HIGHLEVEL_NOTEBOOKS = tuple(p for p in NOTEBOOKS if "HighLevel" in p.name)
PERSIAN_RE = re.compile(r"[\u0600-\u06FF]")


def _project_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def _module():
    path = ROOT / "src" / "scrarebench" / "runtime.py"
    spec = spec_from_file_location("scrarebench_runtime", path)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_runtime_helper_preserves_scientific_anchors_contract(tmp_path):
    mod = _module()
    versions = {"numpy": "2.2.0", "scipy": "1.15.0", "pandas": "2.2.3"}
    path = tmp_path / "constraints.txt"
    mod._write_constraints(path, versions)
    text = path.read_text()
    assert "numpy==2.2.0" in text
    assert "scipy==1.15.0" in text
    assert "pandas==2.2.3" in text


def test_runtime_has_no_method_registry_or_legacy_method_setup():
    mod = _module()
    for name in (
        "METHOD_RUNTIME_PROFILES",
        "MethodRuntimeProfile",
        "available_methods",
        "get_method_profile",
        "setup_notebook",
        "install_notebook_runtime",
    ):
        assert not hasattr(mod, name)
    assert hasattr(mod, "setup_runtime")


def test_pyproject_has_no_method_specific_extras():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = "\n".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]
    assert "harmonypy" not in core
    assert "scvi-tools" not in core
    assert set(extras) == {"dev"}


def test_setup_runtime_accepts_arbitrary_scrarep_dependencies(monkeypatch):
    mod = _module()
    calls: list[list[str]] = []
    smoke: list[tuple[str, ...]] = []
    snapshots = iter([
        {"numpy": "2.2.0", "torch": "2.7.0"},
        {"numpy": "2.2.0", "torch": "2.7.0"},
    ])
    monkeypatch.setattr(mod, "_snapshot", lambda anchors: next(snapshots))
    monkeypatch.setattr(
        mod,
        "_base_requirements_from_installed_package",
        lambda: ("scanpy>=1.10", "scib-metrics==0.5.9"),
    )
    monkeypatch.setattr(mod, "_installed_version", lambda name: _project_version() if name == "scrarebench" else None)
    monkeypatch.setattr(mod, "_run", lambda cmd, quiet: calls.append(list(cmd)))
    monkeypatch.setattr(mod, "_fresh_process_smoke", lambda imports, quiet: smoke.append(tuple(imports)))
    monkeypatch.setattr(mod, "_pip_check_issues", lambda: ())

    report = mod.setup_runtime(
        extra_requirements=("scRareP==1.2.3", "some-custom-lib>=4"),
        extra_imports=("scrarep", "some_custom_lib"),
        quiet=False,
    )
    cmd = calls[0]
    assert "scRareP==1.2.3" in cmd
    assert "some-custom-lib>=4" in cmd
    assert "scanpy>=1.10" in cmd
    assert report.extra_requirements == ("scRareP==1.2.3", "some-custom-lib>=4")
    assert "scrarep" in smoke[0]
    assert "some_custom_lib" in smoke[0]


def test_setup_runtime_accepts_single_string_without_splitting(monkeypatch):
    mod = _module()
    calls: list[list[str]] = []
    smokes: list[tuple[str, ...]] = []
    snapshots = iter([{}, {}])
    monkeypatch.setattr(mod, "_snapshot", lambda anchors: next(snapshots))
    monkeypatch.setattr(mod, "_base_requirements_from_installed_package", lambda: ())
    monkeypatch.setattr(mod, "_installed_version", lambda name: _project_version() if name == "scrarebench" else None)
    monkeypatch.setattr(mod, "_run", lambda cmd, quiet: calls.append(list(cmd)))
    monkeypatch.setattr(mod, "_fresh_process_smoke", lambda imports, quiet: smokes.append(tuple(imports)))
    monkeypatch.setattr(mod, "_pip_check_issues", lambda: ())
    report = mod.setup_runtime(extra_requirements="scRareP==1.0", extra_imports="scrarep")
    assert report.extra_requirements == ("scRareP==1.0",)
    assert "scRareP==1.0" in calls[0]
    assert "scrarep" in smokes[0]


def test_setup_runtime_does_not_infer_dependency_from_method_name():
    mod = _module()
    with pytest.raises(TypeError):
        mod.setup_runtime(method="scRareP")


def test_runtime_install_command_has_no_unconditional_upgrade():
    text = (ROOT / "src" / "scrarebench" / "runtime.py").read_text()
    assert '"--upgrade",' not in text
    assert '"only-if-needed"' in text


def test_lightweight_runtime_import_works_before_scientific_imports():
    version = _project_version()
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import scrarebench; import scrarebench.runtime; "
        "print(scrarebench.__version__, hasattr(scrarebench, 'setup_runtime'))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == f"{version} True"


def test_release_notebooks_are_clean_english_pinned_and_runtime_check_optional():
    version = _project_version()
    assert len(NOTEBOOKS) == 8
    assert len(DETAILED_NOTEBOOKS) == 6
    assert len(HIGHLEVEL_NOTEBOOKS) == 2
    for path in NOTEBOOKS:
        assert "FIXED" not in path.name.upper()
        assert not re.search(r"_v\d", path.name, flags=re.IGNORECASE)
        nb = json.loads(path.read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        assert not PERSIAN_RE.search(text)
        assert "@main" not in text
        assert "GitHub main" not in text
        assert f"scRareBench_.git@v{version}" in text

        code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
        optional_cells = [
            cell for cell in code_cells
            if "OPTIONAL: validate this runtime" in "".join(cell.get("source", []))
        ]
        assert len(optional_cells) == 1
        optional_source = "".join(optional_cells[0].get("source", []))
        assert "_EXPECTED_COLAB_ANCHORS" in optional_source
        assert "Google Colab 2026.07" in optional_source
        assert "NOT required" in optional_source
        assert all(
            (not line.strip()) or line.lstrip().startswith("#")
            for line in optional_source.splitlines()
        )

        executable_code = "\n".join(
            "".join(cell.get("source", []))
            for cell in code_cells
            if cell is not optional_cells[0]
        )
        assert "setup_runtime(" in executable_code
        assert "setup_notebook(" not in executable_code
        assert "install_notebook_runtime(" not in executable_code
        assert "_EXPECTED_COLAB_ANCHORS" not in executable_code
        assert "CONSTRAINT_URL" not in executable_code
        assert "constraint_files=(CONSTRAINT_PATH,)" not in executable_code

        assert "not a hard requirement" in text or "not required" in text.lower()
        for i, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "code":
                compile("".join(cell.get("source", [])), f"{path}:{i}", "exec")


def test_highlevel_scvi_notebooks_keep_method_implementation_user_side():
    for path in HIGHLEVEL_NOTEBOOKS:
        code = _notebook_code(path)
        assert "from scrarebench.runtime import setup_runtime" in code
        assert "scvi-tools==1.4.3" in code
        assert 'extra_imports=("scvi",)' in code
        assert "scvi.model.SCVI" in code
        assert "benchmark_latent(" in code
        assert "scrarebench.methods" not in code
        assert "run_benchmark" not in code


def test_dataset2_reference_notebooks_use_registered_package_scenarios():
    paths = [
        ROOT / "notebooks" / "scRareBench_Harmony_Dataset2_mBDRC_Colab.ipynb",
        ROOT / "notebooks" / "scRareBench_MrVI_Dataset2_mBDRC_Colab.ipynb",
        ROOT / "notebooks" / "scRareBench_scVI_Dataset2_mBDRC_Colab.ipynb",
    ]
    for path in paths:
        code = _notebook_code(path)
        assert "scenario_table_from_adata" in code
        assert "infer_distribution_classes" not in code
        assert "build_distribution_only_scenarios" not in code
        assert "HTML_INCLUDE_RARE_SCENARIO_ANALYSIS = True" in code


def test_runtime_source_has_no_named_method_knowledge():
    text = (ROOT / "src" / "scrarebench" / "runtime.py").read_text().lower()
    for token in ("scvi", "mrvi", "harmony", "harmonypy"):
        assert token not in text


def test_no_method_implementation_or_legacy_runtime_shim_in_package():
    assert not (ROOT / "src" / "scrarebench" / "methods").exists()
    assert not (ROOT / "tools" / "notebook_runtime.py").exists()


def test_setup_runtime_fails_on_new_transitive_pip_conflict(monkeypatch):
    mod = _module()
    snapshots = iter([{"numpy": "2.0.2"}, {"numpy": "2.0.2"}])
    checks = iter([
        ("oldpkg 1.0 has requirement x<2, but you have x 2.0.",),
        (
            "oldpkg 1.0 has requirement x<2, but you have x 2.0.",
            "torchmetrics 1.0 has requirement lightning<3, but you have lightning 3.1.",
        ),
    ])
    monkeypatch.setattr(mod, "_snapshot", lambda anchors: next(snapshots))
    monkeypatch.setattr(mod, "_pip_check_issues", lambda: next(checks))
    monkeypatch.setattr(mod, "_base_requirements_from_installed_package", lambda: ("scanpy>=1.10",))
    monkeypatch.setattr(mod, "_installed_version", lambda name: _project_version() if name == "scrarebench" else None)
    monkeypatch.setattr(mod, "_run", lambda cmd, quiet: None)
    monkeypatch.setattr(mod, "_fresh_process_smoke", lambda imports, quiet: None)

    with pytest.raises(RuntimeError, match="torchmetrics"):
        mod.setup_runtime(extra_requirements="scRareP==1.0")


def test_setup_runtime_preserves_preexisting_pip_conflict_as_warning(monkeypatch):
    mod = _module()
    issue = "oldpkg 1.0 has requirement x<2, but you have x 2.0."
    snapshots = iter([{"numpy": "2.0.2"}, {"numpy": "2.0.2"}])
    checks = iter([(issue,), (issue,)])
    monkeypatch.setattr(mod, "_snapshot", lambda anchors: next(snapshots))
    monkeypatch.setattr(mod, "_pip_check_issues", lambda: next(checks))
    monkeypatch.setattr(mod, "_base_requirements_from_installed_package", lambda: ())
    monkeypatch.setattr(mod, "_installed_version", lambda name: _project_version() if name == "scrarebench" else None)
    monkeypatch.setattr(mod, "_run", lambda cmd, quiet: None)
    monkeypatch.setattr(mod, "_fresh_process_smoke", lambda imports, quiet: None)

    report = mod.setup_runtime()
    assert report.pip_check_warnings == (issue,)
    assert report.new_pip_check_issues == ()


def test_setup_runtime_adds_user_constraint_files(monkeypatch, tmp_path):
    mod = _module()
    constraint = tmp_path / "custom.txt"
    constraint.write_text("some-custom-lib==4.2\n")
    snapshots = iter([{}, {}])
    calls: list[list[str]] = []
    monkeypatch.setattr(mod, "_snapshot", lambda anchors: next(snapshots))
    monkeypatch.setattr(mod, "_pip_check_issues", lambda: ())
    monkeypatch.setattr(mod, "_base_requirements_from_installed_package", lambda: ())
    monkeypatch.setattr(mod, "_installed_version", lambda name: _project_version() if name == "scrarebench" else None)
    monkeypatch.setattr(mod, "_run", lambda cmd, quiet: calls.append(list(cmd)))
    monkeypatch.setattr(mod, "_fresh_process_smoke", lambda imports, quiet: None)

    report = mod.setup_runtime(constraint_files=constraint)
    cmd = calls[0]
    assert cmd.count("--constraint") == 2
    assert str(constraint.resolve()) in cmd
    assert report.user_constraint_files == (str(constraint.resolve()),)


def test_release_constraint_file_is_documented_anchor_set():
    path = ROOT / "constraints" / "colab-2026.07-anchors.txt"
    text = path.read_text()
    assert "numpy==2.0.2" in text
    assert "torch==2.11.0" in text
    assert "jax==0.7.2" in text
    assert "scib-metrics==0.5.9" in text
    assert "NOT a complete lockfile" in text


def test_release_version_is_coherent_across_package_citation_and_notebooks():
    version = _project_version()
    init_text = (ROOT / "src" / "scrarebench" / "__init__.py").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    checklist = (ROOT / "RELEASE_CHECKLIST.md").read_text()
    assert f'__version__ = "{version}"' in init_text
    assert f"version: {version}" in citation
    assert f"v{version}" in checklist
    for path in NOTEBOOKS:
        assert f"@v{version}" in path.read_text(encoding="utf-8")


def test_no_legacy_versioned_release_files():
    forbidden = re.compile(r"(?:v0[._-]?[0-9]|_v\d|FIXED)", re.IGNORECASE)
    for path in ROOT.rglob("*"):
        if any(part in {".git", ".pytest_cache", "__pycache__", "dist", "build"} for part in path.parts):
            continue
        if path.name == "pyproject.toml":
            continue
        assert not forbidden.search(path.name), f"Legacy versioned filename remains: {path.relative_to(ROOT)}"
