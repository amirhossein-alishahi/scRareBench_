from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import subprocess
import sys
import tomllib


ROOT = Path(__file__).parents[1]
NOTEBOOKS = tuple(sorted((ROOT / "notebooks").glob("*.ipynb")))


def _module():
    path = ROOT / "src" / "scrarebench" / "runtime.py"
    spec = spec_from_file_location("scrarebench_runtime", path)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_runtime_helper_preserves_scientific_anchors_contract(tmp_path):
    mod = _module()
    versions = {"numpy": "2.2.0", "scipy": "1.15.0", "pandas": "2.2.3"}
    path = tmp_path / "constraints.txt"
    mod._write_constraints(path, versions)
    text = path.read_text()
    assert "numpy==2.2.0" in text
    assert "scipy==1.15.0" in text
    assert "pandas==2.2.3" in text


def test_method_profiles_are_registered_in_package():
    mod = _module()
    assert mod.available_methods() == ("scvi", "mrvi", "harmony")
    assert mod.get_method_profile("scvi").requirements == ("scvi-tools==1.4.3",)
    assert mod.get_method_profile("mrvi").requirements == ("scvi-tools==1.4.3",)
    assert mod.get_method_profile("harmonypy").requirements == ("harmonypy==2.0.0",)


def test_method_packages_are_optional_extras_not_core_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    core = "\n".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]
    assert "harmonypy" not in core
    assert "scvi-tools" not in core
    assert extras["harmony"] == ["harmonypy==2.0.0"]
    assert extras["scvi"] == ["scvi-tools==1.4.3"]
    assert extras["mrvi"] == ["scvi-tools==1.4.3"]


def test_runtime_install_command_has_no_unconditional_upgrade():
    text = (ROOT / "src" / "scrarebench" / "runtime.py").read_text()
    assert '"--upgrade",' not in text
    assert '"only-if-needed"' in text


def test_lightweight_runtime_import_works_before_scientific_imports():
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import scrarebench; import scrarebench.runtime; "
        "print(scrarebench.__version__)"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "0.9.2"


def test_all_colab_notebooks_are_github_first_and_delegate_runtime_setup():
    assert len(NOTEBOOKS) == 6
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text())
        text = path.read_text()
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "github.com/amirhossein-alishahi/scRareBench_.git@main" in code
        assert '"--no-deps"' in code
        assert "from scrarebench.runtime import print_install_report, setup_notebook" in code
        assert "setup_notebook(METHOD" in code
        assert "PROJECT_ZIP_PATH" not in text
        assert "PROJECT_ROOT" not in text
        assert "PROJECT_SRC" not in text
        assert "notebook_runtime.py" not in text
        assert '"--upgrade"' not in text


def test_notebooks_do_not_own_method_dependency_versions():
    for path in NOTEBOOKS:
        text = path.read_text()
        # prose may mention a library version for scientific reproducibility, but
        # installation requirements must not be hard-coded in code cells.
        nb = json.loads(text)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "scvi-tools==1.4.3" not in code
        assert "harmonypy==2.0.0" not in code
