from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_fixture_path = Path(__file__).with_name("test_delivery_v0104.py")
_fixture_spec = importlib.util.spec_from_file_location("scrarebench_delivery_v0104_fixtures", _fixture_path)
if _fixture_spec is None or _fixture_spec.loader is None:
    raise RuntimeError(f"Could not load fixtures from {_fixture_path}")
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
MiniAdata = _fixture_module.MiniAdata
MiniResult = _fixture_module.MiniResult
_make_reports = _fixture_module._make_reports

from scrarebench import (  # noqa: E402
    create_report_bundle,
    dataset_contract_hash,
    evaluation_contract_hash,
    finalize_multiseed_delivery,
    method_training_hash,
    validate_multiseed_report,
    write_interactive_report,
)
from scrarebench.multiseed import configuration_hash, extract_embedded_report_payload, make_multirun_container  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _strong_statuses(reports, bundles, latents, out_dir: Path):
    out = {}
    for report in reports:
        validation = validate_multiseed_report(report)
        ident = validation["payloads"][0]
        seed = ident["method_seed"]
        latent = np.load(latents[seed], allow_pickle=False)
        payload = {
            "status": "complete",
            "scrarebench_version": "0.10.5",
            "method": ident["method_name"],
            "dataset": ident["dataset_key"],
            "method_seed": seed,
            "benchmark_seed": ident["benchmark_seed"],
            "method_training_hash": ident["method_training_hash"],
            "evaluation_contract_hash": ident["evaluation_contract_hash"],
            "configuration_hash": ident["configuration_hash"],
            "run_id": ident["run_id"],
            "dataset_fingerprint": ident["dataset_fingerprint"],
            "dataset_contract_sha256": ident["dataset_contract_sha256"],
            "cell_order_sha256": ident["cell_order_sha256"],
            "latent_array_sha256": _array_sha256(latent),
            "latent_file_sha256": _sha256_file(latents[seed]),
            "report_sha256": _sha256_file(report),
            "bundle_sha256": _sha256_file(bundles[seed]),
        }
        path = out_dir / f"strong_status_{seed}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        out[seed] = path
    return out


def test_benchmark_seed_is_evaluation_invariant_and_method_seed_is_not():
    method_a = {"seed": 42, "n_latent": 30}
    method_b = {"seed": 123, "n_latent": 30}
    assert method_training_hash(method_a) == method_training_hash(method_b)
    a = {"random_state": 42, "n_neighbors": 15, "reference_resolution": 1.0}
    b = {"random_state": 123, "n_neighbors": 15, "reference_resolution": 1.0}
    assert evaluation_contract_hash(a, method_a) != evaluation_contract_hash(b, method_b)


def test_conflicting_benchmark_seed_aliases_fail_closed():
    with pytest.raises(ValueError, match="benchmark_seed.*differs from random_state"):
        evaluation_contract_hash({"benchmark_seed": 42, "random_state": 123, "n_neighbors": 15}, {"seed": 42})


def test_nonfinite_configuration_values_fail_closed_with_path():
    with pytest.raises(ValueError, match=r"non-finite.*\$\.layers\[1\]"):
        configuration_hash({"layers": [32.0, np.nan]})
    with pytest.raises(ValueError, match=r"non-finite"):
        configuration_hash({"x": np.inf})


def test_dataset_contract_detects_annotation_and_feature_changes():
    adata = MiniAdata()
    base = dataset_contract_hash(
        adata, dataset_key="demo", label_key="celltype", batch_key="BATCH", scenario_key="scrarebench_scenario"
    )
    changed_label = MiniAdata(); changed_label.obs.loc["c0", "celltype"] = "Z"
    assert dataset_contract_hash(
        changed_label, dataset_key="demo", label_key="celltype", batch_key="BATCH", scenario_key="scrarebench_scenario"
    ) != base
    changed_feature = MiniAdata(); changed_feature.var_names = changed_feature.var_names[::-1]
    assert dataset_contract_hash(
        changed_feature, dataset_key="demo", label_key="celltype", batch_key="BATCH", scenario_key="scrarebench_scenario"
    ) != base


def test_finalizer_rejects_wrong_latent_and_wrong_bundle_even_when_internally_valid(tmp_path: Path):
    reports, bundles, statuses, latents = _make_reports(tmp_path)
    # First prove the normal handoff is valid.
    finalize_multiseed_delivery(
        reports, tmp_path / "ok", method_name="Demo", dataset_key="demo",
        expected_seeds=[42, 123, 2026], bundles_by_seed=bundles,
        status_files_by_seed=statuses, latent_files_by_seed=latents, include_latents=True,
    )

    wrong_latents = dict(latents); wrong_latents[42] = latents[123]
    with pytest.raises(ValueError, match="Latent identity mismatch"):
        finalize_multiseed_delivery(
            reports, tmp_path / "wrong_latent", method_name="Demo", dataset_key="demo",
            expected_seeds=[42, 123, 2026], bundles_by_seed=bundles,
            status_files_by_seed=statuses, latent_files_by_seed=wrong_latents, include_latents=True,
        )

    adata = MiniAdata(); result = MiniResult(tmp_path / "alt_result")
    adata.obsm["X_demo"] = np.arange(18, dtype=np.float32).reshape(6, 3) + 99
    alt_report = tmp_path / "alt_seed42.html"
    write_interactive_report(
        adata, result, alt_report, representation_key="X_demo", umap_key="X_umap_demo",
        method_seed=42, method_config={"seed": 42, "n_latent": 99}, expected_seeds=[42],
    )
    alt_bundle = tmp_path / "alt_seed42.zip"
    create_report_bundle(
        adata, result, alt_bundle, representation_key="X_demo", include_latent=True,
        write_interactive=True, write_pdf=False, existing_interactive_report=alt_report,
        method_seed=42, method_config={"seed": 42, "n_latent": 99}, expected_seeds=[42],
    )
    wrong_bundles = dict(bundles); wrong_bundles[42] = alt_bundle
    with pytest.raises(ValueError, match="identity mismatch"):
        finalize_multiseed_delivery(
            reports, tmp_path / "wrong_bundle", method_name="Demo", dataset_key="demo",
            expected_seeds=[42, 123, 2026], bundles_by_seed=wrong_bundles,
            status_files_by_seed=statuses, latent_files_by_seed=latents, include_latents=True,
        )


def test_strong_status_is_bound_to_report_bundle_and_latent(tmp_path: Path):
    reports, bundles, _, latents = _make_reports(tmp_path)
    statuses = _strong_statuses(reports, bundles, latents, tmp_path)
    result = finalize_multiseed_delivery(
        reports, tmp_path / "strong_ok", method_name="Demo", dataset_key="demo",
        expected_seeds=[42, 123, 2026], bundles_by_seed=bundles,
        status_files_by_seed=statuses, latent_files_by_seed=latents, include_latents=True,
    )
    assert result.archive_path.exists()
    bad = json.loads(statuses[42].read_text(encoding="utf-8")); bad["run_id"] = "wrong"
    statuses[42].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="run_id"):
        finalize_multiseed_delivery(
            reports, tmp_path / "strong_bad", method_name="Demo", dataset_key="demo",
            expected_seeds=[42, 123, 2026], bundles_by_seed=bundles,
            status_files_by_seed=statuses, latent_files_by_seed=latents, include_latents=True,
        )


def test_cross_seed_missing_scientific_row_is_rejected(tmp_path: Path):
    reports, _, _, _ = _make_reports(tmp_path)
    runs = [extract_embedded_report_payload(p)["runs"][0] for p in reports]
    container = make_multirun_container(runs, expected_seeds=[42, 123, 2026])
    damaged = copy.deepcopy(container)
    table = damaged["runs"][1]["payload"]["rare"]["per_type"]
    if isinstance(table, dict):
        table["rows"] = table["rows"][:-1]
    else:
        damaged["runs"][1]["payload"]["rare"]["per_type"] = table[:-1]
    with pytest.raises(ValueError, match="Cross-seed row identity mismatch"):
        validate_multiseed_report(damaged, expected_seeds=[42, 123, 2026])


def test_stored_aggregate_must_equal_recomputed_aggregate(tmp_path: Path):
    reports, _, _, _ = _make_reports(tmp_path)
    runs = [extract_embedded_report_payload(p)["runs"][0] for p in reports]
    container = make_multirun_container(runs, expected_seeds=[42, 123, 2026])
    damaged = copy.deepcopy(container)
    damaged["aggregate"]["metrics_subset"][0]["F1_macro"] = 999.123
    with pytest.raises(ValueError, match="Aggregate numeric mismatch"):
        validate_multiseed_report(damaged, expected_seeds=[42, 123, 2026])


def test_finalizer_rejects_mislabeled_method_or_dataset(tmp_path: Path):
    reports, _, _, _ = _make_reports(tmp_path)
    with pytest.raises(ValueError, match="method_name"):
        finalize_multiseed_delivery(
            reports, tmp_path / "bad_method", method_name="Wrong", dataset_key="demo", expected_seeds=[42, 123, 2026]
        )
    with pytest.raises(ValueError, match="dataset_key"):
        finalize_multiseed_delivery(
            reports, tmp_path / "bad_dataset", method_name="Demo", dataset_key="wrong", expected_seeds=[42, 123, 2026]
        )


def test_strong_status_requires_a_report_path_for_byte_binding(tmp_path: Path):
    reports, bundles, _, latents = _make_reports(tmp_path)
    statuses = _strong_statuses(reports, bundles, latents, tmp_path)
    containers = [extract_embedded_report_payload(path) for path in reports]
    with pytest.raises(ValueError, match="in-memory mapping"):
        finalize_multiseed_delivery(
            containers, tmp_path / "mapping_status", method_name="Demo", dataset_key="demo",
            expected_seeds=[42, 123, 2026], bundles_by_seed=bundles,
            status_files_by_seed=statuses, latent_files_by_seed=latents, include_latents=True,
        )
