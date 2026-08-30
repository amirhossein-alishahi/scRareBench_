from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._version import __version__
from .dashboard import write_multiseed_interactive_report
from .multiseed import (
    aggregate_dashboard_runs,
    canonicalize_embedded_run,
    evaluation_contract_hash,
    extract_embedded_report_payload,
    make_run_id,
    method_training_hash,
    normalize_method_seeds,
    validate_compatible_run,
)


@dataclass(frozen=True)
class MultiseedDelivery:
    """Paths and validated identity of a finalized multi-seed handoff."""

    report_path: Path
    summary_path: Path
    archive_path: Path
    completed_seeds: tuple[int, ...]
    n_runs: int
    report_sha256: str
    archive_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(contiguous.dtype).encode("ascii"))
    h.update(str(tuple(contiguous.shape)).encode("ascii"))
    h.update(contiguous.tobytes(order="C"))
    return h.hexdigest()


def _identity_value(run: Mapping[str, Any], key: str) -> Any:
    value = run.get(key)
    if value is not None:
        return value
    payload = run.get("payload")
    meta = payload.get("meta") if isinstance(payload, Mapping) else None
    return meta.get(key) if isinstance(meta, Mapping) else None


def _run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "run_id", "method_seed", "method_name", "dataset_key",
        "configuration_hash", "evaluation_contract_hash", "method_training_hash",
        "benchmark_seed", "dataset_fingerprint", "dataset_contract_sha256",
        "cell_order_sha256", "latent_sha256",
    )
    return {key: _identity_value(run, key) for key in keys}


def _require_identity_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
    keys: Sequence[str],
    require_actual: bool = True,
) -> None:
    for key in keys:
        exp = expected.get(key)
        got = actual.get(key)
        if exp is None:
            continue
        if got is None:
            if require_actual:
                raise ValueError(f"{context} is missing required identity field {key!r}.")
            continue
        if str(exp) != str(got):
            raise ValueError(f"{context} identity mismatch for {key}: {got!r} != expected {exp!r}.")




def _legacy_bundle_identity(
    manifest: Mapping[str, Any],
    results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize a pre-0.10.4 bundle identity under the current contract.

    Older bundles stored a configuration hash that intentionally ignored the
    benchmark seed and often omitted run_id/dataset_fingerprint from the bundle
    manifest.  Recovery must not compare those stale identifiers directly.
    Instead, reconstruct the same canonical identity now used by embedded HTML
    reports while preserving strict binding to method config, evaluation config,
    cell order and latent content.
    """
    benchmark_cfg = manifest.get("benchmark_config")
    if not isinstance(benchmark_cfg, Mapping):
        benchmark_cfg = {}
    method_cfg = manifest.get("method_configuration")
    if not isinstance(method_cfg, Mapping):
        method_cfg = benchmark_cfg.get("method_config") if isinstance(benchmark_cfg.get("method_config"), Mapping) else {}

    results = results if isinstance(results, Mapping) else {}
    results_run = results.get("run") if isinstance(results.get("run"), Mapping) else {}
    results_prov = results.get("provenance") if isinstance(results.get("provenance"), Mapping) else {}

    dataset_meta = manifest.get("dataset") if isinstance(manifest.get("dataset"), Mapping) else {}
    dataset_key = (
        manifest.get("dataset_key")
        or dataset_meta.get("dataset_key")
        or dataset_meta.get("key")
        or dataset_meta.get("display_name")
    )
    cell_hash = manifest.get("cell_order_sha256") or results_prov.get("cell_order_sha256")
    n_cells = manifest.get("n_cells")
    dataset_fingerprint = manifest.get("dataset_fingerprint") or results_run.get("dataset_fingerprint") or results_prov.get("dataset_fingerprint")
    if dataset_fingerprint is None and dataset_key is not None and cell_hash is not None and n_cells is not None:
        dataset_fingerprint = hashlib.sha256(
            f"{dataset_key}|{cell_hash}|{int(n_cells)}".encode("utf-8")
        ).hexdigest()

    latent_meta = manifest.get("latent") if isinstance(manifest.get("latent"), Mapping) else {}
    latent_hash = latent_meta.get("sha256")
    method_name = manifest.get("method_name")
    method_seed = manifest.get("method_seed")
    config_hash = evaluation_contract_hash(benchmark_cfg, method_cfg)
    training_hash = method_training_hash(method_cfg)
    benchmark_seed = benchmark_cfg.get("benchmark_seed", benchmark_cfg.get("random_state"))
    run_id = None
    if method_name is not None and dataset_fingerprint is not None:
        run_id = make_run_id(
            method_name=str(method_name),
            dataset_fingerprint=str(dataset_fingerprint),
            method_seed=method_seed,
            config_hash=config_hash,
            latent_hash=latent_hash,
        )
    return {
        "run_id": run_id,
        "method_seed": method_seed,
        "method_name": method_name,
        "dataset_key": str(dataset_key) if dataset_key is not None else None,
        "configuration_hash": config_hash,
        "evaluation_contract_hash": config_hash,
        "method_training_hash": training_hash,
        "benchmark_seed": int(benchmark_seed) if benchmark_seed is not None else None,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_contract_sha256": manifest.get("dataset_contract_sha256"),
        "cell_order_sha256": cell_hash,
        "latent_sha256": latent_hash,
    }

def _safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text or "scrarebench"


def _validate_status_file(
    path: Path,
    seed: int,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    report_path: Path | None = None,
    bundle_path: Path | None = None,
    latent_path: Path | None = None,
) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Status file for seed {seed} is not valid JSON: {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"Status file for seed {seed} must contain a JSON object: {path}")
    if data.get("status") != "complete":
        raise ValueError(f"Status file for seed {seed} is not complete: {data.get('status')!r}.")
    if data.get("method_seed") != seed:
        raise ValueError(
            f"Status file seed mismatch for seed {seed}: stored method_seed={data.get('method_seed')!r}."
        )

    expected = dict(expected_identity or {})
    is_strong = bool(data.get("evaluation_contract_hash") or data.get("method_training_hash") or data.get("run_id"))
    if is_strong and report_path is None:
        raise ValueError(
            f"Strong status file for seed {seed} cannot be bound because the corresponding "
            "per-seed report was supplied only as an in-memory mapping. Supply the report path "
            "when status_files_by_seed is used."
        )
    if expected:
        # 0.10.4 statuses bind directly to the report identity. For older statuses,
        # retain safe recovery by mapping the historical configuration_hash to the
        # method-training hash and checking the separately stored benchmark seed.
        if is_strong:
            status_identity = {
                "run_id": data.get("run_id"),
                "method_seed": data.get("method_seed"),
                "method_name": data.get("method", data.get("method_name")),
                "dataset_key": data.get("dataset", data.get("dataset_key")),
                "configuration_hash": data.get("configuration_hash", data.get("evaluation_contract_hash")),
                "evaluation_contract_hash": data.get("evaluation_contract_hash"),
                "method_training_hash": data.get("method_training_hash"),
                "benchmark_seed": data.get("benchmark_seed"),
                "dataset_fingerprint": data.get("dataset_fingerprint"),
                "dataset_contract_sha256": data.get("dataset_contract_sha256"),
                "cell_order_sha256": data.get("cell_order_sha256"),
                "latent_sha256": data.get("latent_array_sha256"),
            }
            _require_identity_match(
                expected, status_identity,
                context=f"Status file for seed {seed}",
                keys=(
                    "run_id", "method_seed", "method_name", "dataset_key",
                    "configuration_hash", "evaluation_contract_hash", "method_training_hash",
                    "benchmark_seed", "dataset_fingerprint", "dataset_contract_sha256",
                    "cell_order_sha256", "latent_sha256",
                ),
                require_actual=True,
            )
        else:
            legacy_training = data.get("configuration_hash")
            if expected.get("method_training_hash") is not None and legacy_training is not None:
                if str(legacy_training) != str(expected.get("method_training_hash")):
                    raise ValueError(
                        f"Legacy status training configuration mismatch for seed {seed}: "
                        f"{legacy_training!r} != expected {expected.get('method_training_hash')!r}."
                    )
            if expected.get("benchmark_seed") is not None and data.get("benchmark_seed") is not None:
                if int(data.get("benchmark_seed")) != int(expected.get("benchmark_seed")):
                    raise ValueError(
                        f"Legacy status benchmark seed mismatch for seed {seed}: "
                        f"{data.get('benchmark_seed')!r} != expected {expected.get('benchmark_seed')!r}."
                    )

    checks = [
        ("report_sha256", report_path),
        ("bundle_sha256", bundle_path),
    ]
    for field, artifact in checks:
        stored = data.get(field)
        if artifact is not None:
            actual = _sha256_file(artifact)
            if stored is None:
                if is_strong:
                    raise ValueError(f"Status file for seed {seed} is missing {field} needed to bind {artifact.name}.")
            elif str(stored) != actual:
                raise ValueError(f"Status file for seed {seed} {field} does not match {artifact}.")
    if latent_path is not None:
        actual_file_sha = _sha256_file(latent_path)
        stored_file_sha = data.get("latent_file_sha256")
        if stored_file_sha is None and not is_strong:
            # v0.10.3 stored the .npy file hash under latent_sha256.
            stored_file_sha = data.get("latent_sha256")
        if stored_file_sha is None:
            if is_strong:
                raise ValueError(f"Status file for seed {seed} is missing latent_file_sha256.")
        elif str(stored_file_sha) != actual_file_sha:
            raise ValueError(f"Status file for seed {seed} latent file SHA256 does not match {latent_path}.")
    return dict(data)


def _validate_latent_file(
    path: Path,
    seed: int,
    expected_n_cells: int | None = None,
    *,
    expected_latent_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        latent = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Latent file for seed {seed} is not a readable pickle-free .npy file: {path}") from exc
    if latent.ndim != 2 or latent.shape[0] <= 0 or latent.shape[1] <= 0:
        raise ValueError(f"Latent file for seed {seed} has invalid shape {latent.shape}.")
    if expected_n_cells is not None and int(latent.shape[0]) != int(expected_n_cells):
        raise ValueError(
            f"Latent row count mismatch for seed {seed}: {latent.shape[0]} != expected {expected_n_cells}."
        )
    if not np.issubdtype(latent.dtype, np.number) or not np.isfinite(latent).all():
        raise ValueError(f"Latent file for seed {seed} contains non-numeric or non-finite values.")
    array_sha = _array_sha256(latent)
    if expected_latent_sha256 is not None and array_sha != str(expected_latent_sha256):
        raise ValueError(
            f"Latent identity mismatch for seed {seed}: array SHA256 {array_sha} != "
            f"report latent_sha256 {expected_latent_sha256}."
        )
    return {
        "shape": [int(x) for x in latent.shape],
        "dtype": str(latent.dtype),
        "file_sha256": _sha256_file(path),
        "array_sha256": array_sha,
    }


def _validate_result_bundle(
    path: Path,
    seed: int,
    *,
    expected_n_cells: int | None = None,
    require_latent: bool = False,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_identity = dict(expected_identity or {})
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"Per-seed result bundle {path} failed CRC at {bad!r}.")
            names = set(zf.namelist())
            required = {
                "bundle_manifest.json",
                "artifact_hashes.json",
                "benchmark_results/results.json",
                "reports/interactive_report.html",
            }
            missing = sorted(required - names)
            if missing:
                raise ValueError(f"Per-seed result bundle for seed {seed} is missing {missing!r}: {path}")

            manifest = json.loads(zf.read("bundle_manifest.json"))
            if not isinstance(manifest, Mapping):
                raise ValueError(f"bundle_manifest.json for seed {seed} is not an object.")
            if manifest.get("method_seed") != seed:
                raise ValueError(
                    f"Per-seed bundle seed mismatch for seed {seed}: "
                    f"bundle_manifest method_seed={manifest.get('method_seed')!r}."
                )
            if expected_n_cells is not None and int(manifest.get("n_cells", -1)) != int(expected_n_cells):
                raise ValueError(
                    f"Per-seed bundle n_cells mismatch for seed {seed}: "
                    f"{manifest.get('n_cells')!r} != expected {expected_n_cells}."
                )
            bundle_is_current = bool(
                manifest.get("evaluation_contract_hash")
                and manifest.get("method_training_hash")
                and manifest.get("run_id")
            )
            if bundle_is_current:
                manifest_identity = {
                    "run_id": manifest.get("run_id"),
                    "method_seed": manifest.get("method_seed"),
                    "method_name": manifest.get("method_name"),
                    "dataset_key": manifest.get("dataset_key"),
                    "configuration_hash": manifest.get("configuration_hash"),
                    "evaluation_contract_hash": manifest.get("evaluation_contract_hash"),
                    "method_training_hash": manifest.get("method_training_hash"),
                    "benchmark_seed": manifest.get("benchmark_seed"),
                    "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                    "dataset_contract_sha256": manifest.get("dataset_contract_sha256"),
                    "cell_order_sha256": manifest.get("cell_order_sha256"),
                    "latent_sha256": (manifest.get("latent") or {}).get("sha256") if isinstance(manifest.get("latent"), Mapping) else None,
                }
            else:
                # Canonicalize legacy manifest identity rather than comparing its
                # historical/stale configuration hash to the current contract.
                manifest_identity = _legacy_bundle_identity(manifest)

            strong_keys = [
                "run_id", "method_seed", "method_name", "configuration_hash",
                "dataset_fingerprint", "cell_order_sha256", "latent_sha256",
            ]
            for optional in ("dataset_key", "evaluation_contract_hash", "method_training_hash", "benchmark_seed", "dataset_contract_sha256"):
                if expected_identity.get(optional) is not None:
                    strong_keys.append(optional)
            _require_identity_match(
                expected_identity, manifest_identity,
                context=f"Per-seed bundle manifest for seed {seed}", keys=strong_keys,
                require_actual=bundle_is_current,
            )

            latent_meta = manifest.get("latent") if isinstance(manifest.get("latent"), Mapping) else {}
            if require_latent and not bool(latent_meta.get("included")):
                raise ValueError(
                    f"Per-seed result bundle for seed {seed} does not include latent data, "
                    "but finalization requested include_latents=True. Recreate that per-seed "
                    "bundle with create_report_bundle(..., include_latent=True), or finalize "
                    "with include_latents=False."
                )

            hashes = json.loads(zf.read("artifact_hashes.json"))
            if not isinstance(hashes, Mapping):
                raise ValueError(f"artifact_hashes.json for seed {seed} is not an object.")
            for member, info in hashes.items():
                if member not in names:
                    raise ValueError(f"Per-seed bundle seed {seed} hash manifest references missing {member!r}.")
                if not isinstance(info, Mapping):
                    raise ValueError(f"Invalid hash entry for {member!r} in seed {seed} bundle.")
                data = zf.read(member)
                expected_hash = info.get("sha256")
                expected_size = info.get("size_bytes")
                if expected_hash and _sha256_bytes(data) != expected_hash:
                    raise ValueError(f"Per-seed bundle seed {seed} hash mismatch for {member!r}.")
                if expected_size is not None and len(data) != int(expected_size):
                    raise ValueError(f"Per-seed bundle seed {seed} size mismatch for {member!r}.")

            results = json.loads(zf.read("benchmark_results/results.json"))
            benchmark = results.get("benchmark") if isinstance(results, Mapping) else None
            if not isinstance(benchmark, Mapping):
                raise ValueError(f"Per-seed bundle seed {seed} results.json has no benchmark object.")
            if expected_n_cells is not None and int(benchmark.get("n_cells", -1)) != int(expected_n_cells):
                raise ValueError(
                    f"Per-seed bundle results n_cells mismatch for seed {seed}: "
                    f"{benchmark.get('n_cells')!r} != expected {expected_n_cells}."
                )
            run_obj = results.get("run") if isinstance(results.get("run"), Mapping) else {}
            prov = results.get("provenance") if isinstance(results.get("provenance"), Mapping) else {}
            method_obj = results.get("method") if isinstance(results.get("method"), Mapping) else {}
            if bundle_is_current:
                results_identity = {
                    "run_id": run_obj.get("run_id", prov.get("run_id")),
                    "method_seed": run_obj.get("method_seed", prov.get("method_seed")),
                    "method_name": method_obj.get("name"),
                    "dataset_key": run_obj.get("dataset_key", prov.get("dataset_key")),
                    "configuration_hash": run_obj.get("configuration_hash", prov.get("configuration_hash")),
                    "evaluation_contract_hash": run_obj.get("evaluation_contract_hash", prov.get("evaluation_contract_hash")),
                    "method_training_hash": run_obj.get("method_training_hash", prov.get("method_training_hash")),
                    "benchmark_seed": benchmark.get("benchmark_seed"),
                    "dataset_fingerprint": run_obj.get("dataset_fingerprint", prov.get("dataset_fingerprint")),
                    "dataset_contract_sha256": run_obj.get("dataset_contract_sha256", prov.get("dataset_contract_sha256")),
                    "cell_order_sha256": prov.get("cell_order_sha256"),
                    "latent_sha256": (prov.get("latent") or {}).get("sha256") if isinstance(prov.get("latent"), Mapping) else None,
                }
            else:
                # results.json is already covered by the bundle's cryptographic
                # artifact manifest. Bind its legacy identity by recomputing the
                # modern contract from the same benchmark/method metadata.
                legacy_manifest = dict(manifest)
                legacy_manifest["benchmark_config"] = benchmark.get("config") if isinstance(benchmark.get("config"), Mapping) else manifest.get("benchmark_config")
                legacy_manifest["method_configuration"] = (
                    benchmark.get("method_config")
                    if isinstance(benchmark.get("method_config"), Mapping)
                    else manifest.get("method_configuration")
                )
                legacy_manifest["dataset_fingerprint"] = run_obj.get("dataset_fingerprint", prov.get("dataset_fingerprint"))
                results_identity = _legacy_bundle_identity(legacy_manifest, results)
                results_identity["method_name"] = method_obj.get("name") or results_identity.get("method_name")
                results_identity["cell_order_sha256"] = prov.get("cell_order_sha256") or results_identity.get("cell_order_sha256")
                if isinstance(prov.get("latent"), Mapping):
                    results_identity["latent_sha256"] = prov["latent"].get("sha256") or results_identity.get("latent_sha256")
                if results_identity.get("run_id") is None and results_identity.get("method_name") is not None and results_identity.get("dataset_fingerprint") is not None:
                    results_identity["run_id"] = make_run_id(
                        method_name=str(results_identity["method_name"]),
                        dataset_fingerprint=str(results_identity["dataset_fingerprint"]),
                        method_seed=results_identity.get("method_seed"),
                        config_hash=str(results_identity["configuration_hash"]),
                        latent_hash=results_identity.get("latent_sha256"),
                    )
            _require_identity_match(
                expected_identity, results_identity,
                context=f"Per-seed results.json for seed {seed}", keys=strong_keys,
                require_actual=bundle_is_current,
            )

            if not bundle_is_current:
                # The historical bundle manifest did not carry dataset_fingerprint;
                # results.json did. Recompute once more with that trusted (hashed)
                # value so run_id/config binding is strict even during recovery.
                manifest_identity = _legacy_bundle_identity(manifest, results)
                _require_identity_match(
                    expected_identity, manifest_identity,
                    context=f"Legacy per-seed bundle manifest for seed {seed}", keys=strong_keys,
                    require_actual=False,
                )

            embedded_html = zf.read("reports/interactive_report.html").decode("utf-8")
            embedded_validation = validate_multiseed_report(
                embedded_html,
                expected_seeds=None,
                require_all_expected=False,
                require_core_tables=True,
                require_points_when_enabled=True,
            )
            if embedded_validation["completed_seeds"] != [seed]:
                raise ValueError(
                    f"Per-seed bundle interactive report contains seeds "
                    f"{embedded_validation['completed_seeds']!r}; expected [{seed}]."
                )
            embedded_identity = embedded_validation["payloads"][0]
            _require_identity_match(
                expected_identity, embedded_identity,
                context=f"Per-seed embedded HTML for seed {seed}", keys=strong_keys,
                require_actual=bool(manifest.get("scrarebench_version") == __version__),
            )

            latent_member = latent_meta.get("path") if isinstance(latent_meta, Mapping) else None
            if bool(latent_meta.get("included")):
                # Older manifests do not store the member path explicitly. Locate the
                # sole latent .npy member when necessary.
                if not latent_member:
                    candidates = sorted(name for name in names if name.startswith("latent/") and name.endswith("_latent.npy"))
                    latent_member = candidates[0] if len(candidates) == 1 else None
                if not latent_member or latent_member not in names:
                    raise ValueError(f"Per-seed bundle seed {seed} declares latent inclusion but no latent .npy member is present.")
                try:
                    latent_arr = np.load(io.BytesIO(zf.read(latent_member)), allow_pickle=False)
                except Exception as exc:
                    raise ValueError(f"Per-seed bundle seed {seed} latent member is unreadable.") from exc
                if not np.issubdtype(latent_arr.dtype, np.number) or not np.isfinite(latent_arr).all():
                    raise ValueError(f"Per-seed bundle seed {seed} latent member is non-numeric/non-finite.")
                member_array_sha = _array_sha256(latent_arr)
                expected_latent = expected_identity.get("latent_sha256")
                if expected_latent is not None and member_array_sha != str(expected_latent):
                    raise ValueError(
                        f"Per-seed bundle seed {seed} latent array identity mismatch: "
                        f"{member_array_sha} != expected {expected_latent}."
                    )

            return {
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
                "bundle_version": manifest.get("scrarebench_version"),
                "latent_included": bool(latent_meta.get("included")),
                "identity": manifest_identity,
            }
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Per-seed result bundle for seed {seed} is not a valid ZIP: {path}") from exc


def _table_row_count(value: Any) -> int:
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        return len(value["rows"])
    if isinstance(value, list):
        return len(value)
    return 0


def _decoded_length(encoded: Mapping[str, Any] | None) -> int | None:
    if not isinstance(encoded, Mapping):
        return None
    raw = encoded.get("codes_b64")
    if not isinstance(raw, str) or not raw:
        return None
    dtype = str(encoded.get("dtype") or "uint16").lower()
    itemsize = {"uint8": 1, "uint16": 2, "uint32": 4}.get(dtype)
    if itemsize is None:
        return None
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        return None
    if len(data) % itemsize:
        return None
    return len(data) // itemsize


def _coordinate_length(encoded: Mapping[str, Any] | None) -> int | None:
    if not isinstance(encoded, Mapping):
        return None
    raw = encoded.get("data_b64")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        return None
    if len(data) % 2:
        return None
    return len(data) // 2



def _table_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and isinstance(value.get("rows"), list):
        return [dict(row) for row in value["rows"] if isinstance(row, Mapping)]
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _nested_value(root: Mapping[str, Any], path: Sequence[str]) -> Any:
    node: Any = root
    for part in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(part)
    return node


def _identity_set_for_table(
    run: Mapping[str, Any],
    path: Sequence[str],
    identity_keys: Sequence[str],
    *,
    table_name: str,
) -> set[tuple[Any, ...]]:
    payload = run.get("payload") if isinstance(run.get("payload"), Mapping) else {}
    rows = _table_rows(_nested_value(payload, path))
    identities = [tuple(row.get(key) for key in identity_keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(
            f"Run seed {run.get('method_seed')!r} contains duplicate row identities in {table_name}."
        )
    return set(identities)


def _validate_cross_seed_table_identities(runs: Sequence[Mapping[str, Any]]) -> None:
    if len(runs) < 2:
        return
    specs = [
        (("metrics", "subset"), ("subset",), "metrics.subset", True),
        (("metrics", "per_type"), ("cell_type",), "metrics.per_type", True),
        (("rare", "per_type"), ("cell_type", "scenario", "distribution", "topology"), "rare.per_type", True),
        (("rare", "summary"), ("metric", "metric_type"), "rare.summary", True),
        (("rare", "scenarios"), ("scenario", "distribution", "topology"), "rare.scenarios", True),
        (("rare", "resolution_sensitivity"), ("resolution", "cell_type", "scenario"), "rare.resolution_sensitivity", True),
        (("scib", "metrics"), ("metric",), "scib.metrics", False),
        (("scib", "aggregates"), ("metric",), "scib.aggregates", False),
    ]
    first = runs[0]
    for path, keys, name, required in specs:
        baseline = _identity_set_for_table(first, path, keys, table_name=name)
        if required and not baseline:
            raise ValueError(f"Run seed {first.get('method_seed')!r} has no rows in required table {name}.")
        for run in runs[1:]:
            current = _identity_set_for_table(run, path, keys, table_name=name)
            if not baseline and not current:
                continue
            if current != baseline:
                missing = sorted(baseline - current, key=lambda x: tuple(str(v) for v in x))
                extra = sorted(current - baseline, key=lambda x: tuple(str(v) for v in x))
                raise ValueError(
                    f"Cross-seed row identity mismatch in {name} for seed {run.get('method_seed')!r}: "
                    f"missing={missing[:8]!r}, extra={extra[:8]!r}. "
                    "Metric values may be missing/NaN, but scientific rows may not silently disappear or appear across seeds."
                )


def _deep_compare(expected: Any, actual: Any, *, path: str = "$", rtol: float = 1e-12, atol: float = 1e-12) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        if set(expected) != set(actual):
            raise ValueError(
                f"Aggregate structure mismatch at {path}: keys {sorted(actual)} != expected {sorted(expected)}."
            )
        for key in expected:
            _deep_compare(expected[key], actual[key], path=f"{path}.{key}", rtol=rtol, atol=atol)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"Aggregate list length mismatch at {path}: {len(actual)} != expected {len(expected)}.")
        for i, (exp, got) in enumerate(zip(expected, actual)):
            _deep_compare(exp, got, path=f"{path}[{i}]", rtol=rtol, atol=atol)
        return
    numeric_types = (int, float, np.integer, np.floating)
    if isinstance(expected, numeric_types) and not isinstance(expected, (bool, np.bool_)) and isinstance(actual, numeric_types) and not isinstance(actual, (bool, np.bool_)):
        exp = float(expected); got = float(actual)
        if not (math.isfinite(exp) and math.isfinite(got)):
            if exp != got:
                raise ValueError(f"Aggregate numeric mismatch at {path}: {got!r} != expected {exp!r}.")
            return
        if not math.isclose(exp, got, rel_tol=rtol, abs_tol=atol):
            raise ValueError(f"Aggregate numeric mismatch at {path}: {got!r} != expected {exp!r}.")
        return
    if expected != actual:
        raise ValueError(f"Aggregate value mismatch at {path}: {actual!r} != expected {expected!r}.")


def validate_dashboard_run_payload(
    run: Mapping[str, Any],
    *,
    require_core_tables: bool = True,
    require_points_when_enabled: bool = True,
) -> dict[str, Any]:
    """Validate that an embedded run still contains usable dashboard data."""

    payload = run.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Run {run.get('run_id')!r} has no dashboard payload.")

    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"Run seed {run.get('method_seed')!r} has no payload.meta mapping.")
    try:
        n_cells = int(meta.get("n_cells"))
    except Exception as exc:
        raise ValueError(f"Run seed {run.get('method_seed')!r} has invalid meta.n_cells.") from exc
    if n_cells <= 0:
        raise ValueError(f"Run seed {run.get('method_seed')!r} has non-positive meta.n_cells={n_cells}.")

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    rare = payload.get("rare") if isinstance(payload.get("rare"), Mapping) else {}
    counts = {
        "metrics_subset_rows": _table_row_count(metrics.get("subset")),
        "metrics_per_type_rows": _table_row_count(metrics.get("per_type")),
        "rare_per_type_rows": _table_row_count(rare.get("per_type")),
        "rare_summary_rows": _table_row_count(rare.get("summary")),
        "rare_scenario_rows": _table_row_count(rare.get("scenarios")),
    }
    if require_core_tables:
        if counts["metrics_subset_rows"] <= 0:
            raise ValueError(f"Run seed {run.get('method_seed')!r} lost the required metrics.subset table.")
        if counts["rare_per_type_rows"] <= 0:
            raise ValueError(f"Run seed {run.get('method_seed')!r} lost the required rare.per_type table.")

    sections = payload.get("sections") if isinstance(payload.get("sections"), Mapping) else {}
    points = payload.get("points") if isinstance(payload.get("points"), Mapping) else None
    features = payload.get("features") if isinstance(payload.get("features"), Mapping) else {}
    needs_points = bool(sections.get("umap")) or bool(features.get("rare_umap"))
    if require_points_when_enabled and needs_points:
        if not isinstance(points, Mapping):
            raise ValueError(f"Run seed {run.get('method_seed')!r} enables UMAP but has no points payload.")
        q = points.get("coords_q16")
        if not isinstance(q, Mapping):
            raise ValueError(f"Run seed {run.get('method_seed')!r} has no quantized coordinate payload.")
        x_n = _coordinate_length(q.get("x"))
        y_n = _coordinate_length(q.get("y"))
        if x_n != n_cells or y_n != n_cells:
            raise ValueError(
                f"Run seed {run.get('method_seed')!r} coordinate payload length mismatch: "
                f"x={x_n}, y={y_n}, expected={n_cells}."
            )
        fields = points.get("fields")
        if not isinstance(fields, Mapping):
            raise ValueError(f"Run seed {run.get('method_seed')!r} has no points.fields mapping.")
        for required in ("celltype", "batch", "cluster", "prediction"):
            length = _decoded_length(fields.get(required))
            if length != n_cells:
                raise ValueError(
                    f"Run seed {run.get('method_seed')!r} field {required!r} has "
                    f"{length} encoded values; expected {n_cells}."
                )
        cell_ids = points.get("cell_id")
        if isinstance(cell_ids, list) and cell_ids and len(cell_ids) != n_cells:
            raise ValueError(
                f"Run seed {run.get('method_seed')!r} cell_id length {len(cell_ids)} != {n_cells}."
            )

    if bool(sections.get("reproducibility")):
        repro = payload.get("reproducibility")
        if not isinstance(repro, Mapping) or not str(repro.get("run_config") or "").strip():
            raise ValueError(f"Run seed {run.get('method_seed')!r} lost reproducibility.run_config.")

    identity = _run_identity(run)
    # Top-level run identity and payload.meta must never disagree. Missing fields
    # remain tolerated for legacy reports, but 0.10.4 reports populate all of them.
    identity_keys = (
        "run_id", "method_seed", "method_name", "dataset_key",
        "configuration_hash", "evaluation_contract_hash", "method_training_hash",
        "benchmark_seed", "dataset_fingerprint", "dataset_contract_sha256",
        "cell_order_sha256", "latent_sha256",
    )
    meta_identity = {key: meta.get(key) for key in identity_keys}
    _require_identity_match(identity, meta_identity, context=f"Dashboard payload.meta for seed {run.get('method_seed')!r}", keys=identity_keys, require_actual=False)

    return {
        **identity,
        "n_cells": n_cells,
        **counts,
        "has_points": isinstance(points, Mapping),
    }


def validate_multiseed_report(
    report_or_container: str | Path | Mapping[str, Any],
    *,
    expected_seeds: Sequence[int] | None = None,
    require_all_expected: bool = True,
    require_core_tables: bool = True,
    require_points_when_enabled: bool = True,
) -> dict[str, Any]:
    """Deep-validate a multi-seed HTML/container and its embedded scientific data."""

    if isinstance(report_or_container, Mapping):
        data = dict(report_or_container)
    else:
        data = extract_embedded_report_payload(report_or_container)
    if data.get("report_type") != "scrarebench_multi_run":
        raise ValueError("Expected a scrarebench_multi_run report container.")
    raw_runs = data.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("Multi-seed report contains no runs.")

    runs: list[dict[str, Any]] = []
    payload_summaries: list[dict[str, Any]] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            raise ValueError("Multi-seed report contains a non-mapping run entry.")
        run = canonicalize_embedded_run(raw)
        validate_compatible_run(runs, run)
        payload_summaries.append(
            validate_dashboard_run_payload(
                run,
                require_core_tables=require_core_tables,
                require_points_when_enabled=require_points_when_enabled,
            )
        )
        runs.append(run)

    seeds = [r.get("method_seed") for r in runs]
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError(f"Every finalized run must have an integer method_seed; got {seeds!r}.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate method seeds in finalized report: {seeds!r}.")

    expected = normalize_method_seeds(expected_seeds) if expected_seeds is not None else []
    if expected and require_all_expected and set(seeds) != set(expected):
        raise ValueError(f"Finalized seeds {seeds!r} do not match expected seeds {expected!r}.")
    if expected and not require_all_expected:
        unexpected = [seed for seed in seeds if seed not in expected]
        if unexpected:
            raise ValueError(f"Finalized report contains unexpected seeds {unexpected!r}; expected subset of {expected!r}.")

    aggregate = data.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("Multi-seed report has no aggregate payload.")
    if int(aggregate.get("n_stored", -1)) != len(runs):
        raise ValueError("Aggregate n_stored does not match embedded run count.")
    included_count = sum(bool(r.get("included", True)) for r in runs)
    if int(aggregate.get("n_included", -1)) != included_count:
        raise ValueError("Aggregate n_included does not match embedded run inclusion flags.")

    # Scientific row identities must be identical across method seeds. A metric
    # value may legitimately be unavailable, but an entire biological/scenario row
    # disappearing is a structural error and must fail closed.
    _validate_cross_seed_table_identities(runs)

    recomputed = aggregate_dashboard_runs(runs)
    aggregate_for_compare = dict(aggregate)
    # Legacy reports may contain aggregate.included_run_ids derived from the old
    # configuration-hash semantics. Canonicalizing their embedded runs changes
    # only those deterministic IDs. Accept that migration iff the stored IDs
    # exactly match the *raw* included run IDs; all scientific aggregate values
    # remain subject to the same deep comparison.
    raw_included_ids = [
        raw.get("run_id") for raw in raw_runs
        if isinstance(raw, Mapping) and bool(raw.get("included", True))
    ]
    stored_included_ids = aggregate_for_compare.get("included_run_ids")
    canonical_included_ids = recomputed.get("included_run_ids")
    if (
        isinstance(stored_included_ids, list)
        and stored_included_ids == raw_included_ids
        and stored_included_ids != canonical_included_ids
    ):
        aggregate_for_compare["included_run_ids"] = canonical_included_ids
    _deep_compare(recomputed, aggregate_for_compare, path="$.aggregate")

    return {
        "n_runs": len(runs),
        "completed_seeds": seeds,
        "included_seeds": [r.get("method_seed") for r in runs if bool(r.get("included", True))],
        "payloads": payload_summaries,
    }


def _normalize_seed_file_map(
    value: Mapping[int, str | Path] | None,
    *,
    name: str,
) -> dict[int, Path]:
    if value is None:
        return {}
    out: dict[int, Path] = {}
    for raw_seed, raw_path in value.items():
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError(f"{name} keys must be integer method seeds; got {raw_seed!r}.")
        seed = int(raw_seed)
        path = Path(raw_path)
        if seed in out:
            raise ValueError(f"Duplicate seed {seed} in {name}.")
        out[seed] = path
    return out


def finalize_multiseed_delivery(
    reports: Sequence[str | Path | Mapping[str, Any]],
    output_dir: str | Path,
    *,
    method_name: str,
    dataset_key: str,
    expected_seeds: Sequence[int],
    title: str | None = None,
    report_filename: str = "interactive_report_multiseed.html",
    summary_filename: str = "multi_seed_run_summary.json",
    archive_filename: str | None = None,
    bundles_by_seed: Mapping[int, str | Path] | None = None,
    status_files_by_seed: Mapping[int, str | Path] | None = None,
    latent_files_by_seed: Mapping[int, str | Path] | None = None,
    include_latents: bool = False,
    require_all_expected: bool = True,
    extra_summary: Mapping[str, Any] | None = None,
) -> MultiseedDelivery:
    """Atomically build, validate, and archive a complete multi-seed handoff."""

    if not reports:
        raise ValueError("At least one per-seed interactive report is required.")
    expected = normalize_method_seeds(expected_seeds)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = _normalize_seed_file_map(bundles_by_seed, name="bundles_by_seed")
    statuses = _normalize_seed_file_map(status_files_by_seed, name="status_files_by_seed")
    latents = _normalize_seed_file_map(latent_files_by_seed, name="latent_files_by_seed")

    if archive_filename is None:
        archive_filename = (
            f"{_safe_slug(method_name)}_{_safe_slug(dataset_key)}_"
            f"scRareBench_v{__version__.replace('.', '_')}_multiseed_results.zip"
        )

    final_report = out_dir / report_filename
    final_summary = out_dir / summary_filename
    final_archive = out_dir / archive_filename

    with tempfile.TemporaryDirectory(prefix="scrarebench_multiseed_finalize_", dir=out_dir) as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        tmp_report = temp_dir / report_filename
        tmp_summary = temp_dir / summary_filename
        tmp_archive = temp_dir / archive_filename

        write_multiseed_interactive_report(
            reports,
            tmp_report,
            title=title or f"scRareBench — {method_name} — {dataset_key} — multi-seed",
            expected_seeds=expected,
        )
        validation = validate_multiseed_report(
            tmp_report,
            expected_seeds=expected,
            require_all_expected=require_all_expected,
            require_core_tables=True,
            require_points_when_enabled=True,
        )
        completed = list(validation["completed_seeds"])

        payload_by_seed = {int(item["method_seed"]): item for item in validation["payloads"]}
        # Caller-provided naming must agree with the scientific payload; otherwise
        # a correctly computed report could be mislabeled at final handoff.
        for seed, identity in payload_by_seed.items():
            if identity.get("method_name") is not None and str(identity.get("method_name")) != str(method_name):
                raise ValueError(
                    f"Finalizer method_name={method_name!r} does not match report seed {seed} "
                    f"method_name={identity.get('method_name')!r}."
                )
            if identity.get("dataset_key") is not None and str(identity.get("dataset_key")) != str(dataset_key):
                raise ValueError(
                    f"Finalizer dataset_key={dataset_key!r} does not match report seed {seed} "
                    f"dataset_key={identity.get('dataset_key')!r}."
                )

        by_seed_report: dict[int, Path] = {}
        for item in reports:
            if isinstance(item, Mapping):
                data = item
                source_path = None
            else:
                source_path = Path(item)
                data = extract_embedded_report_payload(source_path)
            if isinstance(data, Mapping):
                for run in data.get("runs", []):
                    seed = run.get("method_seed") if isinstance(run, Mapping) else None
                    if isinstance(seed, int) and source_path is not None:
                        by_seed_report.setdefault(seed, source_path)

        for mapping, label in ((bundles, "bundle"), (statuses, "status file")):
            if mapping:
                missing_seeds = [seed for seed in completed if seed not in mapping]
                if missing_seeds:
                    raise FileNotFoundError(f"Missing {label} mapping for finalized seeds {missing_seeds!r}.")
                for seed in completed:
                    path = mapping[seed]
                    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                        raise FileNotFoundError(f"Missing/empty {label} for seed {seed}: {path}")
        if include_latents:
            missing_seeds = [seed for seed in completed if seed not in latents]
            if missing_seeds:
                raise FileNotFoundError(f"Latent inclusion requested but no latent mapping for seeds {missing_seeds!r}.")
        if latents:
            missing_mapped = [seed for seed in completed if seed not in latents]
            if missing_mapped:
                raise FileNotFoundError(f"Latent mappings were supplied but are missing finalized seeds {missing_mapped!r}.")
            for seed in completed:
                path = latents[seed]
                if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                    raise FileNotFoundError(f"Missing/empty latent for seed {seed}: {path}")

        companion_validation: dict[str, Any] = {"statuses": {}, "bundles": {}, "latents": {}}
        for seed in completed:
            identity = payload_by_seed[seed]
            n_cells = int(identity["n_cells"])
            if seed in bundles:
                companion_validation["bundles"][str(seed)] = _validate_result_bundle(
                    bundles[seed],
                    seed,
                    expected_n_cells=n_cells,
                    require_latent=include_latents,
                    expected_identity=identity,
                )
            if seed in latents:
                companion_validation["latents"][str(seed)] = _validate_latent_file(
                    latents[seed],
                    seed,
                    expected_n_cells=n_cells,
                    expected_latent_sha256=identity.get("latent_sha256"),
                )
            if seed in statuses:
                companion_validation["statuses"][str(seed)] = _validate_status_file(
                    statuses[seed],
                    seed,
                    expected_identity=identity,
                    report_path=by_seed_report.get(seed),
                    bundle_path=bundles.get(seed),
                    latent_path=latents.get(seed),
                )

        report_sha = _sha256_file(tmp_report)
        report_size = tmp_report.stat().st_size
        if report_size <= 1024:
            raise ValueError(f"Generated multi-seed report is implausibly small ({report_size} bytes).")

        summary: dict[str, Any] = {
            "scrarebench_version": __version__,
            "method": str(method_name),
            "dataset": str(dataset_key),
            "requested_method_seeds": expected,
            "completed_method_seeds": completed,
            "n_runs": int(validation["n_runs"]),
            "multi_report": report_filename,
            "multi_report_sha256": report_sha,
            "multi_report_size_bytes": report_size,
            "payload_validation": validation["payloads"],
            "companion_validation": companion_validation,
            "include_latents_in_delivery": bool(include_latents),
        }
        if extra_summary:
            summary["extra"] = json.loads(json.dumps(dict(extra_summary), default=str))
        tmp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

        member_hashes: dict[str, dict[str, Any]] = {}
        with zipfile.ZipFile(tmp_archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            def add_file(path: Path, arcname: str) -> None:
                zf.write(path, arcname=arcname)
                member_hashes[arcname] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}

            add_file(tmp_report, report_filename)
            add_file(tmp_summary, summary_filename)

            for seed in completed:
                if seed in by_seed_report:
                    add_file(by_seed_report[seed], f"seed_{seed}/interactive_report.html")
                if seed in statuses:
                    add_file(statuses[seed], f"seed_{seed}/status.json")
                if seed in bundles:
                    add_file(bundles[seed], f"seed_{seed}/result_bundle.zip")
                if include_latents:
                    add_file(latents[seed], f"seed_{seed}/latent.npy")

            manifest = {
                "scrarebench_version": __version__,
                "method": str(method_name),
                "dataset": str(dataset_key),
                "completed_seeds": completed,
                "files": member_hashes,
            }
            zf.writestr("delivery_manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))

        with zipfile.ZipFile(tmp_archive, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise ValueError(f"Final delivery ZIP failed CRC validation at member {bad!r}.")
            names = set(zf.namelist())
            for required in (report_filename, summary_filename, "delivery_manifest.json"):
                if required not in names:
                    raise ValueError(f"Final delivery ZIP is missing required member {required!r}.")
            delivery_manifest = json.loads(zf.read("delivery_manifest.json"))
            if not isinstance(delivery_manifest, Mapping) or not isinstance(delivery_manifest.get("files"), Mapping):
                raise ValueError("Final delivery ZIP has an invalid delivery_manifest.json.")
            for member, info in delivery_manifest["files"].items():
                if member not in names:
                    raise ValueError(f"Final delivery manifest references missing member {member!r}.")
                if not isinstance(info, Mapping):
                    raise ValueError(f"Invalid delivery manifest entry for {member!r}.")
                data = zf.read(member)
                if info.get("sha256") and _sha256_bytes(data) != str(info.get("sha256")):
                    raise ValueError(f"Final delivery ZIP SHA256 mismatch for member {member!r}.")
                if info.get("size_bytes") is not None and len(data) != int(info.get("size_bytes")):
                    raise ValueError(f"Final delivery ZIP size mismatch for member {member!r}.")

            html_bytes = zf.read(report_filename)
            if _sha256_bytes(html_bytes) != report_sha:
                raise ValueError("Report bytes inside final ZIP do not match the validated report on disk.")
            validate_multiseed_report(
                html_bytes.decode("utf-8"),
                expected_seeds=expected,
                require_all_expected=require_all_expected,
                require_core_tables=True,
                require_points_when_enabled=True,
            )
            for seed in completed:
                if bundles and f"seed_{seed}/result_bundle.zip" not in names:
                    raise ValueError(f"Final ZIP is missing result bundle for seed {seed}.")
                if statuses and f"seed_{seed}/status.json" not in names:
                    raise ValueError(f"Final ZIP is missing status file for seed {seed}.")
                if include_latents and f"seed_{seed}/latent.npy" not in names:
                    raise ValueError(f"Final ZIP is missing latent file for seed {seed}.")

        os.replace(tmp_report, final_report)
        os.replace(tmp_summary, final_summary)
        os.replace(tmp_archive, final_archive)

    return MultiseedDelivery(
        report_path=final_report,
        summary_path=final_summary,
        archive_path=final_archive,
        completed_seeds=tuple(completed),
        n_runs=len(completed),
        report_sha256=_sha256_file(final_report),
        archive_sha256=_sha256_file(final_archive),
    )
