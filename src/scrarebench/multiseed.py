from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._version import __version__
from .constants import RESULTS_SCHEMA_VERSION

# Fields that may legitimately differ between repeated method-seed runs even
# when the scientific benchmark configuration is identical.  In particular,
# realized clustering/graph statistics are *outputs*, not configuration inputs.
_VOLATILE_CONFIG_KEYS = {
    # Only the stochastic METHOD seed is volatile across replicates. The
    # benchmark/evaluation seed is part of the scientific contract and MUST
    # remain identical across runs that are aggregated together.
    "seed", "method_seed", "timestamp",
    "started_at", "finished_at", "training_seconds", "runtime_seconds",
    "output_dir", "results_dir", "work_dir", "model_dir", "artifact_dir",
    "cache_dir", "device", "devices", "accelerator",
    "reference_n_clusters", "cluster_count_warning",
    "realized_degree_mean", "realized_degree_min", "realized_degree_max",
}


def normalize_method_seeds(value: int | np.integer | Sequence[int] | None, *, default: int = 42) -> list[int]:
    """Normalize a scalar or sequence of method seeds into a unique ordered list.

    Strings are intentionally rejected so a typo such as ``"42,123"`` does not
    silently become a malformed iterable. Duplicate seeds are rejected because
    treating two runs with the same seed as independent replicates would create
    pseudo-replication in aggregate statistics.
    """
    if value is None:
        values: list[Any] = [default]
    elif isinstance(value, (int, np.integer)):
        values = [int(value)]
    elif isinstance(value, (str, bytes)):
        raise TypeError("METHOD_SEED must be an integer or a sequence of integers, not a string.")
    else:
        values = list(value)
        if not values:
            raise ValueError("At least one method seed is required.")
    out: list[int] = []
    seen: set[int] = set()
    for raw in values:
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError(f"Invalid method seed {raw!r}; every seed must be an integer.")
        seed = int(raw)
        if seed < 0:
            raise ValueError(f"Invalid method seed {seed}; seeds must be non-negative integers.")
        if seed in seen:
            raise ValueError(f"Duplicate method seed {seed}; each seed may appear only once.")
        seen.add(seed)
        out.append(seed)
    return out


def _json_safe(value: Any, *, _path: str = "$") -> Any:
    """Return a deterministic JSON-safe representation for configuration identity.

    Sets are sorted by their canonical JSON representation so hash identity does
    not depend on Python hash randomization. NumPy arrays/scalars are normalized
    explicitly. Unsupported objects fail with a contextual path rather than a raw
    ``json.dumps`` TypeError.
    """
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v, _path=f"{_path}.{k}") for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, _path=f"{_path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, set):
        items = [_json_safe(v, _path=f"{_path}{{item}}") for v in value]
        return sorted(
            items,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False),
        )
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), _path=_path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(
                f"Configuration contains a non-finite numeric value at {_path}: {numeric!r}. "
                "NaN and +/-Inf are not valid configuration values."
            )
        return numeric
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    raise TypeError(
        f"Configuration contains a non-JSON-serializable value at {_path}: "
        f"{type(value).__module__}.{type(value).__qualname__}. "
        "Use JSON-compatible scalars/containers, pathlib.Path, or NumPy arrays/scalars."
    )


def _strip_volatile(value: Any, *, _path: str = "$") -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _strip_volatile(v, _path=f"{_path}.{k}")
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k).lower() not in _VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(v, _path=f"{_path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, set):
        items = [_strip_volatile(v, _path=f"{_path}{{item}}") for v in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ),
        )
    return _json_safe(value, _path=_path)


def configuration_hash(config: Mapping[str, Any] | None) -> str:
    """Hash configuration deterministically while ignoring method-run volatility.

    ``method_seed``/``seed`` and runtime/output fields are excluded so stochastic
    method replicates can be compared.  The benchmark seed (``random_state`` or
    ``benchmark_seed``) is deliberately *not* excluded: changing evaluation
    randomness changes the benchmark contract and therefore must prevent merging.
    Realized downstream outcomes such as ``reference_n_clusters`` remain excluded.
    """
    cleaned = _strip_volatile(dict(config or {}))
    raw = json.dumps(cleaned, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


_EVALUATION_CONTRACT_KEYS = (
    "method_name", "representation_key", "label_key", "batch_key", "scenario_key",
    "reference_resolution", "resolution_sweep", "n_neighbors", "distance_metric",
    "leiden_flavor", "leiden_n_iterations", "overwrite", "rare_evaluation",
    "scenario_policy", "strict_scenario_labels", "scib",
)


def method_training_hash(method_config: Mapping[str, Any] | None) -> str:
    """Identity of the method training/preprocessing configuration.

    The method seed itself is excluded by :func:`configuration_hash`, making this
    suitable for deciding whether a cached latent/model can be reused.
    """
    return configuration_hash(dict(method_config or {}))


def evaluation_contract(benchmark_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract requested evaluation settings, excluding realized evaluation outputs."""
    cfg = dict(benchmark_config or {})
    out: dict[str, Any] = {}
    for key in _EVALUATION_CONTRACT_KEYS:
        if key in cfg:
            out[key] = cfg[key]
    # EvaluationConfig uses random_state; persisted run configs also expose the
    # same value as benchmark_seed. Canonicalize both spellings to one field.
    explicit_seed = cfg.get("benchmark_seed")
    random_state = cfg.get("random_state")
    if explicit_seed is not None and random_state is not None:
        try:
            same_seed = int(explicit_seed) == int(random_state)
        except (TypeError, ValueError):
            same_seed = str(explicit_seed) == str(random_state)
        if not same_seed:
            raise ValueError(
                "Evaluation configuration is internally inconsistent: "
                f"benchmark_seed={explicit_seed!r} differs from random_state={random_state!r}."
            )
    benchmark_seed = explicit_seed if explicit_seed is not None else random_state
    if benchmark_seed is not None:
        out["benchmark_seed"] = benchmark_seed
    return out


def evaluation_contract_hash(
    benchmark_config: Mapping[str, Any] | None,
    method_config: Mapping[str, Any] | None = None,
) -> str:
    """Hash the complete requested scientific contract for one benchmark family."""
    cfg = dict(benchmark_config or {})
    method_cfg = dict(method_config or cfg.get("method_config") or {})
    return configuration_hash({"benchmark": evaluation_contract(cfg), "method": method_cfg})


def dataset_contract_hash(
    adata: Any,
    *,
    dataset_key: str,
    label_key: str,
    batch_key: str,
    scenario_key: str = "scrarebench_scenario",
) -> str:
    """Hash the ordered dataset/reference contract used for multi-seed comparison.

    Unlike the historical cell-order fingerprint, this includes the ordered cell
    IDs, reference labels, batch labels, rare-scenario labels, and feature IDs.
    Thus an annotation/HVG change cannot masquerade as the same dataset contract.
    """
    obs = getattr(adata, "obs", None)
    if obs is None:
        raise ValueError("dataset_contract_hash requires an AnnData-like object with .obs.")
    n_obs = int(getattr(adata, "n_obs", len(obs)))

    def values_for(key: str, *, missing: str) -> list[str]:
        if key not in obs.columns:
            return [missing] * n_obs
        series = obs[key].astype("string").fillna(missing)
        if key == scenario_key:
            series = series.replace("", missing)
        return series.astype(str).tolist()

    fields: list[tuple[str, Sequence[Any]]] = [
        ("cell_id", list(getattr(adata, "obs_names", getattr(obs, "index", [])))),
        ("label", values_for(label_key, missing="unknown")),
        ("batch", values_for(batch_key, missing="unknown")),
        ("scenario", values_for(scenario_key, missing="non_rare")),
        ("feature_id", list(getattr(adata, "var_names", []))),
    ]
    h = hashlib.sha256()
    h.update(b"scrarebench-dataset-contract-v1\0")
    h.update(str(dataset_key).encode("utf-8")); h.update(b"\0")
    h.update(str(n_obs).encode("ascii")); h.update(b"\0")
    for field_name, values in fields:
        h.update(field_name.encode("utf-8")); h.update(b"\0")
        h.update(str(len(values)).encode("ascii")); h.update(b"\0")
        for value in values:
            encoded = str(value).encode("utf-8")
            h.update(len(encoded).to_bytes(8, "big")); h.update(encoded)
    return h.hexdigest()


def make_run_id(*, method_name: str, dataset_fingerprint: str, method_seed: int | None, config_hash: str, latent_hash: str | None = None) -> str:
    material = "|".join([
        str(method_name), str(dataset_fingerprint), "none" if method_seed is None else str(method_seed), str(config_hash), str(latent_hash or "")
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def mean_sd(values: Iterable[Any]) -> dict[str, Any]:
    nums = [float(v) for v in values if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v))]
    if not nums:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0, "values": []}
    arr = np.asarray(nums, dtype=float)
    return {
        "mean": float(arr.mean()),
        "sd": float(arr.std(ddof=1)) if len(arr) >= 2 else None,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(len(arr)),
        "values": [float(x) for x in arr],
    }


def consensus(values: Iterable[Any]) -> dict[str, Any]:
    """Return strict-majority categorical consensus without alphabetical tie claims.

    ``label`` is populated only when the leading category has >50% support. The
    leading category is still exposed as ``plurality_label`` for diagnostics, but
    exact ties and non-majority pluralities are explicitly non-consensus.
    """
    vals = [str(v) for v in values if v is not None and str(v) not in {"", "nan", "None"}]
    if not vals:
        return {
            "label": None, "plurality_label": None, "agreement": None,
            "n": 0, "counts": {}, "consensus": False, "is_tie": False,
        }
    counts = Counter(vals)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    plurality_label, count = ranked[0]
    top_ties = sum(1 for _, n in ranked if n == count)
    agreement = float(count / len(vals))
    has_consensus = bool(agreement > 0.5 and top_ties == 1)
    return {
        "label": plurality_label if has_consensus else None,
        "plurality_label": plurality_label,
        "agreement": agreement,
        "n": len(vals),
        "counts": dict(sorted(counts.items())),
        "consensus": has_consensus,
        "is_tie": top_ties > 1,
    }


def _table_rows(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, Mapping) and isinstance(table.get("rows"), list):
        return [dict(x) for x in table["rows"] if isinstance(x, Mapping)]
    if isinstance(table, list):
        return [dict(x) for x in table if isinstance(x, Mapping)]
    return []


def _aggregate_row_group(rows: Sequence[Mapping[str, Any]], identity_keys: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in identity_keys:
        vals = [r.get(key) for r in rows if r.get(key) not in (None, "")]
        out[key] = vals[0] if vals else None
    all_keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    for key in all_keys:
        if key in identity_keys:
            continue
        vals = [r.get(key) for r in rows]
        stats = mean_sd(vals)
        if stats["n"]:
            out[key] = stats["mean"]
            out[f"{key}__sd"] = stats["sd"]
            out[f"{key}__n"] = stats["n"]
            out[f"{key}__min"] = stats["min"]
            out[f"{key}__max"] = stats["max"]
            out[f"{key}__values"] = stats["values"]
        elif key in {"failure_archetype", "failure_archetype_v2", "distribution", "topology", "scenario"}:
            c = consensus(vals)
            if c["n"]:
                out[key] = c["label"]
                out[f"{key}__plurality_label"] = c["plurality_label"]
                out[f"{key}__agreement"] = c["agreement"]
                out[f"{key}__counts"] = c["counts"]
                out[f"{key}__consensus"] = c["consensus"]
                out[f"{key}__is_tie"] = c["is_tie"]
    return out


def _group_rows(runs: Sequence[Mapping[str, Any]], path: Sequence[str], identity_keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        node: Any = run.get("payload", run)
        for part in path:
            if not isinstance(node, Mapping):
                node = None
                break
            node = node.get(part)
        for row in _table_rows(node):
            ident = tuple(row.get(k) for k in identity_keys)
            grouped[ident].append(row)
    return [_aggregate_row_group(rows, identity_keys) for _, rows in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def aggregate_dashboard_runs(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate included single-run dashboard payloads without averaging embeddings.

    The returned object contains metric/table summaries only. UMAP coordinates,
    Sankey flows, cluster IDs and cell-level sandbox state intentionally remain
    seed-specific and are never averaged.
    """
    included = [r for r in runs if bool(r.get("included", True))]
    return {
        "n_stored": len(runs),
        "n_included": len(included),
        "included_run_ids": [r.get("run_id") for r in included],
        "included_seeds": [r.get("method_seed") for r in included],
        "excluded_seeds": [r.get("method_seed") for r in runs if not bool(r.get("included", True))],
        "metrics_subset": _group_rows(included, ["metrics", "subset"], ["subset"]),
        "metrics_per_type": _group_rows(included, ["metrics", "per_type"], ["cell_type"]),
        "rare_summary": _group_rows(included, ["rare", "summary"], ["metric", "metric_type"]),
        "rare_per_type": _group_rows(included, ["rare", "per_type"], ["cell_type", "scenario", "distribution", "topology"]),
        "rare_scenarios": _group_rows(included, ["rare", "scenarios"], ["scenario", "distribution", "topology"]),
        "resolution_sensitivity": _group_rows(included, ["rare", "resolution_sensitivity"], ["resolution", "cell_type", "scenario"]),
        "scib_metrics": _group_rows(included, ["scib", "metrics"], ["metric"]),
        "scib_aggregates": _group_rows(included, ["scib", "aggregates"], ["metric"]),
    }


def canonicalize_embedded_run(run: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of an embedded dashboard run with a canonical config hash.

    scRareBench 0.10.0 accidentally included realized clustering outcomes in
    ``configuration_hash``.  Consequently, valid multi-seed reports could carry
    different hashes solely because (for example) Leiden produced 22 clusters
    for one seed and 27 for another.  When reproducibility metadata is embedded,
    recompute the identity from that metadata using the current canonical hash
    semantics.  This also makes 0.10.1 able to merge already-generated 0.10.0
    per-seed HTML reports without retraining or reevaluation.

    Runs without parseable embedded run configuration are returned unchanged so
    legacy reports retain their existing compatibility semantics.
    """
    out = deepcopy(dict(run))
    payload = out.get("payload")
    if not isinstance(payload, Mapping):
        return out
    reproducibility = payload.get("reproducibility")
    if not isinstance(reproducibility, Mapping):
        return out
    raw_config = reproducibility.get("run_config")
    if not isinstance(raw_config, str) or not raw_config.strip():
        return out
    try:
        import yaml
        benchmark_cfg = yaml.safe_load(raw_config) or {}
    except Exception:
        return out
    if not isinstance(benchmark_cfg, Mapping):
        return out

    meta = payload.get("meta") if isinstance(payload.get("meta"), Mapping) else {}
    method_cfg = out.get("method_configuration")
    if not isinstance(method_cfg, Mapping):
        method_cfg = meta.get("method_configuration") if isinstance(meta, Mapping) else None
    if not isinstance(method_cfg, Mapping):
        candidate = benchmark_cfg.get("method_config")
        method_cfg = candidate if isinstance(candidate, Mapping) else {}

    old_hash = out.get("configuration_hash")
    new_hash = evaluation_contract_hash(benchmark_cfg, method_cfg)
    training_hash = method_training_hash(method_cfg)
    benchmark_seed = benchmark_cfg.get("benchmark_seed", benchmark_cfg.get("random_state"))
    out["configuration_hash"] = new_hash
    out["evaluation_contract_hash"] = new_hash
    out["method_training_hash"] = training_hash
    if benchmark_seed is not None:
        out["benchmark_seed"] = int(benchmark_seed)
    out["method_configuration"] = dict(method_cfg)

    # Keep payload metadata internally consistent.
    if isinstance(payload, dict):
        payload_meta = payload.setdefault("meta", {})
        if isinstance(payload_meta, dict):
            payload_meta["configuration_hash"] = new_hash
            payload_meta["evaluation_contract_hash"] = new_hash
            payload_meta["method_training_hash"] = training_hash
            if benchmark_seed is not None:
                payload_meta["benchmark_seed"] = int(benchmark_seed)
            payload_meta["method_configuration"] = dict(method_cfg)

    # Generated run IDs include configuration_hash.  Regenerate only when the
    # current ID matches the deterministic ID implied by the old hash; preserve
    # explicit/custom IDs supplied by callers.
    method_name = out.get("method_name")
    dataset_fingerprint = out.get("dataset_fingerprint")
    if method_name is not None and dataset_fingerprint is not None:
        current_id = out.get("run_id")
        old_derived = None
        if old_hash is not None:
            old_derived = make_run_id(
                method_name=str(method_name),
                dataset_fingerprint=str(dataset_fingerprint),
                method_seed=out.get("method_seed"),
                config_hash=str(old_hash),
                latent_hash=out.get("latent_sha256"),
            )
        if current_id is None or current_id == old_derived:
            new_id = make_run_id(
                method_name=str(method_name),
                dataset_fingerprint=str(dataset_fingerprint),
                method_seed=out.get("method_seed"),
                config_hash=new_hash,
                latent_hash=out.get("latent_sha256"),
            )
            out["run_id"] = new_id
            if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
                payload["meta"]["run_id"] = new_id
    return out


def validate_compatible_run(existing_runs: Sequence[Mapping[str, Any]], new_run: Mapping[str, Any]) -> None:
    """Validate that a run can be aggregated with existing runs.

    Compatibility is intentionally strict: method, cell universe/order and
    seed-independent method configuration must match. Duplicate method seeds are
    rejected even if the run IDs differ.
    """
    if not existing_runs:
        return
    first = existing_runs[0]
    fields = ["method_name", "dataset_fingerprint", "configuration_hash"]
    # Newer reports carry stronger contracts. Compare them whenever both runs
    # provide the field, while retaining backward compatibility for old reports.
    for optional_field in ("dataset_contract_sha256", "evaluation_contract_hash", "benchmark_seed"):
        if first.get(optional_field) is not None and new_run.get(optional_field) is not None:
            fields.append(optional_field)
    for field in fields:
        a, b = first.get(field), new_run.get(field)
        if a is not None and b is not None and str(a) != str(b):
            raise ValueError(f"Incompatible run: {field} differs ({a!r} != {b!r}).")
    seed = new_run.get("method_seed")
    if seed is not None and any(r.get("method_seed") == seed for r in existing_runs):
        raise ValueError(f"Duplicate method seed {seed}; the same seed cannot be added twice.")
    run_id = new_run.get("run_id")
    if run_id and any(r.get("run_id") == run_id for r in existing_runs):
        raise ValueError(f"Duplicate run_id {run_id}; this run is already stored.")


def make_multirun_container(runs: Sequence[Mapping[str, Any]], *, expected_seeds: Sequence[int] | None = None, title: str | None = None) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for raw in runs:
        run = canonicalize_embedded_run(raw)
        validate_compatible_run(normalized, run)
        run.setdefault("included", True)
        normalized.append(run)
    # Canonicalize run order so the same run set produces byte-stable aggregate
    # statistics and report payloads regardless of import order.
    normalized.sort(
        key=lambda r: (
            r.get("method_seed") is None,
            int(r.get("method_seed")) if r.get("method_seed") is not None else 0,
            str(r.get("run_id") or ""),
        )
    )
    expected = normalize_method_seeds(expected_seeds) if expected_seeds is not None else [r["method_seed"] for r in normalized if r.get("method_seed") is not None]
    return {
        "report_type": "scrarebench_multi_run",
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_by": {"package": "scrarebench", "version": __version__},
        "title": title or "scRareBench multi-seed interactive report",
        "expected_seeds": expected,
        "runs": normalized,
        "aggregate": aggregate_dashboard_runs(normalized),
        "edit_history": [],
        "notes": {
            "embedding_aggregation": "UMAP/latent coordinates, Sankey flows and cluster assignments are seed-specific and are never averaged.",
            "sd_single_seed": "SD is missing/not-applicable when fewer than two included runs provide a metric; it is never encoded as zero.",
            "exclusion_semantics": "Excluding a run changes aggregate display only; stored run data remain embedded and can be restored.",
        },
    }


def extract_embedded_report_payload(path_or_text: str | Path) -> dict[str, Any]:
    """Extract v0.10 JSON-script payload or legacy v0.9.x ``const P=...`` payload."""
    path = Path(path_or_text) if isinstance(path_or_text, Path) or (isinstance(path_or_text, str) and "<" not in path_or_text[:200]) else None
    text = path.read_text(encoding="utf-8") if path is not None and path.exists() else str(path_or_text)
    marker = '<script id="scrarebench-report-data" type="application/json">'
    start = text.find(marker)
    if start >= 0:
        start += len(marker)
        end = text.find("</script>", start)
        if end < 0:
            raise ValueError("Embedded scRareBench report JSON did not terminate.")
        return json.loads(text[start:end])
    legacy = "const P="
    start = text.find(legacy)
    if start < 0:
        raise ValueError("Could not find an embedded scRareBench report payload.")
    i = start + len(legacy)
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        raise ValueError("Malformed legacy scRareBench payload.")
    obj_start = i
    depth = 0
    in_string = False
    escaped = False
    for j in range(i, len(text)):
        ch = text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[obj_start:j + 1])
    raise ValueError("Legacy scRareBench payload did not terminate.")
