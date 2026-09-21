from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from ..exceptions import DatasetValidationError


_PREPARATION_UNS_KEY = "scrarebench_preparation"

_DATASET3_KEY = "wu_breast_cancer_atlas"
_DATASET4_KEY = "covid19_autoimmunity_pbmc"
_DATASET5_KEY = "nygc_seurat_v4_pbmc"

_EXPECTED_SOURCE_SHAPES: dict[str, tuple[int, int]] = {
    _DATASET3_KEY: (100_064, 28_468),
    _DATASET4_KEY: (97_499, 21_858),
    _DATASET5_KEY: (161_764, 20_729),
}

_EXPECTED_ANALYSIS_CONTRACTS: dict[str, dict[str, Any]] = {
    _DATASET3_KEY: {
        "n_obs": 100_064,
        "n_vars": 28_468,
        "label_key": "celltype_subset",
        "n_labels": 49,
        "batch_key": "donor_id",
        "n_batches": 26,
    },
    _DATASET4_KEY: {
        "n_obs": 97_499,
        "n_vars": 21_858,
        "label_key": "cell_type",
        "n_labels": 26,
        "batch_key": "donor_id",
        "n_batches": 22,
    },
    _DATASET5_KEY: {
        "n_obs": 152_145,
        "n_vars": 20_729,
        "label_key": "celltype.l2",
        "n_labels": 30,
        "batch_key": "orig.ident",
        "n_batches": 24,
    },
}

_DATASET5_RETAINED_CELL_SHA256 = (
    "830dd8a9ebb828d39dbd8cacd6a426edaf62385e9df4009424edc930589b3ec0"
)


def _require_in_memory(adata: Any, dataset_key: str) -> None:
    if bool(getattr(adata, "isbacked", False)):
        raise ValueError(
            f"Benchmark-ready loading for {dataset_key!r} requires backed=None. "
            "Use download_dataset(...) plus anndata.read_h5ad(..., backed=...) "
            "when raw backed access is required."
        )


def _validate_source_shape(adata: Any, dataset_key: str) -> None:
    expected = _EXPECTED_SOURCE_SHAPES[dataset_key]
    observed = (int(adata.n_obs), int(adata.n_vars))
    if observed != expected:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} source shape changed: observed {observed}, "
            f"expected audited shape {expected}."
        )


def _validate_obs_contract(
    adata: Any,
    *,
    dataset_key: str,
    label_key: str,
    batch_key: str,
    expected_labels: int,
    expected_batches: int,
) -> None:
    for key in (label_key, batch_key):
        if key not in adata.obs.columns:
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} requires obs[{key!r}]."
            )
        series = adata.obs[key]
        empty = series.astype("string").fillna("").str.strip().eq("")
        if bool(series.isna().any()) or bool(empty.any()):
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} obs[{key!r}] contains missing/empty values."
            )

    if not adata.obs_names.is_unique:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} requires unique obs_names."
        )
    if not adata.var_names.is_unique:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} requires unique var_names."
        )

    n_labels = int(adata.obs[label_key].astype(str).nunique())
    n_batches = int(adata.obs[batch_key].astype(str).nunique())
    if n_labels != int(expected_labels):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} label cardinality changed: "
            f"observed {n_labels}, expected {expected_labels}."
        )
    if n_batches != int(expected_batches):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} batch cardinality changed: "
            f"observed {n_batches}, expected {expected_batches}."
        )


def _full_count_diagnostics(matrix: Any, *, row_chunk: int = 4096) -> dict[str, Any]:
    """Validate the complete matrix without densifying the full dataset."""
    n_obs, n_vars = matrix.shape
    library = np.empty(int(n_obs), dtype=np.float64)
    finite = True
    nonnegative = True
    integer_like = True
    observed_min = np.inf
    observed_max = -np.inf
    sparse_seen = False
    implicit_zero_seen = False

    for start in range(0, int(n_obs), row_chunk):
        end = min(int(n_obs), start + row_chunk)
        chunk = matrix[start:end, :]
        if hasattr(chunk, "to_memory"):
            chunk = chunk.to_memory()

        if sparse.issparse(chunk):
            sparse_seen = True
            values = np.asarray(chunk.data)
            implicit_zero_seen = (
                implicit_zero_seen
                or int(chunk.nnz) < int(chunk.shape[0] * chunk.shape[1])
            )
            row_sum = np.asarray(chunk.sum(axis=1)).reshape(-1)
        else:
            array = np.asarray(chunk)
            values = array.reshape(-1)
            row_sum = np.asarray(array.sum(axis=1)).reshape(-1)

        library[start:end] = row_sum
        if values.size:
            if not np.isfinite(values).all():
                finite = False
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                observed_min = min(observed_min, float(finite_values.min()))
                observed_max = max(observed_max, float(finite_values.max()))
                if np.any(finite_values < 0):
                    nonnegative = False
                if not np.allclose(
                    finite_values, np.rint(finite_values), atol=1e-6, rtol=0.0
                ):
                    integer_like = False

    if not np.isfinite(library).all():
        finite = False
    if implicit_zero_seen:
        observed_min = min(observed_min, 0.0)

    return {
        "shape": [int(n_obs), int(n_vars)],
        "sparse": bool(sparse_seen),
        "finite": bool(finite),
        "nonnegative": bool(nonnegative),
        "integer_like": bool(integer_like),
        "minimum": float(observed_min) if np.isfinite(observed_min) else 0.0,
        "maximum": float(observed_max) if np.isfinite(observed_max) else 0.0,
        "zero_or_negative_library_cells": int(np.count_nonzero(library <= 0)),
        "library_min": float(np.min(library)),
        "library_median": float(np.median(library)),
        "library_max": float(np.max(library)),
    }


def _require_count_like(
    matrix: Any, *, dataset_key: str, source: str
) -> dict[str, Any]:
    diag = _full_count_diagnostics(matrix)
    if not (
        diag["finite"]
        and diag["nonnegative"]
        and diag["integer_like"]
        and diag["zero_or_negative_library_cells"] == 0
    ):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} canonical count source {source!r} "
            f"failed full count diagnostics: {diag}"
        )
    return diag


def _aligned_raw_counts(adata: Any, *, dataset_key: str):
    if adata.raw is None:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} requires adata.raw for canonical counts."
        )

    raw_names = pd.Index(adata.raw.var_names.astype(str))
    current_names = pd.Index(adata.var_names.astype(str))
    if not raw_names.is_unique:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} raw.var_names are not unique."
        )

    if len(raw_names) == len(current_names) and raw_names.equals(current_names):
        counts = adata.raw.X
    else:
        positions = raw_names.get_indexer(current_names)
        missing = int(np.count_nonzero(positions < 0))
        if missing:
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} has {missing} current features absent from adata.raw."
            )
        counts = adata.raw.X[:, positions]
    if counts.shape != adata.shape:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} aligned raw counts have shape {counts.shape}, "
            f"expected {adata.shape}."
        )
    diagnostics = _require_count_like(
        counts,
        dataset_key=dataset_key,
        source="adata.raw.X aligned to adata.var_names",
    )
    return counts, diagnostics


def _prepare_raw_aligned_dataset(adata: Any, *, dataset_key: str) -> Any:
    contract = _EXPECTED_ANALYSIS_CONTRACTS[dataset_key]
    _validate_source_shape(adata, dataset_key)
    _validate_obs_contract(
        adata,
        dataset_key=dataset_key,
        label_key=str(contract["label_key"]),
        batch_key=str(contract["batch_key"]),
        expected_labels=int(contract["n_labels"]),
        expected_batches=int(contract["n_batches"]),
    )
    counts, diagnostics = _aligned_raw_counts(adata, dataset_key=dataset_key)
    adata.layers["counts"] = counts
    adata.uns[_PREPARATION_UNS_KEY] = {
        "dataset_key": dataset_key,
        "source_n_obs": int(adata.n_obs),
        "analysis_n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "canonical_count_source": "adata.raw.X aligned to adata.var_names",
        "count_layer": "counts",
        "cell_filter": "none",
        "count_diagnostics": diagnostics,
        "source_contract": "audited_2026_09_21",
    }
    return adata


def _pbmc_official_qc_mask(
    adata: Any, *, row_chunk: int = 4096
) -> tuple[np.ndarray, dict[str, int]]:
    if "celltype.l2" not in adata.obs.columns:
        raise DatasetValidationError(
            "Dataset 'nygc_seurat_v4_pbmc' requires obs['celltype.l2']."
        )
    if "protein_counts" not in adata.obsm:
        raise DatasetValidationError(
            "Dataset 'nygc_seurat_v4_pbmc' requires obsm['protein_counts']."
        )

    protein = adata.obsm["protein_counts"]
    if isinstance(protein, pd.DataFrame):
        protein_values = protein.to_numpy()
        protein_library = protein_values.sum(axis=1)
        proteins_detected = (protein_values > 0).sum(axis=1)
    elif sparse.issparse(protein):
        protein_library = np.asarray(protein.sum(axis=1)).reshape(-1)
        proteins_detected = np.asarray((protein > 0).sum(axis=1)).reshape(-1)
    else:
        protein_values = np.asarray(protein)
        protein_library = protein_values.sum(axis=1)
        proteins_detected = (protein_values > 0).sum(axis=1)

    protein_log_library = np.full(protein_library.shape, -np.inf, dtype=float)
    positive_library = protein_library > 0
    protein_log_library[positive_library] = np.log(protein_library[positive_library])

    mt_mask = np.asarray(adata.var_names.astype(str).str.startswith("MT-"))
    total_counts = np.empty(adata.n_obs, dtype=np.float64)
    mitochondrial_counts = np.zeros(adata.n_obs, dtype=np.float64)
    for start in range(0, adata.n_obs, row_chunk):
        end = min(adata.n_obs, start + row_chunk)
        chunk = adata.X[start:end, :]
        if hasattr(chunk, "to_memory"):
            chunk = chunk.to_memory()
        total_counts[start:end] = np.asarray(chunk.sum(axis=1)).reshape(-1)
        if mt_mask.any():
            mitochondrial_counts[start:end] = np.asarray(
                chunk[:, mt_mask].sum(axis=1)
            ).reshape(-1)

    pct_mito = np.divide(
        mitochondrial_counts * 100.0,
        total_counts,
        out=np.full_like(total_counts, np.inf, dtype=float),
        where=total_counts > 0,
    )

    labels = adata.obs["celltype.l2"].astype("string")
    mask = (
        (protein_log_library > 7.6)
        & (protein_log_library < 10.3)
        & (proteins_detected > 150)
        & labels.ne("Doublet").to_numpy()
        & (pct_mito < 12.0)
    )

    reasons = {
        "source_cells": int(adata.n_obs),
        "retained_cells": int(mask.sum()),
        "removed_cells": int((~mask).sum()),
        "protein_log_library_le_7_6": int(np.count_nonzero(protein_log_library <= 7.6)),
        "protein_log_library_ge_10_3": int(np.count_nonzero(protein_log_library >= 10.3)),
        "proteins_detected_le_150": int(np.count_nonzero(proteins_detected <= 150)),
        "doublet_label": int(labels.eq("Doublet").sum()),
        "pct_mito_ge_12": int(np.count_nonzero(pct_mito >= 12.0)),
    }
    return np.asarray(mask, dtype=bool), reasons


def _prepare_nygc_pbmc(adata: Any) -> Any:
    dataset_key = _DATASET5_KEY
    _validate_source_shape(adata, dataset_key)
    source_count_diagnostics = _require_count_like(
        adata.X, dataset_key=dataset_key, source="adata.X"
    )

    mask, reasons = _pbmc_official_qc_mask(adata)
    filtered = adata[mask, :].copy()
    contract = _EXPECTED_ANALYSIS_CONTRACTS[dataset_key]
    observed_shape = (int(filtered.n_obs), int(filtered.n_vars))
    expected_shape = (int(contract["n_obs"]), int(contract["n_vars"]))
    if observed_shape != expected_shape:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} official QC contract changed: "
            f"observed {observed_shape}, expected {expected_shape}."
        )

    retained_ids = filtered.obs_names.astype(str).tolist()
    retained_hash = hashlib.sha256(
        "\0".join(retained_ids).encode("utf-8")
    ).hexdigest()
    if retained_hash != _DATASET5_RETAINED_CELL_SHA256:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} official-QC retained-cell identity/order changed: "
            f"observed SHA256 {retained_hash}, expected {_DATASET5_RETAINED_CELL_SHA256}."
        )

    _validate_obs_contract(
        filtered,
        dataset_key=dataset_key,
        label_key=str(contract["label_key"]),
        batch_key=str(contract["batch_key"]),
        expected_labels=int(contract["n_labels"]),
        expected_batches=int(contract["n_batches"]),
    )
    analysis_count_diagnostics = _require_count_like(
        filtered.X, dataset_key=dataset_key, source="adata.X after official QC"
    )

    filtered.uns[_PREPARATION_UNS_KEY] = {
        "dataset_key": dataset_key,
        **reasons,
        "analysis_n_obs": int(filtered.n_obs),
        "n_vars": int(filtered.n_vars),
        "canonical_count_source": "adata.X after official scvi-tools PBMC QC",
        "count_layer": "",
        "cell_filter": "official_scvi_tools_pbmc_qc",
        "retained_cell_order_sha256": retained_hash,
        "source_count_diagnostics": source_count_diagnostics,
        "analysis_count_diagnostics": analysis_count_diagnostics,
        "source_contract": "audited_2026_09_21",
    }
    return filtered


def prepare_builtin_benchmark_dataset(adata: Any, *, dataset_key: str) -> Any:
    """Prepare only the audited external benchmark contracts for datasets 3--5.

    This function performs dataset-specific source validation, canonical raw-count
    exposure, and the fixed Dataset 5 source-cell QC. It never normalizes,
    log-transforms, selects HVGs, scales, computes PCA/neighbors, or runs an
    integration method.
    """
    key = str(dataset_key)
    if key not in {_DATASET3_KEY, _DATASET4_KEY, _DATASET5_KEY}:
        raise KeyError(
            "prepare_builtin_benchmark_dataset is defined only for "
            "wu_breast_cancer_atlas, covid19_autoimmunity_pbmc, and "
            "nygc_seurat_v4_pbmc."
        )
    _require_in_memory(adata, key)

    if key in {_DATASET3_KEY, _DATASET4_KEY}:
        return _prepare_raw_aligned_dataset(adata, dataset_key=key)
    return _prepare_nygc_pbmc(adata)


__all__ = ["prepare_builtin_benchmark_dataset"]
