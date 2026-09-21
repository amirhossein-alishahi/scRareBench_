from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..exceptions import DatasetValidationError


PREPARATION_UNS_KEY = "scrarebench_dataset_preparation"

EXTERNAL_BENCHMARK_CONTRACTS: dict[str, dict[str, Any]] = {
    "wu_breast_cancer_atlas": {
        "label_key": "celltype_subset",
        "batch_key": "donor_id",
        "source_n_obs": 100_064,
        "analysis_n_obs": 100_064,
        "n_vars": 28_468,
        "n_labels": 49,
        "n_batches": 26,
        "count_source": "raw_aligned",
    },
    "covid19_autoimmunity_pbmc": {
        "label_key": "cell_type",
        "batch_key": "donor_id",
        "source_n_obs": 97_499,
        "analysis_n_obs": 97_499,
        "n_vars": 21_858,
        "n_labels": 26,
        "n_batches": 22,
        "count_source": "raw_aligned",
    },
    "nygc_seurat_v4_pbmc": {
        "label_key": "celltype.l2",
        "batch_key": "orig.ident",
        "source_n_obs": 161_764,
        "analysis_n_obs": 152_145,
        "n_vars": 20_729,
        "n_labels": 30,
        "n_batches": 24,
        "count_source": "X",
        "retained_cell_order_sha256": "830dd8a9ebb828d39dbd8cacd6a426edaf62385e9df4009424edc930589b3ec0",
    },
}


def _materialize_matrix_chunk(chunk: Any) -> Any:
    if hasattr(chunk, "to_memory"):
        return chunk.to_memory()
    return chunk


def _validate_identifiers(adata: Any, *, dataset_key: str) -> None:
    if not adata.obs_names.is_unique:
        raise DatasetValidationError(f"Dataset {dataset_key!r} has non-unique cell identifiers.")
    if not adata.var_names.is_unique:
        raise DatasetValidationError(f"Dataset {dataset_key!r} has non-unique feature identifiers.")


def _validate_shape(
    adata: Any,
    *,
    dataset_key: str,
    expected_n_obs: int,
    expected_n_vars: int,
    stage: str,
) -> None:
    observed = (int(adata.n_obs), int(adata.n_vars))
    expected = (int(expected_n_obs), int(expected_n_vars))
    if observed != expected:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} {stage} shape changed: observed {observed}, expected {expected}. "
            "The registered benchmark contract must be re-audited before continuing."
        )


def _validate_obs_contract(
    adata: Any,
    *,
    dataset_key: str,
    label_key: str,
    batch_key: str,
    expected_n_labels: int,
    expected_n_batches: int,
) -> None:
    for key in (label_key, batch_key):
        if key not in adata.obs.columns:
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} requires adata.obs[{key!r}] for its benchmark contract."
            )
        values = adata.obs[key]
        missing = int(values.isna().sum())
        empty = int(values.astype("string").fillna("").str.strip().eq("").sum())
        if missing or empty:
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} adata.obs[{key!r}] contains "
                f"{missing} missing and {empty} empty values."
            )

    n_labels = int(adata.obs[label_key].astype(str).nunique())
    n_batches = int(adata.obs[batch_key].astype(str).nunique())
    if n_labels != int(expected_n_labels):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} label cardinality changed: observed {n_labels}, "
            f"expected {expected_n_labels} for {label_key!r}."
        )
    if n_batches != int(expected_n_batches):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} batch cardinality changed: observed {n_batches}, "
            f"expected {expected_n_batches} for {batch_key!r}."
        )


def _validate_count_matrix(
    matrix: Any,
    *,
    dataset_key: str,
    row_chunk: int = 4096,
) -> dict[str, Any]:
    n_obs, n_vars = matrix.shape
    library = np.empty(int(n_obs), dtype=np.float64)
    finite = True
    nonnegative = True
    integer_like = True
    observed_min = np.inf
    observed_max = -np.inf

    for start in range(0, int(n_obs), int(row_chunk)):
        end = min(int(n_obs), start + int(row_chunk))
        chunk = _materialize_matrix_chunk(matrix[start:end, :])
        values = chunk.data if sp.issparse(chunk) else np.asarray(chunk).reshape(-1)
        values = np.asarray(values)
        if values.size:
            finite_mask = np.isfinite(values)
            if not finite_mask.all():
                finite = False
            checked = values[finite_mask]
            if checked.size:
                observed_min = min(observed_min, float(checked.min()))
                observed_max = max(observed_max, float(checked.max()))
                if np.any(checked < 0):
                    nonnegative = False
                if not np.allclose(checked, np.rint(checked), atol=1e-6, rtol=0.0):
                    integer_like = False
        library[start:end] = np.asarray(chunk.sum(axis=1)).reshape(-1)

    finite = bool(finite and np.isfinite(library).all())
    zero_library = int(np.count_nonzero(library <= 0))
    diagnostics = {
        "shape": [int(n_obs), int(n_vars)],
        "finite": finite,
        "nonnegative": bool(nonnegative),
        "integer_like": bool(integer_like),
        "minimum": float(observed_min) if np.isfinite(observed_min) else 0.0,
        "maximum": float(observed_max) if np.isfinite(observed_max) else 0.0,
        "zero_or_negative_library_cells": zero_library,
        "library_min": float(np.min(library)),
        "library_median": float(np.median(library)),
        "library_max": float(np.max(library)),
    }
    if not (finite and nonnegative and integer_like and zero_library == 0):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} canonical raw-count matrix failed validation: {diagnostics}"
        )
    return diagnostics


def _aligned_raw_counts(adata: Any, *, dataset_key: str):
    if adata.raw is None:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} requires adata.raw.X as its canonical raw-count source."
        )
    raw_names = pd.Index(adata.raw.var_names.astype(str))
    current_names = pd.Index(adata.var_names.astype(str))
    if not raw_names.is_unique:
        raise DatasetValidationError(f"Dataset {dataset_key!r} raw feature identifiers are not unique.")
    positions = raw_names.get_indexer(current_names)
    missing = int(np.count_nonzero(positions < 0))
    if missing:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} has {missing} current features absent from adata.raw.var_names."
        )
    matrix = adata.raw.X[:, positions]
    if matrix.shape != adata.shape:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} aligned raw-count shape {matrix.shape} "
            f"does not match AnnData shape {adata.shape}."
        )
    return matrix, {
        "raw_shape": [int(adata.raw.n_obs), int(adata.raw.n_vars)],
        "same_var_order_as_current": bool(raw_names.equals(current_names)),
        "current_vars_missing_in_raw": missing,
    }


def _prepare_raw_aligned_dataset(adata: Any, *, dataset_key: str):
    contract = EXTERNAL_BENCHMARK_CONTRACTS[dataset_key]
    _validate_identifiers(adata, dataset_key=dataset_key)
    _validate_shape(
        adata,
        dataset_key=dataset_key,
        expected_n_obs=contract["source_n_obs"],
        expected_n_vars=contract["n_vars"],
        stage="source",
    )
    _validate_obs_contract(
        adata,
        dataset_key=dataset_key,
        label_key=contract["label_key"],
        batch_key=contract["batch_key"],
        expected_n_labels=contract["n_labels"],
        expected_n_batches=contract["n_batches"],
    )
    matrix, alignment = _aligned_raw_counts(adata, dataset_key=dataset_key)
    diagnostics = _validate_count_matrix(matrix, dataset_key=dataset_key)
    # The downloaded H5AD remains unchanged; this canonical layer exists only
    # on the in-memory AnnData returned by load_dataset().
    adata.layers["counts"] = matrix
    adata.obs[contract["batch_key"]] = pd.Categorical(
        adata.obs[contract["batch_key"]].astype(str)
    )
    adata.uns[PREPARATION_UNS_KEY] = {
        "dataset_key": dataset_key,
        "count_source": "adata.raw.X aligned to adata.var_names",
        "count_layer": "counts",
        "source_cells": int(adata.n_obs),
        "analysis_cells": int(adata.n_obs),
        "raw_alignment": alignment,
        "count_diagnostics": diagnostics,
        "source_h5ad_modified": False,
    }
    return adata


def _protein_qc_summaries(protein: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(protein, pd.DataFrame):
        array = protein.to_numpy()
        return np.asarray(array.sum(axis=1)).reshape(-1), np.asarray((array > 0).sum(axis=1)).reshape(-1)
    if sp.issparse(protein):
        return (
            np.asarray(protein.sum(axis=1)).reshape(-1),
            np.asarray(protein.getnnz(axis=1)).reshape(-1),
        )
    array = np.asarray(protein)
    return np.asarray(array.sum(axis=1)).reshape(-1), np.asarray((array > 0).sum(axis=1)).reshape(-1)


def _nygc_official_qc_mask(adata: Any, *, row_chunk: int = 4096) -> tuple[np.ndarray, dict[str, int]]:
    if "celltype.l2" not in adata.obs.columns:
        raise DatasetValidationError("NYGC PBMC QC requires adata.obs['celltype.l2'].")
    if "protein_counts" not in adata.obsm:
        raise DatasetValidationError("NYGC PBMC QC requires adata.obsm['protein_counts'].")

    protein_library, proteins_detected = _protein_qc_summaries(adata.obsm["protein_counts"])
    protein_log_library = np.full(protein_library.shape, -np.inf, dtype=float)
    positive = protein_library > 0
    protein_log_library[positive] = np.log(protein_library[positive])

    mt_mask = np.asarray(adata.var_names.astype(str).str.startswith("MT-"))
    total_counts = np.empty(int(adata.n_obs), dtype=np.float64)
    mitochondrial_counts = np.zeros(int(adata.n_obs), dtype=np.float64)
    for start in range(0, int(adata.n_obs), int(row_chunk)):
        end = min(int(adata.n_obs), start + int(row_chunk))
        chunk = _materialize_matrix_chunk(adata.X[start:end, :])
        total_counts[start:end] = np.asarray(chunk.sum(axis=1)).reshape(-1)
        if mt_mask.any():
            mitochondrial_counts[start:end] = np.asarray(chunk[:, mt_mask].sum(axis=1)).reshape(-1)

    pct_mito = np.divide(
        mitochondrial_counts * 100.0,
        total_counts,
        out=np.full_like(total_counts, np.inf),
        where=total_counts > 0,
    )
    doublet = adata.obs["celltype.l2"].astype("string").eq("Doublet").to_numpy()
    mask = (
        (protein_log_library > 7.6)
        & (protein_log_library < 10.3)
        & (proteins_detected > 150)
        & ~doublet
        & (pct_mito < 12.0)
    )
    reasons = {
        "source_cells": int(adata.n_obs),
        "retained_cells": int(mask.sum()),
        "removed_cells": int((~mask).sum()),
        "protein_log_library_le_7_6": int(np.count_nonzero(protein_log_library <= 7.6)),
        "protein_log_library_ge_10_3": int(np.count_nonzero(protein_log_library >= 10.3)),
        "proteins_detected_le_150": int(np.count_nonzero(proteins_detected <= 150)),
        "doublet_label": int(np.count_nonzero(doublet)),
        "pct_mito_ge_12": int(np.count_nonzero(pct_mito >= 12.0)),
    }
    return np.asarray(mask, dtype=bool), reasons


def _ordered_cell_hash(names: Any) -> str:
    return hashlib.sha256("\0".join(map(str, names)).encode("utf-8")).hexdigest()


def prepare_nygc_seurat_v4_pbmc(adata: Any):
    dataset_key = "nygc_seurat_v4_pbmc"
    contract = EXTERNAL_BENCHMARK_CONTRACTS[dataset_key]
    _validate_identifiers(adata, dataset_key=dataset_key)
    _validate_shape(
        adata,
        dataset_key=dataset_key,
        expected_n_obs=contract["source_n_obs"],
        expected_n_vars=contract["n_vars"],
        stage="source",
    )
    for key in (contract["label_key"], contract["batch_key"]):
        if key not in adata.obs.columns:
            raise DatasetValidationError(
                f"Dataset {dataset_key!r} requires adata.obs[{key!r}] before canonical QC."
            )

    mask, qc = _nygc_official_qc_mask(adata)
    if int(mask.sum()) != int(contract["analysis_n_obs"]):
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} official QC retained {int(mask.sum())} cells; "
            f"expected {contract['analysis_n_obs']}. Re-audit the source/QC contract."
        )
    retained_names = adata.obs_names[mask]
    retained_hash = _ordered_cell_hash(retained_names)
    if retained_hash != contract["retained_cell_order_sha256"]:
        raise DatasetValidationError(
            f"Dataset {dataset_key!r} official-QC cell identity/order changed: "
            f"observed {retained_hash}, expected {contract['retained_cell_order_sha256']}."
        )

    target = adata[mask, :].copy()
    _validate_identifiers(target, dataset_key=dataset_key)
    _validate_shape(
        target,
        dataset_key=dataset_key,
        expected_n_obs=contract["analysis_n_obs"],
        expected_n_vars=contract["n_vars"],
        stage="post-QC",
    )
    _validate_obs_contract(
        target,
        dataset_key=dataset_key,
        label_key=contract["label_key"],
        batch_key=contract["batch_key"],
        expected_n_labels=contract["n_labels"],
        expected_n_batches=contract["n_batches"],
    )
    diagnostics = _validate_count_matrix(target.X, dataset_key=dataset_key)
    target.obs[contract["batch_key"]] = pd.Categorical(
        target.obs[contract["batch_key"]].astype(str)
    )
    target.uns[PREPARATION_UNS_KEY] = {
        "dataset_key": dataset_key,
        "count_source": "adata.X",
        "count_layer": None,
        "source_cells": int(qc["source_cells"]),
        "analysis_cells": int(target.n_obs),
        "qc": qc,
        "retained_cell_order_sha256": retained_hash,
        "count_diagnostics": diagnostics,
        "source_h5ad_modified": False,
    }
    return target


def prepare_external_benchmark_dataset(adata: Any, *, dataset_key: str):
    key = str(dataset_key)
    if key == "wu_breast_cancer_atlas":
        return _prepare_raw_aligned_dataset(adata, dataset_key=key)
    if key == "covid19_autoimmunity_pbmc":
        return _prepare_raw_aligned_dataset(adata, dataset_key=key)
    if key == "nygc_seurat_v4_pbmc":
        return prepare_nygc_seurat_v4_pbmc(adata)
    return adata


__all__ = [
    "EXTERNAL_BENCHMARK_CONTRACTS",
    "PREPARATION_UNS_KEY",
    "prepare_external_benchmark_dataset",
    "prepare_nygc_seurat_v4_pbmc",
]
