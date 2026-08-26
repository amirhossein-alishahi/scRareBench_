from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .exceptions import LatentAlignmentError

LatentSourceType = Literal["auto", "npy", "npz", "csv", "tsv", "obsm"]


def load_latent(
    source: Any,
    *,
    source_type: LatentSourceType = "auto",
    key: str | None = None,
    adata: Any | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load latent coordinates from an array, DataFrame, file, or adata.obsm.

    A DataFrame index is interpreted as a barcode vector. NPZ files may contain
    `latent` and optionally `barcodes`; otherwise `key` selects the latent array.
    """
    if isinstance(source, np.ndarray):
        return np.asarray(source), None
    if isinstance(source, pd.DataFrame):
        return source.to_numpy(), source.index.astype(str).to_numpy()
    if source_type == "obsm":
        if adata is None or key is None:
            raise ValueError("source_type='obsm' requires adata and key")
        return np.asarray(adata.obsm[key]), None

    path = Path(source)
    resolved = source_type
    if source_type == "auto":
        suffix = path.suffix.lower()
        resolved = {".npy": "npy", ".npz": "npz", ".csv": "csv", ".tsv": "tsv"}.get(suffix)  # type: ignore[assignment]
        if resolved is None:
            raise ValueError(f"Cannot infer latent source type from: {path}")
    if resolved == "npy":
        return np.load(path, allow_pickle=False), None
    if resolved == "npz":
        archive = np.load(path, allow_pickle=False)
        latent_key = key or ("latent" if "latent" in archive.files else archive.files[0])
        latent = np.asarray(archive[latent_key])
        barcodes = np.asarray(archive["barcodes"]).astype(str) if "barcodes" in archive.files else None
        return latent, barcodes
    if resolved in {"csv", "tsv"}:
        frame = pd.read_csv(path, sep="," if resolved == "csv" else "\t", index_col=0)
        return frame.to_numpy(), frame.index.astype(str).to_numpy()
    raise ValueError(f"Unsupported latent source type: {resolved}")


def validate_and_align_latent(
    latent: np.ndarray,
    obs_names: Any,
    *,
    latent_barcodes: np.ndarray | list[str] | None = None,
    allow_reorder: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(latent)
    expected = np.asarray(obs_names).astype(str)
    if matrix.ndim != 2:
        raise LatentAlignmentError(f"Latent must be 2D, observed shape {matrix.shape}")
    if matrix.shape[0] != len(expected):
        raise LatentAlignmentError(
            f"Latent has {matrix.shape[0]} rows but dataset has {len(expected)} cells."
        )
    if not np.isfinite(matrix).all():
        raise LatentAlignmentError("Latent contains NaN or infinite values.")
    alignment = {
        "n_cells": len(expected),
        "n_dimensions": matrix.shape[1],
        "barcodes_provided": latent_barcodes is not None,
        "reordered": False,
        "exact_order_match": None,
    }
    if latent_barcodes is None:
        alignment["exact_order_match"] = "not_verifiable"
        return matrix, alignment

    supplied = np.asarray(latent_barcodes).astype(str)
    if len(supplied) != len(expected):
        raise LatentAlignmentError(
            f"Barcode vector has {len(supplied)} values but expected {len(expected)}."
        )
    if len(np.unique(supplied)) != len(supplied):
        raise LatentAlignmentError("Latent barcode vector contains duplicates.")
    exact = np.array_equal(supplied, expected)
    alignment["exact_order_match"] = bool(exact)
    if exact:
        return matrix, alignment
    if not allow_reorder:
        mismatch = int(np.flatnonzero(supplied != expected)[0])
        raise LatentAlignmentError(
            f"Latent barcode order differs at row {mismatch}: "
            f"latent='{supplied[mismatch]}', dataset='{expected[mismatch]}'."
        )
    supplied_index = pd.Index(supplied)
    missing = pd.Index(expected).difference(supplied_index)
    extra = supplied_index.difference(pd.Index(expected))
    if len(missing) or len(extra):
        raise LatentAlignmentError(
            f"Barcode sets differ; missing={len(missing)}, extra={len(extra)}."
        )
    positions = supplied_index.get_indexer(expected)
    matrix = matrix[positions]
    alignment["reordered"] = True
    alignment["exact_order_match"] = True
    return matrix, alignment


def attach_latent(
    adata: Any,
    latent: np.ndarray,
    *,
    key: str,
    latent_barcodes: np.ndarray | list[str] | None = None,
    allow_reorder: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if key in adata.obsm and not overwrite:
        raise KeyError(f"adata.obsm['{key}'] already exists")
    aligned, report = validate_and_align_latent(
        latent,
        adata.obs_names,
        latent_barcodes=latent_barcodes,
        allow_reorder=allow_reorder,
    )
    adata.obsm[key] = aligned
    return report
