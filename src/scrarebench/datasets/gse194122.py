from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from ..exceptions import DatasetValidationError, MissingDependencyError
from ..scenarios import annotate_paper_scenarios
from ..utils import sha256_file, sha256_strings, write_json

GSE194122_SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/"
    "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
)
SOURCE_GZ_NAME = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz"
SOURCE_H5AD_NAME = "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad"
PAPER_MAIN_H5AD_NAME = "gse194122_paper_main.h5ad"

PAPER_SCENARIO_COLUMNS = (
    "scrarebench_scenario",
    "scrarebench_distribution",
    "scrarebench_topology",
    "scrarebench_parent_type",
    "scrarebench_curation_source",
    "scrarebench_is_rare",
)

# This follows the executed scDML_modify notebook. The listed batches are retained;
# target-cell instances in all other batches are removed.
PAPER_KEEP_BATCHES: dict[str, tuple[str, ...]] = {
    "pDC": ("s3d7", "s3d6", "s2d1"),
    "CD8+ T CD57+ CD45RA+": ("s3d6", "s4d8", "s1d3"),
}

EXPECTED_ORIGINAL_CELLS = 90_261
EXPECTED_PAPER_MAIN_CELLS = 89_199
EXPECTED_REMOVED_CELLS = 1_062
EXPECTED_BATCHES = 12


def _require_anndata() -> Any:
    try:
        import anndata as ad
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "Dataset loading requires anndata. Install the package dependencies."
        ) from exc
    return ad


def download_gse194122(
    cache_dir: str | Path,
    *,
    force: bool = False,
    url: str = GSE194122_SOURCE_URL,
) -> Path:
    """Download and decompress the original GSE194122 processed AnnData file."""
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    gz_path = cache / SOURCE_GZ_NAME
    h5ad_path = cache / SOURCE_H5AD_NAME

    if force or not gz_path.exists():
        urllib.request.urlretrieve(url, gz_path)
    if force or not h5ad_path.exists():
        with gzip.open(gz_path, "rb") as source, h5ad_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return h5ad_path


def load_gse194122(path: str | Path):
    """Load the source file and normalize only benchmark metadata column names."""
    ad = _require_anndata()
    adata = ad.read_h5ad(Path(path))
    rename_map: dict[str, str] = {}
    if "batch" in adata.obs.columns and "BATCH" not in adata.obs.columns:
        rename_map["batch"] = "BATCH"
    if "cell_type" in adata.obs.columns and "celltype" not in adata.obs.columns:
        rename_map["cell_type"] = "celltype"
    if rename_map:
        adata.obs.rename(columns=rename_map, inplace=True)
    for required in ("BATCH", "celltype"):
        if required not in adata.obs.columns:
            raise DatasetValidationError(
                f"Expected adata.obs['{required}']; available columns: {list(adata.obs.columns)}"
            )
        adata.obs[required] = adata.obs[required].astype(str).str.strip()
    return adata


def _validate_gse194122_paper_main(
    adata: Any,
    *,
    batch_key: str = "BATCH",
    label_key: str = "celltype",
    strict_expected_counts: bool = True,
) -> None:
    """Validate that an AnnData object is the package-ready paper benchmark.

    This validation concerns benchmark construction only. It intentionally does
    not require or create normalization, HVGs, PCA, neighbors, or any other
    method-specific preprocessing.
    """
    required_obs = {batch_key, label_key, *PAPER_SCENARIO_COLUMNS}
    missing = sorted(required_obs.difference(adata.obs.columns))
    if missing:
        raise DatasetValidationError(
            "Prepared GSE194122 benchmark is missing required obs columns: "
            f"{missing}. Rebuild it with force_rebuild=True."
        )
    if not adata.obs_names.is_unique:
        raise DatasetValidationError("Prepared GSE194122 benchmark has duplicate cell barcodes.")
    if strict_expected_counts:
        expected = {
            "modified cells": (adata.n_obs, EXPECTED_PAPER_MAIN_CELLS),
            "batches": (adata.obs[batch_key].nunique(), EXPECTED_BATCHES),
        }
        failures = [
            f"{name}: observed {observed}, expected {wanted}"
            for name, (observed, wanted) in expected.items()
            if observed != wanted
        ]
        if failures:
            raise DatasetValidationError("; ".join(failures))


def load_gse194122_benchmark(
    data_dir: str | Path,
    *,
    cache_dir: str | Path | None = None,
    benchmark_filename: str = PAPER_MAIN_H5AD_NAME,
    force_download: bool = False,
    force_rebuild: bool = False,
    strict_expected_counts: bool = True,
    url: str = GSE194122_SOURCE_URL,
):
    """Return the package-ready GSE194122 paper benchmark AnnData.

    This is the recommended high-level dataset API. It combines the two manual
    dataset stages:

    1. download/decompress the original GSE194122 AnnData when needed;
    2. construct the 89,199-cell paper benchmark by benchmark-specific cell
       subsetting and six-scenario annotation.

    The returned object is *not* normalized or otherwise preprocessed for an
    integration method. In particular, this function performs no gene filtering,
    normalization, log transform, HVG selection, scaling, PCA, neighborhood
    construction, or dimensionality reduction. Those steps remain the
    responsibility of the selected integration method (for example scVI,
    Harmony, Scanorama, or a Scanpy workflow).

    Existing prepared data are reused by default. Set ``force_rebuild=True`` to
    rebuild the benchmark; ``force_download=True`` only affects a download that
    is needed as part of such a build.
    """
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).expanduser().resolve() if cache_dir is not None else root / "cache"
    output = root / benchmark_filename

    if force_rebuild or not output.exists():
        source = download_gse194122(cache, force=force_download, url=url)
        prepared, _ = prepare_gse194122_paper_main(
            source,
            output,
            strict_expected_counts=strict_expected_counts,
            overwrite=force_rebuild and output.exists(),
        )
        _validate_gse194122_paper_main(
            prepared,
            strict_expected_counts=strict_expected_counts,
        )
        return prepared

    prepared = load_gse194122(output)
    _validate_gse194122_paper_main(
        prepared,
        strict_expected_counts=strict_expected_counts,
    )
    return prepared


def build_paper_main_mask(
    obs: pd.DataFrame,
    *,
    batch_key: str = "BATCH",
    label_key: str = "celltype",
    keep_batches: dict[str, tuple[str, ...]] | None = None,
) -> pd.Series:
    """Return an order-preserving mask for the manuscript's modified dataset."""
    if batch_key not in obs.columns or label_key not in obs.columns:
        raise DatasetValidationError(
            f"obs must contain '{batch_key}' and '{label_key}'."
        )
    rules = keep_batches or PAPER_KEEP_BATCHES
    keep = pd.Series(True, index=obs.index, dtype=bool)
    labels = obs[label_key].astype(str).str.strip()
    batches = obs[batch_key].astype(str).str.strip()
    for cell_type, retained_batches in rules.items():
        target = labels.eq(cell_type)
        keep.loc[target & ~batches.isin(retained_batches)] = False
    return keep


def _paper_distribution_summary(
    obs: pd.DataFrame,
    *,
    batch_key: str,
    label_key: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_cells = len(obs)
    total_batches = obs[batch_key].nunique()
    for cell_type, retained in PAPER_KEEP_BATCHES.items():
        subset = obs.loc[obs[label_key].eq(cell_type)]
        rows.append(
            {
                "cell_type": cell_type,
                "n_cells": len(subset),
                "global_abundance_pct": 100.0 * len(subset) / total_cells,
                "present_batches": subset[batch_key].nunique(),
                "total_batches": total_batches,
                "batch_fraction_pct": 100.0 * subset[batch_key].nunique() / total_batches,
                "expected_retained_batches": ",".join(retained),
                "observed_batches": ",".join(sorted(subset[batch_key].unique())),
            }
        )
    return pd.DataFrame(rows)


def prepare_gse194122_paper_main(
    source_path: str | Path,
    output_path: str | Path,
    *,
    batch_key: str = "BATCH",
    label_key: str = "celltype",
    strict_expected_counts: bool = True,
    overwrite: bool = False,
):
    """Create the 89,199-cell benchmark dataset without gene preprocessing.

    The original cell order is retained. Only target-cell instances outside the
    manuscript's retained batches are removed. All original features, layers,
    obsm entries and raw data are otherwise left as supplied by the source file.
    """
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")

    adata = load_gse194122(source)
    original_n = adata.n_obs
    original_order = adata.obs_names.astype(str).tolist()
    original_order_hash = sha256_strings(original_order)

    mask = build_paper_main_mask(adata.obs, batch_key=batch_key, label_key=label_key)
    modified = adata[mask.to_numpy()].copy()
    annotate_paper_scenarios(modified, label_key=label_key, inplace=True)

    removed_n = original_n - modified.n_obs
    if strict_expected_counts:
        expected = {
            "original cells": (original_n, EXPECTED_ORIGINAL_CELLS),
            "modified cells": (modified.n_obs, EXPECTED_PAPER_MAIN_CELLS),
            "removed cells": (removed_n, EXPECTED_REMOVED_CELLS),
            "batches": (modified.obs[batch_key].nunique(), EXPECTED_BATCHES),
        }
        failures = [
            f"{name}: observed {observed}, expected {wanted}"
            for name, (observed, wanted) in expected.items()
            if observed != wanted
        ]
        if failures:
            raise DatasetValidationError("; ".join(failures))

    distribution = _paper_distribution_summary(
        modified.obs, batch_key=batch_key, label_key=label_key
    )
    for _, row in distribution.iterrows():
        if int(row["present_batches"]) != 3:
            raise DatasetValidationError(
                f"{row['cell_type']} is present in {row['present_batches']} batches, expected 3."
            )
        if float(row["global_abundance_pct"]) <= 1.0:
            raise DatasetValidationError(
                f"{row['cell_type']} abundance is not >1% after modification."
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    modified.write_h5ad(output)
    barcodes_path = output.with_suffix(".cell_order.npy")
    import numpy as np

    np.save(barcodes_path, modified.obs_names.astype(str).to_numpy())
    distribution_path = output.with_suffix(".distribution.csv")
    distribution.to_csv(distribution_path, index=False)

    manifest = {
        "benchmark_id": "gse194122_paper_main_v1",
        "source_url": GSE194122_SOURCE_URL,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "source_n_obs": original_n,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "output_n_obs": modified.n_obs,
        "output_n_vars": modified.n_vars,
        "removed_n_obs": removed_n,
        "batch_key": batch_key,
        "label_key": label_key,
        "cell_order_sha256": sha256_strings(modified.obs_names.astype(str).tolist()),
        "source_cell_order_sha256": original_order_hash,
        "paper_keep_batches": {key: list(value) for key, value in PAPER_KEEP_BATCHES.items()},
        "preprocessing_applied": False,
        "modification": "cell subsetting only; no normalization, filtering, scaling, HVG selection, or dimensionality reduction",
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    return modified, manifest
