from __future__ import annotations

import json
import re
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..exceptions import DatasetValidationError
from ..utils import sha256_file, write_json
from .gse194122 import (
    GSE194122_SOURCE_URL,
    PAPER_MAIN_H5AD_NAME,
    SOURCE_H5AD_NAME,
    download_gse194122,
    load_gse194122_benchmark,
    prepare_gse194122_paper_main,
)

CELLXGENE_API_BASE = "https://api.cellxgene.cziscience.com/curation/v1"
CELLXGENE_DATA_BASE = "https://datasets.cellxgene.cziscience.com"

NYGC_SEURAT_V4_URL = (
    "https://scverse-exampledata.s3.eu-west-1.amazonaws.com/anndata/"
    "pbmc_seurat_v4.h5ad"
)
NYGC_SEURAT_V4_SHA256 = (
    "c3b0100a6ce27beb64eff53692e09f98da2a58cfdfea08d15ff204f834b41396"
)


@dataclass(frozen=True)
class DatasetSpec:
    """One selectable dataset exposed by :func:`download_dataset`/`load_dataset`."""

    index: int
    key: str
    display_name: str
    aliases: tuple[str, ...]
    source_kind: str
    modified: bool
    filename: str
    source_url: str | None = None
    collection_id: str | None = None
    preferred_title: str | None = None
    exclude_title_terms: tuple[str, ...] = ()
    sha256: str | None = None
    note: str = ""


DATASET_REGISTRY: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        index=0,
        key="gse194122",
        display_name="GSE194122 paper benchmark (six-scenario edited)",
        aliases=(
            "gse194122_benchmark",
            "gse194122_edited",
            "paper_main",
            "main",
        ),
        source_kind="gse194122_benchmark",
        modified=True,
        filename=PAPER_MAIN_H5AD_NAME,
        source_url=GSE194122_SOURCE_URL,
        note=(
            "Downloads the original GSE194122 source when needed, then applies only "
            "the benchmark-specific cell subsetting and six-scenario annotation."
        ),
    ),
    DatasetSpec(
        index=1,
        key="gse194122_raw",
        display_name="GSE194122 original/unmodified",
        aliases=(
            "gse194122_original",
            "gse194122_unmodified",
            "gse194122_source",
            "raw",
        ),
        source_kind="gse194122_raw",
        modified=False,
        filename=SOURCE_H5AD_NAME,
        source_url=GSE194122_SOURCE_URL,
        note="Original source H5AD; no benchmark cell subsetting or scenario annotation.",
    ),
    DatasetSpec(
        index=2,
        key="mbdrc_renal_cortex",
        display_name="mBDRC renal cortex",
        aliases=("mbdrc", "renal_cortex", "mbdrc_renal"),
        source_kind="cellxgene_collection",
        modified=False,
        filename="mbdrc_renal_cortex.h5ad",
        collection_id="4cbb929b-b03b-4aa8-a943-00f61dc22641",
        source_url=(
            "https://cellxgene.cziscience.com/collections/"
            "4cbb929b-b03b-4aa8-a943-00f61dc22641"
        ),
        preferred_title="multimodal benchmarking dataset for renal cortex characterization",
        exclude_title_terms=("multiome only",),
        note=("Main renal-cortex dataset from the collection; the multiome-only companion is excluded. "
              "load_dataset attaches provisional registered GR/LE/SR × DL/RM metadata in memory."),
    ),
    DatasetSpec(
        index=3,
        key="wu_breast_cancer_atlas",
        display_name="Wu breast-cancer atlas",
        aliases=(
            "wu",
            "wu_breast",
            "wu_breast_cancer",
            "breast_cancer_atlas",
            "gse176078",
        ),
        source_kind="cellxgene_collection",
        modified=False,
        filename="wu_breast_cancer_atlas.h5ad",
        collection_id="dea97145-f712-431c-a223-6b5f565f362a",
        source_url=(
            "https://cellxgene.cziscience.com/collections/"
            "dea97145-f712-431c-a223-6b5f565f362a"
        ),
        preferred_title="breast",
        note=("Published CELLxGENE H5AD from the Wu et al. breast-cancer collection. "
              "load_dataset attaches provisional registered GR/LE/SR × DL/RM metadata in memory."),
    ),
    DatasetSpec(
        index=4,
        key="covid19_autoimmunity_pbmc",
        display_name="COVID-19 autoimmunity PBMC",
        aliases=(
            "covid19",
            "covid",
            "covid_pbmc",
            "covid19_pbmc",
            "covid_autoimmunity",
        ),
        source_kind="cellxgene_collection",
        modified=False,
        filename="covid19_autoimmunity_pbmc.h5ad",
        collection_id="eb735cc9-d0a7-48fa-b255-db726bf365af",
        source_url=(
            "https://cellxgene.cziscience.com/collections/"
            "eb735cc9-d0a7-48fa-b255-db726bf365af"
        ),
        preferred_title="covid",
        note=("Published CELLxGENE H5AD; no source-cell editing is applied. "
              "load_dataset attaches provisional registered GR/LE/SR × DL/RM metadata in memory."),
    ),
    DatasetSpec(
        index=5,
        key="nygc_seurat_v4_pbmc",
        display_name="NYGC / Seurat v4 CITE-seq PBMC",
        aliases=(
            "nygc",
            "seurat_v4",
            "pbmc_seurat_v4",
            "nygc_pbmc",
            "cite_seq_pbmc",
        ),
        source_kind="direct_h5ad",
        modified=False,
        filename="pbmc_seurat_v4.h5ad",
        source_url=NYGC_SEURAT_V4_URL,
        sha256=NYGC_SEURAT_V4_SHA256,
        note="Official scverse example-data H5AD; no scRareBench-specific editing is applied.",
    ),
)

_BY_INDEX = {spec.index: spec for spec in DATASET_REGISTRY}


def _normalize_selector(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _alias_map() -> dict[str, DatasetSpec]:
    out: dict[str, DatasetSpec] = {}
    for spec in DATASET_REGISTRY:
        for name in (spec.key, spec.display_name, *spec.aliases):
            normalized = _normalize_selector(name)
            if normalized:
                out[normalized] = spec
    return out


_BY_ALIAS = _alias_map()


def resolve_dataset(selector: int | str) -> DatasetSpec:
    """Resolve a dataset by integer index or human-readable name/alias."""
    if isinstance(selector, bool):
        raise ValueError("Boolean values are not valid dataset selectors.")
    if isinstance(selector, int):
        if selector in _BY_INDEX:
            return _BY_INDEX[selector]
        raise ValueError(f"Unknown dataset index {selector}. Valid indices are 0..5.")

    text = str(selector).strip()
    if text.isdigit():
        return resolve_dataset(int(text))
    normalized = _normalize_selector(text)
    if normalized in _BY_ALIAS:
        return _BY_ALIAS[normalized]
    choices = ", ".join(f"{s.index}:{s.key}" for s in DATASET_REGISTRY)
    raise ValueError(f"Unknown dataset selector {selector!r}. Available datasets: {choices}")


def list_datasets() -> list[dict[str, Any]]:
    """Return the six supported dataset entries as serializable dictionaries."""
    rows: list[dict[str, Any]] = []
    for spec in DATASET_REGISTRY:
        row = asdict(spec)
        row["aliases"] = list(spec.aliases)
        row["exclude_title_terms"] = list(spec.exclude_title_terms)
        rows.append(row)
    return rows


def _fetch_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "scRareBench dataset-downloader"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _download_url(url: str, destination: Path, *, force: bool = False, timeout: int = 120) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return destination

    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "scRareBench dataset-downloader"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return destination


def _dataset_title(dataset: dict[str, Any]) -> str:
    return str(dataset.get("title") or dataset.get("dataset_title") or "").strip()


def _dataset_cell_count(dataset: dict[str, Any]) -> int:
    for key in ("cell_count", "dataset_total_cell_count", "cell_count_estimate"):
        value = dataset.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return -1


def _select_cellxgene_dataset(spec: DatasetSpec, datasets: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(datasets)
    if not candidates:
        raise DatasetValidationError(
            f"CELLxGENE collection {spec.collection_id} contains no datasets."
        )

    excluded = tuple(term.lower() for term in spec.exclude_title_terms)
    filtered = [
        dataset
        for dataset in candidates
        if not any(term in _dataset_title(dataset).lower() for term in excluded)
    ]
    if filtered:
        candidates = filtered

    preferred = (spec.preferred_title or "").lower().strip()
    if preferred:
        exact = [d for d in candidates if _dataset_title(d).lower() == preferred]
        if exact:
            candidates = exact
        else:
            contains = [d for d in candidates if preferred in _dataset_title(d).lower()]
            if contains:
                candidates = contains

    return max(candidates, key=_dataset_cell_count)


def _find_h5ad_asset(dataset: dict[str, Any]) -> dict[str, Any] | None:
    assets = dataset.get("dataset_assets") or dataset.get("assets") or []
    for asset in assets:
        filetype = str(asset.get("filetype") or asset.get("file_type") or "").upper()
        filename = str(asset.get("filename") or asset.get("name") or "").lower()
        if filetype in {"H5AD", "H5AD_V0_10"} or filename.endswith(".h5ad"):
            return asset
    return None


def _http_url_from_asset(asset: dict[str, Any]) -> str | None:
    for key in (
        "url",
        "download_url",
        "presigned_url",
        "asset_url",
        "file_url",
    ):
        value = asset.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value
    return None


def _cellxgene_download_info(spec: DatasetSpec) -> dict[str, Any]:
    if not spec.collection_id:
        raise ValueError(f"Dataset {spec.key} has no CELLxGENE collection id.")
    collection_api = f"{CELLXGENE_API_BASE}/collections/{spec.collection_id}"
    collection = _fetch_json(collection_api)
    dataset = _select_cellxgene_dataset(spec, collection.get("datasets", []))
    asset = _find_h5ad_asset(dataset)

    url = _http_url_from_asset(asset or {})
    dataset_version_id = dataset.get("dataset_version_id") or dataset.get("version_id")
    if url is None and dataset_version_id:
        url = f"{CELLXGENE_DATA_BASE}/{dataset_version_id}.h5ad"

    # Compatibility fallback for older Discover responses that expose an asset id
    # but not a permanent HTTP URL or dataset-version URL.
    if url is None and asset is not None:
        dataset_id = asset.get("dataset_id") or dataset.get("dataset_id") or dataset.get("id")
        asset_id = asset.get("id") or asset.get("asset_id")
        if dataset_id and asset_id:
            endpoint = (
                "https://api.cellxgene.cziscience.com/dp/v1/datasets/"
                f"{dataset_id}/asset/{asset_id}"
            )
            request = urllib.request.Request(
                endpoint,
                method="POST",
                headers={"User-Agent": "scRareBench dataset-downloader"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            url = payload.get("presigned_url")

    if not url:
        raise DatasetValidationError(
            "Could not resolve an H5AD download URL from CELLxGENE collection "
            f"{spec.collection_id}. Dataset metadata keys: {sorted(dataset)}"
        )

    return {
        "collection_id": spec.collection_id,
        "collection_api": collection_api,
        "collection_name": collection.get("name") or collection.get("title"),
        "dataset_id": dataset.get("dataset_id") or dataset.get("id"),
        "dataset_version_id": dataset_version_id,
        "dataset_title": _dataset_title(dataset),
        "cell_count": _dataset_cell_count(dataset),
        "asset_id": (asset or {}).get("id") or (asset or {}).get("asset_id"),
        "download_url": url,
    }


def _download_external_dataset(
    spec: DatasetSpec,
    root: Path,
    *,
    force_download: bool,
) -> Path:
    destination = root / spec.filename
    manifest_path = destination.with_suffix(destination.suffix + ".source.json")

    if destination.exists() and not force_download:
        if spec.sha256 and sha256_file(destination) != spec.sha256:
            raise DatasetValidationError(
                f"Cached file hash does not match the registered SHA256: {destination}"
            )
        return destination

    if spec.source_kind == "direct_h5ad":
        if not spec.source_url:
            raise ValueError(f"Dataset {spec.key} has no direct source URL.")
        info = {
            "dataset_index": spec.index,
            "dataset_key": spec.key,
            "display_name": spec.display_name,
            "source_kind": spec.source_kind,
            "source_url": spec.source_url,
        }
        _download_url(spec.source_url, destination, force=True)
    elif spec.source_kind == "cellxgene_collection":
        info = _cellxgene_download_info(spec)
        info.update(
            {
                "dataset_index": spec.index,
                "dataset_key": spec.key,
                "display_name": spec.display_name,
                "source_kind": spec.source_kind,
                "collection_page": spec.source_url,
            }
        )
        _download_url(info["download_url"], destination, force=True)
    else:  # pragma: no cover - protected by registry construction
        raise ValueError(f"Unsupported external source kind: {spec.source_kind}")

    observed_hash = sha256_file(destination)
    if spec.sha256 and observed_hash != spec.sha256:
        destination.unlink(missing_ok=True)
        raise DatasetValidationError(
            f"Downloaded {spec.key} but SHA256 validation failed: "
            f"observed {observed_hash}, expected {spec.sha256}."
        )
    info["local_path"] = str(destination)
    info["sha256"] = observed_hash
    info["scrarebench_modification_applied"] = False
    write_json(manifest_path, info)
    return destination


def download_dataset(
    selector: int | str,
    data_dir: str | Path | None = None,
    *,
    force_download: bool = False,
    force_rebuild: bool = False,
    strict_expected_counts: bool = True,
) -> Path:
    """Download/construct one of the six registered datasets and return its H5AD path.

    Selectors 0 and ``"gse194122"`` refer to the scRareBench paper benchmark:
    the original source is downloaded if necessary and the benchmark-only cell
    subsetting + six-scenario annotation is applied. Selector 1 exposes the
    original GSE194122 H5AD without those edits. Selectors 2--5 are downloaded
    exactly as published and receive no scRareBench-specific cell editing or
    method-specific preprocessing.
    """
    spec = resolve_dataset(selector)
    if data_dir is None:
        from .metadata import default_data_dir
        data_dir = default_data_dir()
    root = Path(data_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    if spec.source_kind == "gse194122_benchmark":
        output = root / PAPER_MAIN_H5AD_NAME
        if force_rebuild or not output.exists():
            source = download_gse194122(root / "cache", force=force_download)
            prepare_gse194122_paper_main(
                source,
                output,
                strict_expected_counts=strict_expected_counts,
                overwrite=force_rebuild and output.exists(),
            )
        return output

    if spec.source_kind == "gse194122_raw":
        return download_gse194122(root / "cache", force=force_download)

    return _download_external_dataset(spec, root, force_download=force_download)


def load_dataset(
    selector: int | str,
    data_dir: str | Path | None = None,
    *,
    force_download: bool = False,
    force_rebuild: bool = False,
    strict_expected_counts: bool = True,
    backed: str | None = None,
):
    """Return one of the six registered datasets as an AnnData object.

    No normalization, log transform, HVG selection, scaling, PCA, neighbors, or
    integration-method preprocessing is performed here.

    For registered benchmark datasets with scenario metadata (currently dataset
    0 and external datasets 2--4), six-state annotations are attached in memory
    only. The downloaded source H5AD remains unchanged.
    """
    spec = resolve_dataset(selector)
    if data_dir is None:
        from .metadata import default_data_dir
        data_dir = default_data_dir()
    if spec.source_kind == "gse194122_benchmark" and backed is None:
        adata = load_gse194122_benchmark(
            data_dir, force_download=force_download, force_rebuild=force_rebuild,
            strict_expected_counts=strict_expected_counts,
        )
        from .metadata import attach_builtin_dataset_metadata
        path = Path(data_dir).expanduser().resolve() / PAPER_MAIN_H5AD_NAME
        return attach_builtin_dataset_metadata(adata, dataset_key=spec.key, dataset_index=spec.index,
            display_name=spec.display_name, source_path=path)

    path = download_dataset(
        selector,
        data_dir,
        force_download=force_download,
        force_rebuild=force_rebuild,
        strict_expected_counts=strict_expected_counts,
    )
    from .gse194122 import _require_anndata

    ad = _require_anndata()
    adata = ad.read_h5ad(path, backed=backed)

    # External datasets are kept byte-for-byte as published on disk. Scenario
    # metadata is attached only to the in-memory AnnData returned by load_dataset.
    if spec.key in {
        "mbdrc_renal_cortex", "wu_breast_cancer_atlas", "covid19_autoimmunity_pbmc",
    }:
        from ..scenarios import annotate_registered_scenarios
        annotate_registered_scenarios(adata, dataset_key=spec.key, inplace=True, strict_labels=True)
    from .metadata import attach_builtin_dataset_metadata
    return attach_builtin_dataset_metadata(adata, dataset_key=spec.key, dataset_index=spec.index,
        display_name=spec.display_name, source_path=path)
