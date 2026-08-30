from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .constants import DEFAULT_BENCHMARK_SEED
from .exceptions import MissingDependencyError
from .utils import slugify


@dataclass(frozen=True)
class ClusteringResult:
    method_name: str
    representation_key: str
    neighbors_key: str
    cluster_keys: dict[float, str]
    n_neighbors: int
    metric: str
    random_state: int
    leiden_flavor: str
    leiden_n_iterations: int


def _require_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("Clustering requires scanpy.") from exc
    return sc


def run_standard_clustering(
    adata: Any,
    *,
    representation_key: str,
    method_name: str,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    resolutions: Iterable[float] = (1.0,),
    random_state: int = DEFAULT_BENCHMARK_SEED,
    leiden_flavor: str = "igraph",
    leiden_n_iterations: int = 2,
    overwrite: bool = False,
) -> ClusteringResult:
    """Build an isolated neighbor graph and deterministic Leiden clusters.

    scRareBench no longer silently switches Leiden implementations.  The exact
    flavor and iteration count are part of the benchmark contract and are written
    to run metadata so two environments cannot produce different partitions under
    an apparently identical configuration.
    """
    sc = _require_scanpy()
    if representation_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{representation_key}'] is missing")
    slug = slugify(method_name)
    neighbors_key = f"scrarebench_neighbors_{slug}"
    if neighbors_key in adata.uns and not overwrite:
        raise KeyError(f"Neighbor graph '{neighbors_key}' already exists")
    sc.pp.neighbors(
        adata,
        use_rep=representation_key,
        n_neighbors=n_neighbors,
        metric=metric,
        key_added=neighbors_key,
        random_state=random_state,
    )
    cluster_keys: dict[float, str] = {}
    for resolution in resolutions:
        value = float(resolution)
        token = str(value).replace(".", "p")
        cluster_key = f"scrarebench_leiden_{slug}_r{token}"
        if cluster_key in adata.obs and not overwrite:
            raise KeyError(f"Cluster key '{cluster_key}' already exists")
        sc.tl.leiden(
            adata,
            flavor=leiden_flavor,
            n_iterations=int(leiden_n_iterations),
            resolution=value,
            random_state=random_state,
            key_added=cluster_key,
            neighbors_key=neighbors_key,
        )
        cluster_keys[value] = cluster_key
    return ClusteringResult(
        method_name=method_name,
        representation_key=representation_key,
        neighbors_key=neighbors_key,
        cluster_keys=cluster_keys,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        leiden_flavor=leiden_flavor,
        leiden_n_iterations=int(leiden_n_iterations),
    )
