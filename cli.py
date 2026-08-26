from __future__ import annotations

import argparse

import numpy as np

from .datasets import download_dataset, download_gse194122, list_datasets, prepare_gse194122_paper_main
from .evaluation import EvaluationConfig, evaluate_latent
from .latent import attach_latent, load_latent
from .scib_backend import ScibEvaluationConfig


def _read_h5ad(path: str):
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit("anndata is required. Install scrarebench dependencies.") from exc
    return ad.read_h5ad(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scrarebench")
    commands = parser.add_subparsers(dest="command", required=True)

    datasets = commands.add_parser("datasets", help="List the six registered datasets")

    download = commands.add_parser("download-dataset", help="Download/construct a registered dataset")
    download.add_argument("selector", help="Dataset index 0..5 or registered name/alias")
    download.add_argument("--data-dir", required=True)
    download.add_argument("--force-download", action="store_true")
    download.add_argument("--force-rebuild", action="store_true")
    download.add_argument("--no-strict-counts", action="store_true")

    prepare = commands.add_parser("prepare-gse194122", help="Download and create the paper-main benchmark dataset")
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--force-download", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--no-strict-counts", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate a user-supplied latent space")
    evaluate.add_argument("--adata", required=True)
    evaluate.add_argument("--latent", required=True)
    evaluate.add_argument("--latent-type", default="auto", choices=["auto", "npy", "npz", "csv", "tsv"])
    evaluate.add_argument("--latent-key")
    evaluate.add_argument("--latent-barcodes")
    evaluate.add_argument("--method", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--representation-key")
    evaluate.add_argument("--batch-key", default="BATCH")
    evaluate.add_argument("--label-key", default="celltype")
    evaluate.add_argument("--resolution", type=float, default=1.0)
    evaluate.add_argument("--resolution-sweep", type=float, nargs="*", default=[1.0])
    evaluate.add_argument("--n-neighbors", type=int, default=15)
    evaluate.add_argument("--metric", default="euclidean")
    evaluate.add_argument("--random-state", type=int, default=0)
    evaluate.add_argument("--allow-reorder", action="store_true")
    evaluate.add_argument("--save-evaluated-adata")

    evaluate.add_argument("--skip-scib", action="store_true", help="Skip the standard scIB-compatible layer")
    evaluate.add_argument("--scib-count-layer", default="counts")
    evaluate.add_argument("--scib-n-hvg", type=int, default=4000)
    evaluate.add_argument("--scib-reference-n-pcs", type=int, default=50)
    evaluate.add_argument("--scib-n-jobs", type=int, default=1)
    evaluate.add_argument("--scib-solver", default="arpack", choices=["arpack", "randomized", "auto"])
    evaluate.add_argument("--scib-no-progress", action="store_true")
    evaluate.add_argument("--scib-no-silhouette-batch", action="store_true")
    evaluate.add_argument(
        "--allow-scib-failure",
        action="store_true",
        help="Continue rare-cell evaluation if the optional scIB backend fails.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "datasets":
        for row in list_datasets():
            edited = "edited benchmark" if row["modified"] else "unmodified"
            print(f"{row['index']}: {row['key']} — {row['display_name']} [{edited}]")
        return 0

    if args.command == "download-dataset":
        path = download_dataset(
            args.selector,
            args.data_dir,
            force_download=args.force_download,
            force_rebuild=args.force_rebuild,
            strict_expected_counts=not args.no_strict_counts,
        )
        print(path)
        return 0

    if args.command == "prepare-gse194122":
        source = download_gse194122(args.cache_dir, force=args.force_download)
        _, manifest = prepare_gse194122_paper_main(
            source,
            args.output,
            strict_expected_counts=not args.no_strict_counts,
            overwrite=args.overwrite,
        )
        print(f"Prepared: {args.output}")
        print(f"Cells: {manifest['output_n_obs']}")
        return 0

    adata = _read_h5ad(args.adata)
    latent, embedded_barcodes = load_latent(args.latent, source_type=args.latent_type, key=args.latent_key)
    external_barcodes = np.load(args.latent_barcodes, allow_pickle=False).astype(str) if args.latent_barcodes else None
    barcodes = external_barcodes if external_barcodes is not None else embedded_barcodes
    representation_key = args.representation_key or f"X_{args.method}"
    alignment = attach_latent(
        adata,
        latent,
        key=representation_key,
        latent_barcodes=barcodes,
        allow_reorder=args.allow_reorder,
    )
    result = evaluate_latent(
        adata,
        EvaluationConfig(
            method_name=args.method,
            representation_key=representation_key,
            label_key=args.label_key,
            batch_key=args.batch_key,
            reference_resolution=args.resolution,
            resolution_sweep=tuple(args.resolution_sweep),
            n_neighbors=args.n_neighbors,
            distance_metric=args.metric,
            random_state=args.random_state,
            scib=ScibEvaluationConfig(
                enabled=not args.skip_scib,
                count_layer=args.scib_count_layer or None,
                n_hvg=args.scib_n_hvg,
                reference_n_pcs=args.scib_reference_n_pcs,
                n_jobs=args.scib_n_jobs,
                progress_bar=not args.scib_no_progress,
                solver=args.scib_solver,
                include_silhouette_batch=not args.scib_no_silhouette_batch,
                require_backend=not args.allow_scib_failure,
            ),
        ),
        args.output_dir,
    )
    print(f"Alignment: {alignment}")
    print(f"Report: {result.files['report']}")
    if result.scib is not None:
        print("scIB-compatible aggregate scores:")
        print(result.scib.aggregate_scores.to_string(index=False))
    if args.save_evaluated_adata:
        adata.write_h5ad(args.save_evaluated_adata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
