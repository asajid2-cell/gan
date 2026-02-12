from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.lab2_data import (
    DEFAULT_MANIFESTS,
    assign_genres,
    genre_count_table,
    load_manifests,
    materialize_genre_samples,
)
from src.lab2_encoder_bridge import FrozenLab1Encoder
from src.lab2_pipeline import (
    apply_lda_projection,
    compose_target_space,
    compute_centroid_distances,
    compute_centroids,
    harvest_embeddings,
    run_exit_audit,
    validate_target_space,
    write_artifacts,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_checkpoint() -> Path:
    root = _repo_root()
    candidates = [
        root / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        root / "saves" / "lab1_run_combo_af_gate" / "latest.pt",
        root / "saves" / "lab1_run_a" / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _default_output_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _repo_root() / "saves" / "lab2_calibration" / f"lab2_{ts}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 2 - Genre Target Vector Space pipeline")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_default_checkpoint(),
        help="Frozen Lab 1 checkpoint path",
    )
    p.add_argument(
        "--manifests-root",
        type=Path,
        default=Path("Z:/DataSets/_lab1_manifests"),
        help="Directory containing cleaned manifests",
    )
    p.add_argument(
        "--manifest-files",
        nargs="*",
        default=DEFAULT_MANIFESTS,
        help="Manifest file names to include",
    )
    p.add_argument(
        "--per-genre-samples",
        type=int,
        default=1000,
        help="Max samples per genre",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=328,
        help="Random seed for sampling/splits",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Artifact output directory",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Small quick run for verification",
    )
    p.add_argument(
        "--silhouette-threshold",
        type=float,
        default=0.45,
        help="Pass threshold for silhouette score",
    )
    p.add_argument(
        "--sigma-multiplier",
        type=float,
        default=3.0,
        help="Multiplier for inter-centroid distance >= k*sigma rule",
    )
    p.add_argument(
        "--neighbor-top-k",
        type=int,
        default=5,
        help="Nearest-neighbor hits required per genre",
    )
    p.add_argument(
        "--stability-sample-fraction",
        type=float,
        default=0.2,
        help="Subset fraction used in centroid stability trials",
    )
    p.add_argument(
        "--stability-trials",
        type=int,
        default=20,
        help="Number of centroid stability trials",
    )
    p.add_argument(
        "--tsne-max-points",
        type=int,
        default=5000,
        help="Max points for t-SNE map",
    )
    p.add_argument(
        "--zstyle-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for 128-D z_style block before projection",
    )
    p.add_argument(
        "--descriptor-weight",
        type=float,
        default=1.0,
        help="Weight multiplier for 32-D descriptor block before projection",
    )
    p.add_argument(
        "--centroid-inlier-fraction",
        type=float,
        default=1.0,
        help="Fraction of closest samples per genre used to compute centroids",
    )
    p.add_argument(
        "--neighbor-min-mean-hits",
        type=float,
        default=5.0,
        help="Pass threshold for mean neighbor hits across genres",
    )
    p.add_argument(
        "--audit-inlier-fraction",
        type=float,
        default=None,
        help="Inlier fraction used for spread/neighbor audits (defaults to centroid-inlier-fraction)",
    )
    p.add_argument(
        "--projection",
        type=str,
        default="raw",
        choices=["raw", "lda"],
        help="Target-space projection mode before centroid/audit",
    )
    p.add_argument(
        "--reuse-artifacts-dir",
        type=Path,
        default=None,
        help="Reuse existing embeddings from a prior run dir (embeddings_index.csv + embeddings.npz)",
    )
    return p.parse_args()


def _load_reused_artifacts(run_dir: Path) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    run_dir = Path(run_dir)
    idx_path = run_dir / "embeddings_index.csv"
    npz_path = run_dir / "embeddings.npz"
    if not idx_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"Expected {idx_path} and {npz_path} when using --reuse-artifacts-dir."
        )
    index_df = pd.read_csv(idx_path)
    z = np.load(npz_path)
    required = ["z_content", "z_style", "descriptor32", "target160", "music_prob"]
    missing = [k for k in required if k not in z]
    if missing:
        raise ValueError(f"embeddings.npz missing keys: {missing}")
    arrays = {
        "z_content": z["z_content"].astype(np.float32),
        "z_style": z["z_style"].astype(np.float32),
        "descriptor32": z["descriptor32"].astype(np.float32),
        "target160": z["target160"].astype(np.float32),
        "music_prob": z["music_prob"].astype(np.float32),
    }
    return index_df, arrays


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.per_genre_samples = min(args.per_genre_samples, 40)
    if args.audit_inlier_fraction is None:
        args.audit_inlier_fraction = float(args.centroid_inlier_fraction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assigned_counts = {}
    sample_counts = {}

    if args.reuse_artifacts_dir is not None:
        print(f"[lab2] reusing embeddings from: {args.reuse_artifacts_dir}")
        index_df, arrays = _load_reused_artifacts(args.reuse_artifacts_dir)
        sample_counts = genre_count_table(index_df)
        assigned_counts = dict(sample_counts)
    else:
        print("[lab2] loading manifests...")
        raw_df = load_manifests(args.manifests_root, args.manifest_files)
        assigned_df = assign_genres(raw_df)
        assigned_counts = genre_count_table(assigned_df)
        print(f"[lab2] assigned genres: {assigned_counts}")

        print("[lab2] sampling per genre...")
        samples_df = materialize_genre_samples(
            assigned_df,
            per_genre_samples=args.per_genre_samples,
            seed=args.seed,
        )
        sample_counts = genre_count_table(samples_df)
        print(f"[lab2] sampled genres: {sample_counts}")
        samples_df.to_csv(args.output_dir / "genre_samples.csv", index=False)

        print(f"[lab2] loading frozen encoder from: {args.checkpoint}")
        encoder = FrozenLab1Encoder(args.checkpoint, device=args.device)

        print("[lab2] harvesting embeddings...")
        index_df, arrays = harvest_embeddings(samples_df, encoder, progress_every=50)

    projection_meta = {"projection": "raw", "input_dim": int(arrays["target160"].shape[1]), "output_dim": int(arrays["target160"].shape[1])}
    if args.projection == "lda":
        print("[lab2] applying supervised LDA projection...")
        arrays["target160"] = compose_target_space(
            arrays=arrays,
            zstyle_weight=args.zstyle_weight,
            descriptor_weight=args.descriptor_weight,
            normalize_rows=True,
        )
        X_proj, projection_meta = apply_lda_projection(index_df=index_df, arrays=arrays, seed=args.seed)
        arrays["target160"] = X_proj
        with (args.output_dir / "projection_meta.json").open("w", encoding="utf-8") as f:
            json.dump(projection_meta, f, indent=2)
    else:
        arrays["target160"] = compose_target_space(
            arrays=arrays,
            zstyle_weight=args.zstyle_weight,
            descriptor_weight=args.descriptor_weight,
            normalize_rows=True,
        )

    print("[lab2] computing centroids + validation...")
    centroids_df = compute_centroids(
        index_df=index_df,
        target160=arrays["target160"],
        inlier_fraction=args.centroid_inlier_fraction,
        inlier_metric="cosine",
    )
    centroid_distances_df = compute_centroid_distances(centroids_df)
    validation = validate_target_space(index_df, arrays, seed=args.seed)

    validation["config"] = {
        "checkpoint": str(args.checkpoint),
        "manifests_root": str(args.manifests_root),
        "manifest_files": list(args.manifest_files),
        "per_genre_samples": int(args.per_genre_samples),
        "seed": int(args.seed),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "projection": str(args.projection),
        "reuse_artifacts_dir": str(args.reuse_artifacts_dir) if args.reuse_artifacts_dir else "",
        "zstyle_weight": float(args.zstyle_weight),
        "descriptor_weight": float(args.descriptor_weight),
        "centroid_inlier_fraction": float(args.centroid_inlier_fraction),
        "neighbor_min_mean_hits": float(args.neighbor_min_mean_hits),
        "audit_inlier_fraction": float(args.audit_inlier_fraction),
    }
    validation["assigned_genre_counts"] = assigned_counts
    validation["sampled_genre_counts"] = sample_counts
    validation["projection_meta"] = projection_meta

    write_artifacts(
        output_dir=args.output_dir,
        index_df=index_df,
        arrays=arrays,
        centroids_df=centroids_df,
        centroid_distances_df=centroid_distances_df,
        validation_summary=validation,
    )
    exit_checklist = run_exit_audit(
        output_dir=args.output_dir,
        index_df=index_df,
        arrays=arrays,
        centroids_df=centroids_df,
        centroid_distances_df=centroid_distances_df,
        validation_summary=validation,
        silhouette_threshold=args.silhouette_threshold,
        sigma_multiplier=args.sigma_multiplier,
        neighbor_top_k=args.neighbor_top_k,
        neighbor_min_mean_hits=args.neighbor_min_mean_hits,
        audit_inlier_fraction=args.audit_inlier_fraction,
        stability_sample_fraction=args.stability_sample_fraction,
        stability_trials=args.stability_trials,
        tsne_max_points=args.tsne_max_points,
        seed=args.seed,
    )
    print(f"[lab2] done. artifacts: {args.output_dir}")
    print(f"[lab2] key metrics: {validation.get('metrics', {})}")
    print(f"[lab2] exit checklist: {exit_checklist.get('lab2_done')}")


if __name__ == "__main__":
    main()
