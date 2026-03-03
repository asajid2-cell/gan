from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from .lab2_encoder_bridge import FrozenLab1Encoder


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def harvest_embeddings(
    samples: pd.DataFrame,
    encoder: FrozenLab1Encoder,
    progress_every: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    records: List[Dict] = []
    zc: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    d32: List[np.ndarray] = []
    t160: List[np.ndarray] = []
    mp: List[float] = []

    for i, row in samples.reset_index(drop=True).iterrows():
        path = Path(str(row["path"]))
        out = encoder.infer_file(path=path, start_sec=0.0)
        if out is None:
            continue

        records.append(
            {
                "sample_id": int(row.get("sample_id", i)),
                "path": str(path),
                "source": str(row["source"]),
                "genre": str(row["genre"]),
                "manifest_file": str(row.get("manifest_file", "")),
                "music_prob": _safe_float(out["music_prob"]),
            }
        )
        zc.append(out["z_content"])
        zs.append(out["z_style"])
        d32.append(out["descriptor32"])
        t160.append(out["target160"])
        mp.append(_safe_float(out["music_prob"]))

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[harvest] processed={i + 1}/{len(samples)} kept={len(records)}")

    if not records:
        raise RuntimeError("No embeddings harvested. Check manifests and checkpoint compatibility.")

    index_df = pd.DataFrame.from_records(records)
    arrays = {
        "z_content": np.stack(zc).astype(np.float32),
        "z_style": np.stack(zs).astype(np.float32),
        "descriptor32": np.stack(d32).astype(np.float32),
        "target160": np.stack(t160).astype(np.float32),
        "music_prob": np.asarray(mp, dtype=np.float32),
    }
    return index_df, arrays


def compose_target_space(
    arrays: Dict[str, np.ndarray],
    zstyle_weight: float = 1.0,
    descriptor_weight: float = 1.0,
    normalize_rows: bool = True,
) -> np.ndarray:
    if "z_style" not in arrays or "descriptor32" not in arrays:
        raise ValueError("arrays must contain 'z_style' and 'descriptor32' for weighted composition.")
    zs = arrays["z_style"].astype(np.float32) * float(zstyle_weight)
    d32 = arrays["descriptor32"].astype(np.float32) * float(descriptor_weight)
    X = np.concatenate([zs, d32], axis=1).astype(np.float32)
    if normalize_rows:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


def compute_centroids(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    inlier_fraction: float = 1.0,
    inlier_metric: str = "cosine",
) -> pd.DataFrame:
    work = index_df.copy().reset_index(drop=True)
    work["row_id"] = work.index.astype(int)
    cent_rows: List[Dict] = []
    for genre, gdf in work.groupby("genre", sort=True):
        X = target160[gdf["row_id"].to_numpy()]
        provisional = X.mean(axis=0).astype(np.float32)
        provisional = provisional / (np.linalg.norm(provisional) + 1e-8)
        n_total = int(len(X))
        n_use = n_total
        if float(inlier_fraction) < 1.0 and n_total >= 4:
            keep_n = max(2, int(round(n_total * float(inlier_fraction))))
            if inlier_metric == "cosine":
                d = 1.0 - cosine_similarity(X, provisional[None, :]).reshape(-1)
            elif inlier_metric == "euclidean":
                d = np.linalg.norm(X - provisional[None, :], axis=1)
            else:
                raise ValueError("inlier_metric must be one of {'cosine','euclidean'}")
            order = np.argsort(d)
            X = X[order[:keep_n]]
            n_use = int(len(X))

        vec = X.mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        rec = {"genre": genre, "n_samples": int(len(gdf)), "n_used": n_use, "inlier_fraction": float(inlier_fraction)}
        for j, v in enumerate(vec):
            rec[f"d{j:03d}"] = float(v)
        cent_rows.append(rec)
    return pd.DataFrame(cent_rows).sort_values("genre").reset_index(drop=True)


def compute_centroid_distances(centroids_df: pd.DataFrame) -> pd.DataFrame:
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    names = centroids_df["genre"].tolist()
    C = centroids_df[dims].to_numpy(dtype=np.float32)
    out_rows: List[Dict] = []
    for i, gi in enumerate(names):
        for j, gj in enumerate(names):
            if j <= i:
                continue
            vi = C[i]
            vj = C[j]
            cos = float(np.dot(vi, vj) / ((np.linalg.norm(vi) * np.linalg.norm(vj)) + 1e-8))
            out_rows.append(
                {
                    "genre_a": gi,
                    "genre_b": gj,
                    "cosine_similarity": cos,
                    "cosine_distance": 1.0 - cos,
                    "euclidean_distance": float(np.linalg.norm(vi - vj)),
                }
            )
    return pd.DataFrame(out_rows).sort_values(["genre_a", "genre_b"]).reset_index(drop=True)


def _select_inlier_indices_by_genre(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    centroids_df: pd.DataFrame,
    inlier_fraction: float = 1.0,
    metric: str = "cosine",
) -> Dict[str, np.ndarray]:
    work = index_df.copy().reset_index(drop=True)
    work["row_id"] = work.index.astype(int)
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    centroid_map = {
        str(row["genre"]): row[dims].to_numpy(dtype=np.float32)
        for _, row in centroids_df.iterrows()
    }
    out: Dict[str, np.ndarray] = {}
    for genre, gdf in work.groupby("genre", sort=True):
        idx = gdf["row_id"].to_numpy()
        X = target160[idx]
        c = centroid_map[genre]
        if float(inlier_fraction) >= 1.0 or len(idx) < 4:
            out[genre] = idx
            continue
        keep_n = max(2, int(round(len(idx) * float(inlier_fraction))))
        if metric == "cosine":
            d = 1.0 - cosine_similarity(X, c[None, :]).reshape(-1)
        elif metric == "euclidean":
            d = np.linalg.norm(X - c[None, :], axis=1)
        else:
            raise ValueError("metric must be one of {'cosine','euclidean'}")
        order = np.argsort(d)
        out[genre] = idx[order[:keep_n]]
    return out


def compute_intra_genre_spread(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    centroids_df: pd.DataFrame,
    metric: str = "cosine",
    inlier_fraction: float = 1.0,
) -> pd.DataFrame:
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    centroid_map = {
        str(row["genre"]): row[dims].to_numpy(dtype=np.float32)
        for _, row in centroids_df.iterrows()
    }
    inliers = _select_inlier_indices_by_genre(
        index_df=index_df,
        target160=target160,
        centroids_df=centroids_df,
        inlier_fraction=inlier_fraction,
        metric=metric,
    )

    work = index_df.copy().reset_index(drop=True)
    work["row_id"] = work.index.astype(int)
    out_rows: List[Dict] = []
    for genre, gdf in work.groupby("genre", sort=True):
        idx = inliers.get(genre, gdf["row_id"].to_numpy())
        X = target160[idx]
        c = centroid_map[genre]
        if metric == "cosine":
            d = 1.0 - cosine_similarity(X, c[None, :]).reshape(-1)
        elif metric == "euclidean":
            d = np.linalg.norm(X - c[None, :], axis=1)
        else:
            raise ValueError("metric must be one of {'cosine','euclidean'}")

        out_rows.append(
            {
                "genre": genre,
                "n_samples": int(len(gdf)),
                "n_used": int(len(X)),
                "mean_distance": float(np.mean(d)),
                "std_distance": float(np.std(d)),
                "var_distance": float(np.var(d)),
                "metric": metric,
                "inlier_fraction": float(inlier_fraction),
            }
        )
    return pd.DataFrame(out_rows).sort_values("genre").reset_index(drop=True)


def evaluate_inter_centroid_separation(
    centroid_distances_df: pd.DataFrame,
    spread_df: pd.DataFrame,
    sigma_multiplier: float = 3.0,
    distance_col: str = "cosine_distance",
) -> Tuple[pd.DataFrame, Dict]:
    spread_map = {
        str(r["genre"]): float(r["std_distance"]) for _, r in spread_df.iterrows()
    }
    rows: List[Dict] = []
    for _, r in centroid_distances_df.iterrows():
        ga = str(r["genre_a"])
        gb = str(r["genre_b"])
        d = float(r[distance_col])
        sigma_ref = max(spread_map.get(ga, 0.0), spread_map.get(gb, 0.0))
        threshold = float(sigma_multiplier) * sigma_ref
        rows.append(
            {
                "genre_a": ga,
                "genre_b": gb,
                "distance_col": distance_col,
                "centroid_distance": d,
                "sigma_ref": sigma_ref,
                "threshold": threshold,
                "pass": bool(d >= threshold),
            }
        )
    out = pd.DataFrame(rows).sort_values(["genre_a", "genre_b"]).reset_index(drop=True)
    summary = {
        "sigma_multiplier": float(sigma_multiplier),
        "distance_col": distance_col,
        "all_pairs_pass": bool(out["pass"].all()) if len(out) else False,
        "pass_rate": float(out["pass"].mean()) if len(out) else 0.0,
    }
    return out, summary


def export_target_centroids_json(centroids_df: pd.DataFrame, out_path: Path) -> None:
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    payload: Dict[str, Dict] = {}
    for _, r in centroids_df.iterrows():
        g = str(r["genre"])
        vector = [float(x) for x in r[dims].to_numpy(dtype=np.float32)]
        payload[g] = {
            "n_samples": int(r["n_samples"]),
            "space_dim": int(len(vector)),
            "vector": vector,
        }
        if len(vector) == 160:
            payload[g]["vector160"] = vector
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def apply_lda_projection(
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    seed: int = 328,
) -> Tuple[np.ndarray, Dict]:
    X = arrays["target160"].astype(np.float32)
    y_text = index_df["genre"].astype(str).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(y_text)
    n_classes = int(len(enc.classes_))
    n_dim = int(min(X.shape[1], max(1, n_classes - 1)))

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    lda = LinearDiscriminantAnalysis(
        n_components=n_dim,
        solver="eigen",
        shrinkage="auto",
    )
    X_lda = lda.fit_transform(Xz, y).astype(np.float32)
    X_lda = X_lda / (np.linalg.norm(X_lda, axis=1, keepdims=True) + 1e-8)

    meta = {
        "projection": "lda",
        "input_dim": int(X.shape[1]),
        "output_dim": int(X_lda.shape[1]),
        "n_classes": n_classes,
        "classes": [str(c) for c in enc.classes_],
        "explained_variance_ratio": [
            float(v) for v in getattr(lda, "explained_variance_ratio_", np.array([], dtype=np.float32))
        ],
        "solver": "eigen",
        "shrinkage": "auto",
        "seed": int(seed),
    }
    return X_lda, meta


def centroid_stability_report(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    centroids_df: pd.DataFrame,
    sample_fraction: float = 0.2,
    n_trials: int = 20,
    seed: int = 328,
) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    full_centroid_map = {
        str(row["genre"]): row[dims].to_numpy(dtype=np.float32)
        for _, row in centroids_df.iterrows()
    }

    work = index_df.copy().reset_index(drop=True)
    work["row_id"] = work.index.astype(int)
    trial_rows: List[Dict] = []
    by_genre = {g: gdf for g, gdf in work.groupby("genre", sort=True)}

    for trial in range(int(n_trials)):
        for genre, gdf in by_genre.items():
            idx = gdf["row_id"].to_numpy()
            if len(idx) == 0:
                continue
            n = max(1, int(round(len(idx) * float(sample_fraction))))
            take = rng.choice(idx, size=n, replace=False)
            sub = target160[take]
            c_sub = sub.mean(axis=0).astype(np.float32)
            c_sub = c_sub / (np.linalg.norm(c_sub) + 1e-8)
            c_full = full_centroid_map[genre]
            cos = float(
                np.dot(c_sub, c_full)
                / ((np.linalg.norm(c_sub) * np.linalg.norm(c_full)) + 1e-8)
            )
            euc = float(np.linalg.norm(c_sub - c_full))
            trial_rows.append(
                {
                    "trial": int(trial),
                    "genre": genre,
                    "n_source_samples": int(len(idx)),
                    "n_sub_samples": int(n),
                    "cosine_similarity_to_full": cos,
                    "euclidean_distance_to_full": euc,
                }
            )

    trial_df = pd.DataFrame(trial_rows).sort_values(["genre", "trial"]).reset_index(drop=True)
    agg = (
        trial_df.groupby("genre")
        .agg(
            trials=("trial", "count"),
            mean_cosine=("cosine_similarity_to_full", "mean"),
            std_cosine=("cosine_similarity_to_full", "std"),
            min_cosine=("cosine_similarity_to_full", "min"),
            mean_euclidean=("euclidean_distance_to_full", "mean"),
            std_euclidean=("euclidean_distance_to_full", "std"),
        )
        .reset_index()
    )
    agg["stable_95_cosine"] = agg["mean_cosine"] >= 0.95
    summary = {
        "sample_fraction": float(sample_fraction),
        "n_trials": int(n_trials),
        "global_mean_cosine": float(trial_df["cosine_similarity_to_full"].mean()) if len(trial_df) else float("nan"),
        "global_min_cosine": float(trial_df["cosine_similarity_to_full"].min()) if len(trial_df) else float("nan"),
        "all_genres_stable_95_cosine": bool(agg["stable_95_cosine"].all()) if len(agg) else False,
    }
    return trial_df, {"summary": summary, "by_genre": agg.to_dict(orient="records")}


def neighbor_audit(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    centroids_df: pd.DataFrame,
    top_k: int = 5,
    candidate_inlier_fraction: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    dims = [c for c in centroids_df.columns if c.startswith("d")]
    genre_to_centroid = {
        str(row["genre"]): row[dims].to_numpy(dtype=np.float32)
        for _, row in centroids_df.iterrows()
    }

    rows: List[Dict] = []
    summary_rows: List[Dict] = []
    inliers = _select_inlier_indices_by_genre(
        index_df=index_df,
        target160=target160,
        centroids_df=centroids_df,
        inlier_fraction=candidate_inlier_fraction,
        metric="cosine",
    )
    candidate_idx = np.concatenate([v for v in inliers.values() if len(v) > 0], axis=0)
    candidate_idx = np.unique(candidate_idx)
    X = target160[candidate_idx]
    idx_view = index_df.iloc[candidate_idx].reset_index(drop=True)
    for genre, centroid in genre_to_centroid.items():
        s = cosine_similarity(X, centroid[None, :]).reshape(-1)
        rank_idx = np.argsort(-s)[: int(top_k)]
        hits = 0
        for rank, idx in enumerate(rank_idx, start=1):
            g = str(idx_view.iloc[int(idx)]["genre"])
            hit = bool(g == genre)
            hits += int(hit)
            rows.append(
                {
                    "centroid_genre": genre,
                    "rank": int(rank),
                    "neighbor_genre": g,
                    "path": str(idx_view.iloc[int(idx)]["path"]),
                    "similarity": float(s[int(idx)]),
                    "is_hit": hit,
                }
            )
        summary_rows.append(
            {
                "genre": genre,
                "top_k": int(top_k),
                "hits": int(hits),
                "pass_5of5": bool(hits == int(top_k)),
            }
        )
    rows_df = pd.DataFrame(rows).sort_values(["centroid_genre", "rank"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("genre").reset_index(drop=True)
    summary = {
        "top_k": int(top_k),
        "all_genres_5of5": bool(summary_df["pass_5of5"].all()) if len(summary_df) else False,
        "mean_hits": float(summary_df["hits"].mean()) if len(summary_df) else 0.0,
        "candidate_inlier_fraction": float(candidate_inlier_fraction),
        "candidate_pool_size": int(len(candidate_idx)),
    }
    return rows_df, summary_df, summary


def build_genre_map_tsne(
    index_df: pd.DataFrame,
    target160: np.ndarray,
    output_dir: Path,
    max_points: int = 5000,
    seed: int = 328,
) -> Dict:
    out_csv = output_dir / "global_genre_map_tsne.csv"
    out_png = output_dir / "global_genre_map_tsne.png"
    n_total = len(index_df)
    if n_total < 8:
        return {"created": False, "reason": "Too few points for t-SNE."}

    if n_total > int(max_points):
        df = index_df.copy().reset_index(drop=True)
        df["row_id"] = df.index.astype(int)
        take_parts = []
        per_genre = max(2, int(max_points // max(1, df["genre"].nunique())))
        for _, gdf in df.groupby("genre", sort=True):
            n = min(len(gdf), per_genre)
            take_parts.append(gdf.sample(n=n, random_state=seed))
        take_df = pd.concat(take_parts, ignore_index=True).reset_index(drop=True)
        if len(take_df) > int(max_points):
            take_df = take_df.sample(int(max_points), random_state=seed).reset_index(drop=True)
        idx = take_df["row_id"].to_numpy()
        map_df = take_df.drop(columns=["row_id"]).reset_index(drop=True)
        X = target160[idx]
    else:
        map_df = index_df.copy().reset_index(drop=True)
        X = target160

    n = len(map_df)
    perplexity = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    coords = tsne.fit_transform(X)
    map_df["tsne_x"] = coords[:, 0]
    map_df["tsne_y"] = coords[:, 1]
    map_df.to_csv(out_csv, index=False)

    plot_ok = False
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        for genre, gdf in map_df.groupby("genre", sort=True):
            plt.scatter(gdf["tsne_x"], gdf["tsne_y"], s=9, alpha=0.75, label=genre)
        plt.title("Global Genre Map (t-SNE of 160-D Target Vectors)")
        plt.xlabel("t-SNE x")
        plt.ylabel("t-SNE y")
        plt.legend(markerscale=2, fontsize=8, frameon=False)
        plt.tight_layout()
        plt.savefig(out_png, dpi=180)
        plt.close()
        plot_ok = True
    except Exception:
        plot_ok = False

    return {
        "created": True,
        "n_points": int(n),
        "csv": str(out_csv),
        "png": str(out_png) if plot_ok else "",
        "plot_created": bool(plot_ok),
    }


def run_exit_audit(
    output_dir: Path,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    centroids_df: pd.DataFrame,
    centroid_distances_df: pd.DataFrame,
    validation_summary: Dict,
    silhouette_threshold: float = 0.45,
    sigma_multiplier: float = 3.0,
    neighbor_top_k: int = 5,
    neighbor_min_mean_hits: float = 5.0,
    audit_inlier_fraction: float = 1.0,
    stability_sample_fraction: float = 0.2,
    stability_trials: int = 20,
    tsne_max_points: int = 5000,
    seed: int = 328,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    spread_df = compute_intra_genre_spread(
        index_df=index_df,
        target160=arrays["target160"],
        centroids_df=centroids_df,
        metric="cosine",
        inlier_fraction=audit_inlier_fraction,
    )
    spread_df.to_csv(output_dir / "vector_variance_report.csv", index=False)

    sep_df, sep_summary = evaluate_inter_centroid_separation(
        centroid_distances_df=centroid_distances_df,
        spread_df=spread_df,
        sigma_multiplier=sigma_multiplier,
        distance_col="cosine_distance",
    )
    sep_df.to_csv(output_dir / "inter_centroid_separation.csv", index=False)

    target_centroids_json_path = output_dir / "target_centroids.json"
    export_target_centroids_json(centroids_df, target_centroids_json_path)

    stability_trials_df, stability_report = centroid_stability_report(
        index_df=index_df,
        target160=arrays["target160"],
        centroids_df=centroids_df,
        sample_fraction=stability_sample_fraction,
        n_trials=stability_trials,
        seed=seed,
    )
    stability_trials_df.to_csv(output_dir / "centroid_stability_trials.csv", index=False)
    with (output_dir / "centroid_stability_report.json").open("w", encoding="utf-8") as f:
        json.dump(stability_report, f, indent=2)

    neighbor_rows_df, neighbor_summary_df, neighbor_summary = neighbor_audit(
        index_df=index_df,
        target160=arrays["target160"],
        centroids_df=centroids_df,
        top_k=neighbor_top_k,
        candidate_inlier_fraction=audit_inlier_fraction,
    )
    neighbor_rows_df.to_csv(output_dir / "neighbor_audit.csv", index=False)
    neighbor_summary_df.to_csv(output_dir / "neighbor_audit_summary.csv", index=False)

    tsne_meta = build_genre_map_tsne(
        index_df=index_df,
        target160=arrays["target160"],
        output_dir=output_dir,
        max_points=tsne_max_points,
        seed=seed,
    )

    silhouette = float(validation_summary.get("metrics", {}).get("silhouette", float("nan")))
    silhouette_pass = bool(np.isfinite(silhouette) and silhouette >= float(silhouette_threshold))
    resolution_pass = bool(sep_summary.get("all_pairs_pass", False))
    neighbor_pass = bool(neighbor_summary.get("mean_hits", 0.0) >= float(neighbor_min_mean_hits))
    centroid_export_pass = bool(target_centroids_json_path.exists())

    checklist = {
        "goal_separability": {
            "metric": "silhouette",
            "threshold": float(silhouette_threshold),
            "value": silhouette,
            "pass": silhouette_pass,
        },
        "goal_blueprint_accuracy": {
            "metric": f"neighbor_mean_hits_top{neighbor_top_k}",
            "threshold": float(neighbor_min_mean_hits),
            "value": float(neighbor_summary.get("mean_hits", 0.0)),
            "pass": bool(neighbor_pass),
        },
        "goal_resolution": {
            "metric": "inter_centroid_distance_vs_3sigma",
            "sigma_multiplier": float(sigma_multiplier),
            "value": bool(resolution_pass),
            "pass": bool(resolution_pass),
        },
        "goal_readiness": {
            "artifact": "target_centroids.json",
            "value": str(target_centroids_json_path),
            "pass": bool(centroid_export_pass),
        },
    }
    checklist["lab2_done"] = bool(
        silhouette_pass and neighbor_pass and resolution_pass and centroid_export_pass
    )
    checklist["artifacts"] = {
        "vector_variance_report": str(output_dir / "vector_variance_report.csv"),
        "inter_centroid_separation": str(output_dir / "inter_centroid_separation.csv"),
        "target_centroids_json": str(target_centroids_json_path),
        "centroid_stability_trials": str(output_dir / "centroid_stability_trials.csv"),
        "centroid_stability_report": str(output_dir / "centroid_stability_report.json"),
        "neighbor_audit": str(output_dir / "neighbor_audit.csv"),
        "neighbor_audit_summary": str(output_dir / "neighbor_audit_summary.csv"),
        "global_genre_map_tsne_csv": str(output_dir / "global_genre_map_tsne.csv"),
        "global_genre_map_tsne_png": str(output_dir / "global_genre_map_tsne.png"),
    }
    checklist["aux"] = {
        "inter_centroid_summary": sep_summary,
        "neighbor_summary": neighbor_summary,
        "stability_summary": stability_report.get("summary", {}),
        "tsne_meta": tsne_meta,
    }

    with (output_dir / "lab2_exit_checklist.json").open("w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2)
    return checklist


def _nearest_centroid_predict(X: np.ndarray, centroids: Dict[int, np.ndarray]) -> np.ndarray:
    labels = sorted(centroids.keys())
    C = np.stack([centroids[k] for k in labels], axis=0).astype(np.float32)
    S = cosine_similarity(X, C)
    idx = np.argmax(S, axis=1)
    pred = np.asarray([labels[i] for i in idx], dtype=np.int64)
    return pred


def validate_target_space(index_df: pd.DataFrame, arrays: Dict[str, np.ndarray], seed: int = 328) -> Dict:
    X = arrays["target160"]
    y_text = index_df["genre"].astype(str).to_numpy()
    enc = LabelEncoder()
    y = enc.fit_transform(y_text)

    genre_counts = index_df["genre"].value_counts().sort_index().to_dict()
    metrics: Dict[str, float] = {}
    metrics["n_samples"] = int(len(index_df))
    metrics["n_genres"] = int(len(enc.classes_))

    if len(enc.classes_) < 2 or len(index_df) < 20:
        return {
            "metrics": metrics,
            "genre_counts": {str(k): int(v) for k, v in genre_counts.items()},
            "label_map": {int(i): str(name) for i, name in enumerate(enc.classes_)},
            "note": "Insufficient class/sample count for full validation.",
        }

    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )

    probe = LogisticRegression(max_iter=1500, random_state=seed, n_jobs=1)
    probe.fit(X_tr, y_tr)
    y_pred_probe = probe.predict(X_te)
    metrics["linear_probe_acc"] = float(accuracy_score(y_te, y_pred_probe))

    centroids: Dict[int, np.ndarray] = {}
    for k in np.unique(y_tr):
        v = X_tr[y_tr == k].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        centroids[int(k)] = v.astype(np.float32)
    y_pred_nc = _nearest_centroid_predict(X_te, centroids)
    metrics["nearest_centroid_acc"] = float(accuracy_score(y_te, y_pred_nc))
    metrics["target_fidelity_metric"] = metrics["nearest_centroid_acc"]

    try:
        metrics["silhouette"] = float(silhouette_score(X, y, metric="cosine"))
    except Exception:
        metrics["silhouette"] = float("nan")

    return {
        "metrics": metrics,
        "genre_counts": {str(k): int(v) for k, v in genre_counts.items()},
        "label_map": {int(i): str(name) for i, name in enumerate(enc.classes_)},
    }


def write_artifacts(
    output_dir: Path,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    centroids_df: pd.DataFrame,
    centroid_distances_df: pd.DataFrame,
    validation_summary: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(output_dir / "embeddings_index.csv", index=False)
    np.savez_compressed(
        output_dir / "embeddings.npz",
        z_content=arrays["z_content"],
        z_style=arrays["z_style"],
        descriptor32=arrays["descriptor32"],
        target160=arrays["target160"],
        music_prob=arrays["music_prob"],
    )
    centroids_df.to_csv(output_dir / "centroids_space.csv", index=False)
    centroids_df.to_csv(output_dir / "centroids_160d.csv", index=False)
    centroid_distances_df.to_csv(output_dir / "centroid_distances.csv", index=False)
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)
