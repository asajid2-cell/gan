from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans

from src.lab3_bridge import (
    descriptor32_from_logmel,
    extract_log_mel,
    fix_log_mel_frames,
    load_audio_chunk,
)
from src.lab3_data import DEFAULT_MANIFESTS, genre_source_table, load_manifests
from src.lab3_bridge import FrozenLab1Encoder


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Auto-label manifests by clustering Lab1-derived Lab2 target vectors (z_style+descriptor32). "
            "This is unpaired and data-driven, but clusters are 'style buckets' (not guaranteed human genres)."
        )
    )
    p.add_argument("--manifests-root", type=Path, default=Path("Z:/DataSets/_lab1_manifests"))
    p.add_argument("--manifest-files", nargs="*", default=DEFAULT_MANIFESTS)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-audit-csv", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)

    p.add_argument("--lab1-checkpoint", type=Path, default=None)
    p.add_argument("--n-frames", type=int, default=256)
    p.add_argument("--chunk-seconds", type=float, default=5.0)
    p.add_argument("--start-sec", type=float, default=0.0)

    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--zstyle-weight", type=float, default=1.0)
    p.add_argument("--descriptor-weight", type=float, default=1.0)
    p.add_argument(
        "--min-conf",
        type=float,
        default=0.0,
        help="If >0, samples with cosine-to-centroid below this become 'unassigned'.",
    )

    p.add_argument("--require-is-music", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--cluster-samples-per-source",
        type=int,
        default=0,
        help="If >0, fit KMeans on a source-balanced subset (up to N per source) to reduce source dominance.",
    )
    p.add_argument("--require-min-sources-per-cluster", type=int, default=1)
    p.add_argument(
        "--max-source-fraction-per-cluster",
        type=float,
        default=1.0,
        help="If <1.0, abort when any cluster is dominated by a single source above this fraction.",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=0,
        help="If >0, enforce a minimum assigned samples per cluster (after labeling).",
    )
    p.add_argument(
        "--small-cluster-action",
        choices=["abort", "unassigned", "merge"],
        default="abort",
        help="What to do when a cluster has < min-cluster-size samples.",
    )
    p.add_argument("--max-files", type=int, default=0, help="0 means all")
    p.add_argument("--seed", type=int, default=328)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return p.parse_args()


def _pick_device(arg: str) -> str:
    a = str(arg).strip().lower()
    if a == "cpu":
        return "cpu"
    if a == "cuda":
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_lab1_checkpoint() -> Path:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        root / "saves" / "lab1_run_combo_af_gate" / "latest.pt",
        root / "saves" / "lab1_run_a" / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _compose_target160(z_style: np.ndarray, d32: np.ndarray, wz: float, wd: float) -> np.ndarray:
    z = z_style.astype(np.float32) * float(wz)
    d = d32.astype(np.float32) * float(wd)
    out = np.concatenate([z, d], axis=0).astype(np.float32)
    # Normalize for cosine/kmeans stability.
    out = out / (np.linalg.norm(out) + 1e-8)
    return out


def _cos_to_centroids(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # x: [N, D] unit norm; centroids: [K, D] unit norm
    return x @ centroids.T


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))
    device = _pick_device(args.device)

    ckpt = Path(args.lab1_checkpoint) if args.lab1_checkpoint is not None else _default_lab1_checkpoint()
    lab1 = FrozenLab1Encoder(ckpt, device=device)
    sr = int(lab1.cfg.sample_rate)

    df = load_manifests(Path(args.manifests_root), manifest_files=args.manifest_files)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    if bool(args.require_is_music) and "is_music" in df.columns:
        # Keep unknowns; drop explicit 0s.
        is_music = pd.to_numeric(df["is_music"], errors="coerce").fillna(1).astype(int)
        df = df[is_music == 1].reset_index(drop=True)

    if int(args.max_files) > 0:
        n = min(int(args.max_files), len(df))
        df = df.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)

    print(
        f"[lab2-cluster] device={device} sr={sr} files={len(df)} "
        f"k={int(args.n_clusters)} start={float(args.start_sec):.2f}s chunk={float(args.chunk_seconds):.2f}s"
    )

    feats: List[np.ndarray] = []
    ok: List[int] = []
    for i, p in enumerate(df["path"].astype(str).tolist()):
        try:
            y = load_audio_chunk(
                path=Path(p),
                sample_rate=sr,
                seconds=float(args.chunk_seconds),
                start_sec=float(args.start_sec),
            )
            log_mel = extract_log_mel(y, sr=sr)
            log_mel = fix_log_mel_frames(log_mel, n_frames=int(args.n_frames))
            z = lab1.infer_log_mel(log_mel)["z_style"]
            d = descriptor32_from_logmel(log_mel)
            feats.append(_compose_target160(z, d, wz=float(args.zstyle_weight), wd=float(args.descriptor_weight)))
            ok.append(1)
        except Exception:
            feats.append(np.zeros((160,), dtype=np.float32))
            ok.append(0)
        if (i + 1) % 50 == 0:
            print(f"[lab2-cluster] encoded {i+1}/{len(df)}")

    x = np.stack(feats, axis=0).astype(np.float32)
    ok_arr = np.array(ok, dtype=np.int64)
    x_ok = x[ok_arr == 1]
    if len(x_ok) < max(32, int(args.n_clusters) * 8):
        raise RuntimeError(f"Too few valid encodings ({len(x_ok)}) to cluster.")

    # Optionally fit on a source-balanced subset to reduce single-source dominance.
    fit_mask = np.zeros((len(df),), dtype=bool)
    fit_mask[ok_arr == 1] = True
    if int(args.cluster_samples_per_source) > 0 and "source" in df.columns:
        cap = int(args.cluster_samples_per_source)
        fit_mask[:] = False
        for s, sdf in df[ok_arr == 1].groupby(df.loc[ok_arr == 1, "source"].astype(str), sort=True):
            idx = sdf.index.to_numpy().astype(np.int64)
            if len(idx) == 0:
                continue
            take = min(cap, len(idx))
            pick = rng.choice(idx, size=take, replace=False)
            fit_mask[pick] = True
        if int(fit_mask.sum()) < max(32, int(args.n_clusters) * 8):
            raise RuntimeError(
                f"Too few source-balanced samples to fit ({int(fit_mask.sum())}). "
                "Increase --cluster-samples-per-source or set it to 0."
            )

    x_fit = x[fit_mask]
    # KMeans on unit vectors approximates spherical clustering sufficiently for our use.
    km = KMeans(n_clusters=int(args.n_clusters), random_state=int(args.seed), n_init=10)
    km.fit(x_fit.astype(np.float32))
    cent = km.cluster_centers_.astype(np.float32)
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)

    # Assign all valid points; invalid -> unassigned.
    cluster = np.full((len(df),), fill_value=-1, dtype=np.int64)
    conf = np.zeros((len(df),), dtype=np.float32)
    sim = _cos_to_centroids(x_ok.astype(np.float32), cent)
    best = np.argmax(sim, axis=1).astype(np.int64)
    best_conf = sim[np.arange(len(sim)), best].astype(np.float32)
    ok_idx = np.where(ok_arr == 1)[0]
    cluster[ok_idx] = best
    conf[ok_idx] = best_conf

    # Optional: enforce a minimum cluster size by aborting, unassigning, or merging.
    min_sz = int(args.min_cluster_size)
    if min_sz > 0:
        counts = pd.Series(cluster[ok_idx]).value_counts().to_dict()
        small = sorted([int(k) for k, v in counts.items() if int(v) < min_sz])
        if small:
            act = str(args.small_cluster_action).strip().lower()
            if act == "abort":
                raise RuntimeError(
                    f"Small clusters detected (<{min_sz}): { {int(k): int(counts.get(k, 0)) for k in small} }. "
                    "Try smaller --n-clusters, different --seed, or use --small-cluster-action merge."
                )
            if act == "unassigned":
                for k in small:
                    m = cluster == int(k)
                    cluster[m] = -1
                    conf[m] = 0.0
            else:
                # Merge: reassign samples from small clusters to their next-best centroid among the remaining clusters.
                keep = [k for k in range(int(args.n_clusters)) if k not in set(small)]
                if len(keep) < 1:
                    raise RuntimeError("All clusters are small; cannot merge.")
                # For ok points, sim is [N_ok, K]. Remap to kept clusters.
                sim_keep = sim[:, keep]
                best2 = np.argmax(sim_keep, axis=1).astype(np.int64)
                best2 = np.array([keep[int(b)] for b in best2], dtype=np.int64)
                for j, gi in enumerate(ok_idx.tolist()):
                    if int(cluster[gi]) in set(small):
                        cluster[gi] = int(best2[j])
                        conf[gi] = float(sim[j, int(best2[j])])

    min_conf = float(args.min_conf)
    genre: List[str] = []
    for c, cc, oo in zip(cluster.tolist(), conf.tolist(), ok_arr.tolist()):
        if int(oo) != 1:
            genre.append("unassigned")
            continue
        if int(c) < 0:
            genre.append("unassigned")
            continue
        if min_conf > 0.0 and float(cc) < min_conf:
            genre.append("unassigned")
            continue
        genre.append(f"cluster_{int(c)}")

    out = df.copy()
    out["genre"] = genre
    out["genre_cluster"] = cluster
    out["genre_conf"] = conf
    out["genre_method"] = "lab2_target160_kmeans"
    out["genre_k"] = int(args.n_clusters)
    out["genre_min_conf"] = float(args.min_conf)
    out["genre_ckpt"] = str(ckpt)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[lab2-cluster] wrote {out_csv}")

    audit = genre_source_table(out[out["genre"] != "unassigned"].copy())
    if args.out_audit_csv is None:
        out_audit = out_csv.with_suffix("").with_name(out_csv.stem + "_genre_source_table.csv")
    else:
        out_audit = Path(args.out_audit_csv)
    audit.to_csv(out_audit, index=False)
    print(f"[lab2-cluster] wrote {out_audit}")

    # Fail-fast constraints for unpaired validity.
    if "genre" in out.columns and "source" in out.columns:
        # For each assigned cluster, require >=N sources and (optionally) cap max single-source dominance.
        assigned = out[out["genre"] != "unassigned"].copy()
        if len(assigned) > 0:
            req_sources = int(args.require_min_sources_per_cluster)
            bad: Dict[str, Dict] = {}
            for g, gdf in assigned.groupby("genre", sort=True):
                nsrc = int(gdf["source"].astype(str).nunique())
                src_counts = gdf["source"].astype(str).value_counts()
                frac = float(src_counts.iloc[0] / max(1, int(src_counts.sum()))) if len(src_counts) else 1.0
                if nsrc < req_sources or frac > float(args.max_source_fraction_per_cluster):
                    bad[str(g)] = {"n_sources": nsrc, "max_source_frac": frac}
            if bad:
                raise RuntimeError(
                    "Cluster/source coupling too strong for unpaired validity. "
                    f"Bad clusters: {bad}. Try --cluster-samples-per-source, a different --seed, or a different --n-clusters."
                )

    info = {
        "k": int(args.n_clusters),
        "min_conf": float(args.min_conf),
        "n_rows": int(len(out)),
        "n_valid": int(int(ok_arr.sum())),
        "n_assigned": int(int((out["genre"] != "unassigned").sum())),
        "cluster_samples_per_source": int(args.cluster_samples_per_source),
        "require_min_sources_per_cluster": int(args.require_min_sources_per_cluster),
        "max_source_fraction_per_cluster": float(args.max_source_fraction_per_cluster),
        "mean_conf_assigned": float(
            out.loc[out["genre"] != "unassigned", "genre_conf"].mean()
            if int((out["genre"] != "unassigned").sum()) > 0
            else float("nan")
        ),
        "cluster_sizes": {str(k): int(v) for k, v in pd.Series(out["genre"]).value_counts().to_dict().items()},
    }

    if args.out_json is None:
        out_json = out_csv.with_suffix("").with_name(out_csv.stem + "_info.json")
    else:
        out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[lab2-cluster] wrote {out_json}")


if __name__ == "__main__":
    main()
