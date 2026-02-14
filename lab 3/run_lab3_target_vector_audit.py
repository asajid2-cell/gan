from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.lab3_codec_data import load_codec_cache
from src.lab3_codec_judge import CodecStyleJudge


def _read_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit target-vector integrity and source leakage for Lab3 codec runs."
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=328)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--max-samples", type=int, default=0, help="0 means all")
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def _safe_logreg_train_eval(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> Dict[str, float]:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return {"acc": float("nan"), "chance": 1.0}
    if np.min(counts) < 2:
        return {"acc": float("nan"), "chance": float(np.max(counts) / np.sum(counts))}

    x_tr, x_te, y_tr, y_te = train_test_split(
        x,
        y,
        test_size=float(test_size),
        random_state=int(seed),
        stratify=y,
    )
    clf = LogisticRegression(max_iter=1500, random_state=int(seed), n_jobs=1)
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    acc = float(accuracy_score(y_te, pred))
    chance = float(np.max(np.bincount(y_te.astype(np.int64))) / len(y_te))
    return {"acc": acc, "chance": chance}


def _centroid_cos(x: np.ndarray, y: np.ndarray) -> float:
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    c = []
    for k in classes:
        v = x[y == k].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-8)
        c.append(v)
    c = np.stack(c, axis=0)
    sim = c @ c.T
    offdiag = sim[~np.eye(sim.shape[0], dtype=bool)]
    return float(np.mean(offdiag))


def _loo_source_genre_generalization(
    x: np.ndarray,
    y_genre: np.ndarray,
    y_source: np.ndarray,
    source_names: Dict[int, str],
    seed: int,
) -> List[Dict]:
    rows: List[Dict] = []
    all_sources = np.unique(y_source).tolist()
    for s in all_sources:
        te_mask = y_source == s
        tr_mask = ~te_mask
        if int(np.sum(te_mask)) < 16 or int(np.sum(tr_mask)) < 32:
            continue

        y_tr = y_genre[tr_mask]
        y_te = y_genre[te_mask]
        train_classes = set(np.unique(y_tr).tolist())
        eval_mask = np.array([g in train_classes for g in y_te], dtype=bool)
        if int(np.sum(eval_mask)) < 16:
            continue

        x_tr = x[tr_mask]
        x_te = x[te_mask][eval_mask]
        y_te_f = y_te[eval_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te_f)) < 2:
            continue

        clf = LogisticRegression(max_iter=1500, random_state=int(seed), n_jobs=1)
        clf.fit(x_tr, y_tr)
        pred = clf.predict(x_te)
        acc = float(accuracy_score(y_te_f, pred))
        chance = float(np.max(np.bincount(y_te_f.astype(np.int64))) / len(y_te_f))
        rows.append(
            {
                "heldout_source_idx": int(s),
                "heldout_source": source_names.get(int(s), str(int(s))),
                "n_eval": int(len(y_te_f)),
                "genre_acc": acc,
                "genre_chance": chance,
            }
        )
    return rows


@torch.no_grad()
def _extract_judge_embed(run_dir: Path, q_emb: np.ndarray, batch_size: int = 128) -> np.ndarray:
    ckpt = run_dir / "codec_style_judge.pt"
    if not ckpt.exists():
        return np.zeros((0, 0), dtype=np.float32)
    payload = torch.load(str(ckpt), map_location="cpu")
    cfg = payload.get("config", {})
    model = CodecStyleJudge(
        in_channels=int(cfg.get("in_channels", q_emb.shape[1])),
        n_genres=int(cfg.get("n_genres", 0)),
        hidden=int(cfg.get("hidden", 256)),
        emb_dim=int(cfg.get("emb_dim", 128)),
        n_sources=int(cfg.get("n_sources", 0)),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    x = torch.from_numpy(q_emb.astype(np.float32))
    rows: List[np.ndarray] = []
    for i in range(0, int(x.shape[0]), int(max(1, batch_size))):
        xb = x[i : i + int(max(1, batch_size))]
        emb = model.embed(xb).detach().cpu().numpy().astype(np.float64)
        rows.append(emb)
    if not rows:
        return np.zeros((0, model.emb_dim), dtype=np.float64)
    return np.concatenate(rows, axis=0).astype(np.float64)


def main() -> None:
    args = _read_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    cache_dir = run_dir / "cache"
    if not cache_dir.exists():
        raise FileNotFoundError(cache_dir)

    idx_df, arrays, genre_to_idx, _ = load_codec_cache(cache_dir)
    if "source" not in idx_df.columns:
        raise RuntimeError("codec cache index missing 'source' column.")

    n = len(idx_df)
    take = int(args.max_samples) if int(args.max_samples) > 0 else n
    take = min(take, n)
    rng = np.random.default_rng(int(args.seed))
    pick = rng.choice(np.arange(n), size=take, replace=False) if take < n else np.arange(n)

    idx_sub = idx_df.iloc[pick].reset_index(drop=True)
    z_style = arrays["z_style"][pick].astype(np.float64)
    q_emb = arrays["q_emb"][pick].astype(np.float64)
    q_mean = q_emb.mean(axis=2)  # [N, C]
    y_genre = arrays["genre_idx"][pick].astype(np.int64)
    q_judge_embed = _extract_judge_embed(run_dir, q_emb.astype(np.float32))

    source_cats = idx_sub["source"].astype("category")
    y_source = source_cats.cat.codes.to_numpy().astype(np.int64)
    source_names = {int(i): str(nm) for i, nm in enumerate(source_cats.cat.categories.tolist())}

    # Basic composition diagnostics
    by_genre_source = (
        idx_sub.assign(genre_idx=y_genre, source_idx=y_source)
        .groupby(["genre_idx", "source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # Predictive diagnostics
    genre_from_z = _safe_logreg_train_eval(z_style, y_genre, test_size=args.test_size, seed=args.seed)
    source_from_z = _safe_logreg_train_eval(z_style, y_source, test_size=args.test_size, seed=args.seed)
    genre_from_q = _safe_logreg_train_eval(q_mean, y_genre, test_size=args.test_size, seed=args.seed)
    source_from_q = _safe_logreg_train_eval(q_mean, y_source, test_size=args.test_size, seed=args.seed)
    if q_judge_embed.size > 0:
        genre_from_qe = _safe_logreg_train_eval(q_judge_embed, y_genre, test_size=args.test_size, seed=args.seed)
        source_from_qe = _safe_logreg_train_eval(q_judge_embed, y_source, test_size=args.test_size, seed=args.seed)
    else:
        genre_from_qe = {"acc": float("nan"), "chance": float("nan")}
        source_from_qe = {"acc": float("nan"), "chance": float("nan")}

    # Cross-source generalization (leave-one-source-out)
    loo_z = _loo_source_genre_generalization(
        x=z_style,
        y_genre=y_genre,
        y_source=y_source,
        source_names=source_names,
        seed=args.seed,
    )
    loo_q = _loo_source_genre_generalization(
        x=q_mean,
        y_genre=y_genre,
        y_source=y_source,
        source_names=source_names,
        seed=args.seed,
    )
    loo_qe = _loo_source_genre_generalization(
        x=q_judge_embed if q_judge_embed.size > 0 else q_mean,
        y_genre=y_genre,
        y_source=y_source,
        source_names=source_names,
        seed=args.seed,
    ) if q_judge_embed.size > 0 else []

    def _gap(src: Dict[str, float], gen: Dict[str, float]) -> float:
        a = float(src.get("acc", float("nan")))
        b = float(gen.get("acc", float("nan")))
        if not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return float(a - b)

    out = {
        "run_dir": str(run_dir),
        "n_samples": int(take),
        "n_genres": int(len(np.unique(y_genre))),
        "n_sources": int(len(np.unique(y_source))),
        "genre_to_idx": {str(k): int(v) for k, v in genre_to_idx.items()},
        "source_names": {str(k): v for k, v in source_names.items()},
        "centroid_offdiag_cos": {
            "z_style": _centroid_cos(z_style, y_genre),
            "q_mean": _centroid_cos(q_mean, y_genre),
            "q_judge_embed": _centroid_cos(q_judge_embed, y_genre) if q_judge_embed.size > 0 else float("nan"),
        },
        "predictive": {
            "genre_from_z_style": genre_from_z,
            "source_from_z_style": source_from_z,
            "genre_from_q_mean": genre_from_q,
            "source_from_q_mean": source_from_q,
            "genre_from_q_judge_embed": genre_from_qe,
            "source_from_q_judge_embed": source_from_qe,
        },
        "leakage_gap": {
            "q_mean_source_minus_genre": _gap(source_from_q, genre_from_q),
            "q_judge_embed_source_minus_genre": _gap(source_from_qe, genre_from_qe),
        },
        "source_below_genre_pass": {
            "q_mean": bool(_gap(source_from_q, genre_from_q) <= 0.0) if np.isfinite(_gap(source_from_q, genre_from_q)) else False,
            "q_judge_embed": bool(_gap(source_from_qe, genre_from_qe) <= 0.0) if np.isfinite(_gap(source_from_qe, genre_from_qe)) else False,
        },
        "loo_cross_source": {
            "z_style": loo_z,
            "q_mean": loo_q,
            "q_judge_embed": loo_qe,
        },
        "notes": {
            "risk_source_leakage_if_source_acc_much_higher_than_genre_acc": True,
            "risk_proxy_labels_if_loo_genre_acc_drops_near_chance": True,
        },
    }

    out_json = args.output_json or (run_dir / "target_vector_audit.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out_csv = args.output_csv or (run_dir / "target_vector_audit_by_genre_source.csv")
    by_genre_source.to_csv(out_csv, index=False)

    print(f"[target-audit] wrote {out_json}")
    print(f"[target-audit] wrote {out_csv}")
    print("[target-audit] predictive:")
    for k, v in out["predictive"].items():
        print(f"  - {k}: acc={v.get('acc')} chance={v.get('chance')}")
    print(f"[target-audit] loo_z rows={len(loo_z)} loo_q rows={len(loo_q)} loo_qe rows={len(loo_qe)}")


if __name__ == "__main__":
    main()
