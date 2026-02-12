from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.lab3_bridge import FrozenLab1Encoder, denormalize_log_mel
from src.lab3_data import load_cache, stratified_split_indices
from src.lab3_models import ReconstructionDecoder
from src.lab3_train import build_condition_bank, load_target_centroids


@dataclass
class Lab4Gate:
    min_mps: float = 0.94
    min_style_conf: float = 0.52
    min_style_acc: float = 0.75
    max_target_centroid_mae_norm: float = 0.60
    max_target_hf_mae: float = 0.05
    max_target_lf_mae: float = 0.08
    max_intra_target_diversity_ratio: float = 1.03


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended Lab3 quality audit for Lab4 readiness.")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--runs-root", type=Path, default=Path("saves2/lab3_synthesis"))
    p.add_argument("--max-eval-samples", type=int, default=256)
    p.add_argument("--output-csv", type=Path, default=Path("saves2/lab3_synthesis/quality_audit_summary.csv"))
    p.add_argument("--output-json", type=Path, default=Path("saves2/lab3_synthesis/quality_audit_summary.json"))
    return p.parse_args()


def _sample_shift_targets(source_idx: np.ndarray, n_genres: int, rng: np.random.Generator) -> np.ndarray:
    tgt = rng.integers(0, int(n_genres), size=len(source_idx), endpoint=False)
    clash = tgt == source_idx
    tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt.astype(np.int64)


def _pairwise_cos_mean(feats: np.ndarray) -> float:
    if len(feats) < 2:
        return float("nan")
    x = feats.astype(np.float64)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = x @ x.T
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def _intra_target_mfcc_cos_mean(mfcc_feats: np.ndarray, tgt_idx: np.ndarray, n_genres: int) -> float:
    vals: List[float] = []
    for g in range(int(n_genres)):
        mask = tgt_idx == g
        if int(mask.sum()) < 2:
            continue
        vals.append(_pairwise_cos_mean(mfcc_feats[mask]))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _lab4_passes(row: Dict, gate: Lab4Gate) -> Dict[str, bool]:
    checks = {
        "mps": bool(row["mps"] >= gate.min_mps),
        "style_conf": bool(row["style_conf"] >= gate.min_style_conf),
        "style_acc": bool(row["style_acc"] >= gate.min_style_acc),
        "target_centroid": bool(row["target_centroid_mae_norm"] <= gate.max_target_centroid_mae_norm),
        "target_hf": bool(row["target_hf_mae"] <= gate.max_target_hf_mae),
        "target_lf": bool(row["target_lf_mae"] <= gate.max_target_lf_mae),
        "intra_target_diversity": bool(row["intra_target_diversity_ratio"] <= gate.max_intra_target_diversity_ratio),
    }
    checks["lab4_ready"] = bool(all(checks.values()))
    return checks


def _mel_stats(mel_db: np.ndarray, mel_freqs: np.ndarray, top: int, bot: int) -> tuple[float, float, float]:
    p = np.power(10.0, mel_db / 10.0)
    w = p.sum(axis=1)
    centroid = float((mel_freqs * w).sum() / (w.sum() + 1e-8))
    t = float(p.sum() + 1e-8)
    hf = float(p[-top:, :].sum() / t)
    lf = float(p[:bot, :].sum() / t)
    return centroid, hf, lf


def _mcd_from_mfcc(mfcc_a: np.ndarray, mfcc_b: np.ndarray) -> float:
    """
    Mel-Cepstral Distortion (dB), excluding c0.
    Inputs are [n_mfcc, T].
    """
    if mfcc_a.shape != mfcc_b.shape:
        t = min(mfcc_a.shape[1], mfcc_b.shape[1])
        mfcc_a = mfcc_a[:, :t]
        mfcc_b = mfcc_b[:, :t]
    if mfcc_a.shape[1] == 0:
        return float("nan")
    diff = mfcc_a[1:, :] - mfcc_b[1:, :]
    dist = np.sqrt(np.sum(diff * diff, axis=0))
    scale = (10.0 / np.log(10.0)) * np.sqrt(2.0)
    return float(scale * np.mean(dist))


def _ssm_mae_from_mfcc(mfcc_a: np.ndarray, mfcc_b: np.ndarray, max_frames: int = 96) -> float:
    """
    Self-similarity matrix error between fake and real MFCC trajectories.
    Lower is better.
    """
    if mfcc_a.shape != mfcc_b.shape:
        t = min(mfcc_a.shape[1], mfcc_b.shape[1])
        mfcc_a = mfcc_a[:, :t]
        mfcc_b = mfcc_b[:, :t]
    if mfcc_a.shape[1] <= 1:
        return float("nan")

    def _subsample(x: np.ndarray, n: int) -> np.ndarray:
        if x.shape[1] <= n:
            return x
        idx = np.linspace(0, x.shape[1] - 1, n).astype(np.int64)
        return x[:, idx]

    a = _subsample(mfcc_a, int(max_frames)).T  # [T, C]
    b = _subsample(mfcc_b, int(max_frames)).T
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    s_a = a @ a.T
    s_b = b @ b.T
    return float(np.mean(np.abs(s_a - s_b)))


def _frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """
    Stable Fréchet distance between two Gaussians.
    """
    diff = mu1 - mu2
    cov_prod = cov1 @ cov2
    vals, vecs = np.linalg.eigh(cov_prod)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    sqrt_cov_prod = (vecs * np.sqrt(vals)) @ vecs.T
    return float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * sqrt_cov_prod))


def audit_one_run(run_dir: Path, max_eval_samples: int) -> Dict:
    state_path = run_dir / "run_state.json"
    ckpt_path = run_dir / "checkpoints" / "stage2_latest.pt"
    if not state_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError(f"Missing run_state or stage2 checkpoint in {run_dir}")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    cfg = state.get("config", {})

    _, arrays, genre_to_idx = load_cache(run_dir / "cache")
    n_genres = int(len(genre_to_idx))
    seed = int(cfg.get("seed", 328))
    val_ratio = float(cfg.get("val_ratio", 0.15))
    train_idx, val_idx = stratified_split_indices(arrays["genre_idx"], val_ratio=val_ratio, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen_encoder = FrozenLab1Encoder(Path(cfg["lab1_checkpoint"]), device=device)
    sr = int(frozen_encoder.cfg.sample_rate)
    target_centroids = load_target_centroids(Path(cfg["lab2_centroids_json"]))
    cond_bank = build_condition_bank(genre_to_idx, target_centroids).to(device)

    generator = ReconstructionDecoder(
        zc_dim=int(arrays["z_content"].shape[1]),
        cond_dim=int(cond_bank.shape[1]),
        n_mels=int(arrays["mel_norm"].shape[1]),
        n_frames=int(arrays["mel_norm"].shape[2]),
        norm=str(cfg.get("generator_norm", "instance")),
        upsample=str(cfg.get("generator_upsample", "transpose")),
    ).to(device)
    payload = torch.load(str(ckpt_path), map_location="cpu")
    generator.load_state_dict(payload["generator"], strict=False)
    generator.eval()

    style_clf = LogisticRegression(max_iter=1500, random_state=328, n_jobs=1)
    style_clf.fit(arrays["z_style"][train_idx], arrays["genre_idx"][train_idx])

    n_eval = int(min(max_eval_samples, len(val_idx)))
    use_idx = val_idx[:n_eval]
    src_idx = arrays["genre_idx"][use_idx].astype(np.int64)
    rng = np.random.default_rng(123)
    tgt_idx = _sample_shift_targets(src_idx, n_genres=n_genres, rng=rng)

    mel_freqs = librosa.mel_frequencies(n_mels=int(arrays["mel_norm"].shape[1]), fmin=20, fmax=sr / 2.0)
    top = max(1, int(round(len(mel_freqs) * 0.20)))
    bot = max(1, int(round(len(mel_freqs) * 0.20)))

    # target-genre real reference stats
    target_ref: Dict[int, Dict[str, float]] = {}
    real_mfcc_by_genre: Dict[int, np.ndarray] = {}
    for g in range(n_genres):
        mask = arrays["genre_idx"] == g
        if int(np.sum(mask)) == 0:
            continue
        c_list, hf_list, lf_list, mf_list = [], [], [], []
        for mel_norm in arrays["mel_norm"][mask]:
            mel_db = denormalize_log_mel(torch.from_numpy(mel_norm).unsqueeze(0)).squeeze(0).numpy().astype(np.float32)
            c, hf, lf = _mel_stats(mel_db, mel_freqs, top, bot)
            c_list.append(c)
            hf_list.append(hf)
            lf_list.append(lf)
            mf_list.append(librosa.feature.mfcc(S=mel_db, n_mfcc=13).mean(axis=1))
        target_ref[g] = {
            "centroid": float(np.mean(c_list)),
            "hf": float(np.mean(hf_list)),
            "lf": float(np.mean(lf_list)),
        }
        real_mfcc_by_genre[g] = np.stack(mf_list).astype(np.float32)

    mps_vals: List[float] = []
    style_conf_vals: List[float] = []
    style_margin_vals: List[float] = []
    style_entropy_adj_vals: List[float] = []
    style_hit_vals: List[float] = []
    pred_cls: List[int] = []
    fake_centroid_hz: List[float] = []
    real_centroid_hz: List[float] = []
    fake_hf_ratio: List[float] = []
    real_hf_ratio: List[float] = []
    fake_lf_ratio: List[float] = []
    real_lf_ratio: List[float] = []
    fake_dr: List[float] = []
    real_dr: List[float] = []
    mfcc_fake: List[np.ndarray] = []
    mfcc_real: List[np.ndarray] = []
    mcd_vals: List[float] = []
    ssm_err_vals: List[float] = []

    target_centroid_mae_norm: List[float] = []
    target_hf_mae: List[float] = []
    target_lf_mae: List[float] = []

    batch_size = 32
    for start in range(0, n_eval, batch_size):
        end = min(start + batch_size, n_eval)
        ridx = use_idx[start:end]
        zc = torch.from_numpy(arrays["z_content"][ridx]).to(device).float()
        cond = cond_bank[torch.from_numpy(tgt_idx[start:end]).to(device)]

        with torch.no_grad():
            fake = generator(zc, cond)
        fake_db = denormalize_log_mel(fake).cpu().numpy().astype(np.float32)
        real_db = denormalize_log_mel(torch.from_numpy(arrays["mel_norm"][ridx]).to(device).float()).cpu().numpy().astype(np.float32)

        with torch.no_grad():
            enc_out = frozen_encoder.forward_log_mel_tensor(torch.from_numpy(fake_db).to(frozen_encoder.device).float())
        zc_fake = enc_out["z_content"].detach().cpu().numpy()
        zs_fake = enc_out["z_style"].detach().cpu().numpy()

        zc_src = arrays["z_content"][ridx]
        zc_fake_n = zc_fake / (np.linalg.norm(zc_fake, axis=1, keepdims=True) + 1e-8)
        zc_src_n = zc_src / (np.linalg.norm(zc_src, axis=1, keepdims=True) + 1e-8)
        mps_vals.extend(np.sum(zc_fake_n * zc_src_n, axis=1).tolist())

        probs = style_clf.predict_proba(zs_fake)
        tgt = tgt_idx[start:end]
        style_conf_vals.extend(probs[np.arange(len(tgt)), tgt].tolist())
        if probs.shape[1] > 1:
            second = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2]
            entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1) / np.log(float(probs.shape[1]))
        else:
            second = np.zeros((len(tgt),), dtype=np.float64)
            entropy = np.zeros((len(tgt),), dtype=np.float64)
        style_margin_vals.extend((probs[np.arange(len(tgt)), tgt] - second).tolist())
        style_entropy_adj_vals.extend((probs[np.arange(len(tgt)), tgt] * (1.0 - entropy)).tolist())
        pred = np.argmax(probs, axis=1)
        style_hit_vals.extend((pred == tgt).astype(np.float32).tolist())
        pred_cls.extend(pred.tolist())

        for i in range(fake_db.shape[0]):
            c_f, hf_f, lf_f = _mel_stats(fake_db[i], mel_freqs, top, bot)
            c_r, hf_r, lf_r = _mel_stats(real_db[i], mel_freqs, top, bot)
            fake_centroid_hz.append(c_f)
            real_centroid_hz.append(c_r)
            fake_hf_ratio.append(hf_f)
            real_hf_ratio.append(hf_r)
            fake_lf_ratio.append(lf_f)
            real_lf_ratio.append(lf_r)

            g_t = int(tgt[i])
            if g_t in target_ref:
                ref = target_ref[g_t]
                target_centroid_mae_norm.append(abs(c_f - ref["centroid"]) / (ref["centroid"] + 1e-8))
                target_hf_mae.append(abs(hf_f - ref["hf"]))
                target_lf_mae.append(abs(lf_f - ref["lf"]))

            fake_dr.append(float(np.percentile(fake_db[i], 95) - np.percentile(fake_db[i], 5)))
            real_dr.append(float(np.percentile(real_db[i], 95) - np.percentile(real_db[i], 5)))
            mfcc_f = librosa.feature.mfcc(S=fake_db[i], n_mfcc=13).astype(np.float32)
            mfcc_r = librosa.feature.mfcc(S=real_db[i], n_mfcc=13).astype(np.float32)
            mfcc_fake.append(mfcc_f.mean(axis=1))
            mfcc_real.append(mfcc_r.mean(axis=1))
            mcd_vals.append(_mcd_from_mfcc(mfcc_f, mfcc_r))
            ssm_err_vals.append(_ssm_mae_from_mfcc(mfcc_f, mfcc_r))

    p_dist = np.bincount(np.asarray(pred_cls, dtype=np.int64), minlength=n_genres).astype(np.float64)
    p_dist = p_dist / (p_dist.sum() + 1e-12)
    pred_entropy = float(-(p_dist * np.log(p_dist + 1e-12)).sum() / np.log(float(n_genres)))

    mfcc_fake_arr = np.stack(mfcc_fake).astype(np.float32)
    mfcc_real_arr = np.stack(mfcc_real).astype(np.float32)
    mu_f = np.mean(mfcc_fake_arr, axis=0)
    mu_r = np.mean(mfcc_real_arr, axis=0)
    cov_f = np.cov(mfcc_fake_arr, rowvar=False) + 1e-6 * np.eye(mfcc_fake_arr.shape[1], dtype=np.float64)
    cov_r = np.cov(mfcc_real_arr, rowvar=False) + 1e-6 * np.eye(mfcc_real_arr.shape[1], dtype=np.float64)
    fad_mfcc_proxy = _frechet_distance(mu_f.astype(np.float64), cov_f, mu_r.astype(np.float64), cov_r)
    fake_intra_target = _intra_target_mfcc_cos_mean(mfcc_fake_arr, tgt_idx=tgt_idx, n_genres=n_genres)

    real_intra_vals = []
    for g in range(n_genres):
        g_arr = real_mfcc_by_genre.get(g)
        if g_arr is None or len(g_arr) < 2:
            continue
        real_intra_vals.append(_pairwise_cos_mean(g_arr))
    real_intra_target = float(np.mean(real_intra_vals)) if real_intra_vals else float("nan")
    intra_target_div_ratio = float(fake_intra_target / (real_intra_target + 1e-8)) if np.isfinite(real_intra_target) else float("nan")

    return {
        "run": run_dir.name,
        "n_eval": int(n_eval),
        "generator_upsample": str(cfg.get("generator_upsample", "transpose")),
        "discriminator_arch": str(cfg.get("discriminator_arch", "single")),
        "mps": float(np.mean(mps_vals)),
        "style_conf": float(np.mean(style_conf_vals)),
        "style_conf_margin": float(np.mean(style_margin_vals)),
        "style_conf_entropy_adjusted": float(np.mean(style_entropy_adj_vals)),
        "style_acc": float(np.mean(style_hit_vals)),
        "fake_centroid_hz": float(np.mean(fake_centroid_hz)),
        "real_centroid_hz": float(np.mean(real_centroid_hz)),
        "centroid_ratio": float(np.mean(fake_centroid_hz) / (np.mean(real_centroid_hz) + 1e-8)),
        "fake_hf_ratio": float(np.mean(fake_hf_ratio)),
        "real_hf_ratio": float(np.mean(real_hf_ratio)),
        "hf_excess_ratio": float(np.mean(fake_hf_ratio) / (np.mean(real_hf_ratio) + 1e-8)),
        "fake_lf_ratio": float(np.mean(fake_lf_ratio)),
        "real_lf_ratio": float(np.mean(real_lf_ratio)),
        "lf_recovery_ratio": float(np.mean(fake_lf_ratio) / (np.mean(real_lf_ratio) + 1e-8)),
        "target_centroid_mae_norm": float(np.mean(target_centroid_mae_norm)) if target_centroid_mae_norm else float("nan"),
        "target_hf_mae": float(np.mean(target_hf_mae)) if target_hf_mae else float("nan"),
        "target_lf_mae": float(np.mean(target_lf_mae)) if target_lf_mae else float("nan"),
        "fake_dynamic_range_db": float(np.mean(fake_dr)),
        "real_dynamic_range_db": float(np.mean(real_dr)),
        "style_pred_entropy_norm": pred_entropy,
        "mfcc_cos_mean_fake": _pairwise_cos_mean(mfcc_fake_arr),
        "mfcc_cos_mean_real": _pairwise_cos_mean(mfcc_real_arr),
        "mcd_mean_db": float(np.nanmean(mcd_vals)),
        "ssm_mae": float(np.nanmean(ssm_err_vals)),
        "fad_mfcc_proxy": float(fad_mfcc_proxy),
        "intra_target_mfcc_cos": fake_intra_target,
        "real_intra_target_mfcc_cos": real_intra_target,
        "intra_target_diversity_ratio": intra_target_div_ratio,
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    gate = Lab4Gate()
    rows: List[Dict] = []
    detailed: Dict[str, Dict] = {}

    for run_name in args.runs:
        run_dir = runs_root / run_name
        row = audit_one_run(run_dir=run_dir, max_eval_samples=int(args.max_eval_samples))
        checks = _lab4_passes(row, gate)
        row.update({f"pass_{k}": v for k, v in checks.items()})
        rows.append(row)
        detailed[run_name] = {"metrics": row, "checks": checks}

    df = pd.DataFrame(rows).sort_values("run").reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    report = {"gate": gate.__dict__, "runs": detailed}
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved CSV: {args.output_csv.resolve()}")
    print(f"Saved JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
