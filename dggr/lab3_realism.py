from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import librosa
import numpy as np
import pandas as pd
from scipy import linalg

from .lab3_bridge import load_audio_chunk
from .lab3_mert_bridge import FrozenMERT


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray, mu2: np.ndarray, cov2: np.ndarray) -> float:
    """Stable Fréchet distance between two Gaussians."""
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    val = float(diff @ diff + np.trace(cov1 + cov2 - 2.0 * covmean))
    return max(0.0, val)


@dataclass(frozen=True)
class AudioStatSummary:
    centroid_hz: float
    hf_ratio: float
    lf_ratio: float
    dynamic_range_db: float


@dataclass(frozen=True)
class RealismGate:
    max_fad_mert: Optional[float] = None
    max_target_centroid_mae_norm: Optional[float] = None
    max_target_hf_mae: Optional[float] = None
    max_target_lf_mae: Optional[float] = None
    max_target_dynamic_range_mae_db: Optional[float] = None
    min_mps: Optional[float] = None
    min_style_target_acc: Optional[float] = None
    min_style_target_cos: Optional[float] = None


def summarize_audio_stats(
    audio: np.ndarray,
    sr: int,
    n_mels: int = 80,
    top_frac: float = 0.20,
    bot_frac: float = 0.20,
) -> AudioStatSummary:
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=int(sr),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=int(n_mels),
        fmin=20,
        fmax=int(sr // 2),
        power=2.0,
    ).astype(np.float32)
    mel = np.clip(mel, a_min=1e-10, a_max=None)
    mel_freqs = librosa.mel_frequencies(n_mels=int(n_mels), fmin=20, fmax=sr / 2.0)
    w = mel.sum(axis=1)
    centroid_hz = float((mel_freqs * w).sum() / (w.sum() + 1e-8))
    total = float(mel.sum() + 1e-8)
    top = max(1, int(round(len(mel_freqs) * float(top_frac))))
    bot = max(1, int(round(len(mel_freqs) * float(bot_frac))))
    hf_ratio = float(mel[-top:, :].sum() / total)
    lf_ratio = float(mel[:bot, :].sum() / total)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    dynamic_range_db = float(np.percentile(mel_db, 95) - np.percentile(mel_db, 5))
    return AudioStatSummary(
        centroid_hz=centroid_hz,
        hf_ratio=hf_ratio,
        lf_ratio=lf_ratio,
        dynamic_range_db=dynamic_range_db,
    )


def build_balanced_transfer_plan(
    index_df: pd.DataFrame,
    genre_idx: np.ndarray,
    val_idx: np.ndarray,
    n_samples: int,
    seed: int = 328,
) -> pd.DataFrame:
    """Create a fixed source/target plan shared by every checkpoint in a sweep."""
    rng = np.random.default_rng(int(seed))
    val_idx = np.asarray(val_idx, dtype=np.int64)
    val_genre = np.asarray(genre_idx, dtype=np.int64)[val_idx]
    genres = sorted(np.unique(val_genre).tolist())
    by_genre: Dict[int, np.ndarray] = {}
    for g in genres:
        rows = val_idx[val_genre == int(g)].astype(np.int64)
        if len(rows) > 0:
            by_genre[int(g)] = rows
    if not by_genre:
        raise ValueError("No validation rows available to build transfer plan.")

    src_cycle = sorted(by_genre.keys())
    rows: List[Dict[str, int]] = []
    for sample_id in range(int(n_samples)):
        src_g = int(src_cycle[sample_id % len(src_cycle)])
        tgt_choices = [g for g in src_cycle if g != src_g]
        tgt_g = int(rng.choice(tgt_choices if tgt_choices else src_cycle))
        src_row = int(rng.choice(by_genre[src_g]))
        tgt_ref_row = int(rng.choice(by_genre[tgt_g]))
        rows.append(
            {
                "sample_id": int(sample_id),
                "source_row": int(src_row),
                "source_genre_idx": int(src_g),
                "target_genre_idx": int(tgt_g),
                "target_ref_row": int(tgt_ref_row),
            }
        )

    plan = pd.DataFrame(rows)
    if "path" in index_df.columns:
        path_col = index_df["path"]
        plan["source_path"] = path_col.iloc[plan["source_row"].to_numpy()].astype(str).tolist()
        plan["target_ref_path"] = path_col.iloc[plan["target_ref_row"].to_numpy()].astype(str).tolist()
    if "start_sec" in index_df.columns:
        start_col = index_df["start_sec"]
        plan["source_start_sec"] = start_col.iloc[plan["source_row"].to_numpy()].astype(float).tolist()
        plan["target_ref_start_sec"] = start_col.iloc[plan["target_ref_row"].to_numpy()].astype(float).tolist()
    return plan


def load_plan_audio(
    plan: pd.DataFrame,
    sample_rate: int,
    chunk_seconds: float,
) -> Dict[str, List[np.ndarray]]:
    src_audio: List[np.ndarray] = []
    ref_audio: List[np.ndarray] = []
    for row in plan.itertuples(index=False):
        src_audio.append(
            load_audio_chunk(
                path=Path(str(row.source_path)),
                sample_rate=int(sample_rate),
                seconds=float(chunk_seconds),
                start_sec=float(getattr(row, "source_start_sec", 0.0)),
            )
        )
        ref_audio.append(
            load_audio_chunk(
                path=Path(str(row.target_ref_path)),
                sample_rate=int(sample_rate),
                seconds=float(chunk_seconds),
                start_sec=float(getattr(row, "target_ref_start_sec", 0.0)),
            )
        )
    return {"source_audio": src_audio, "target_ref_audio": ref_audio}


def build_target_profiles(
    plan: pd.DataFrame,
    target_ref_audio: Sequence[np.ndarray],
    sr: int,
) -> Dict[int, AudioStatSummary]:
    grouped: Dict[int, List[AudioStatSummary]] = {}
    for row, audio in zip(plan.itertuples(index=False), target_ref_audio):
        stats = summarize_audio_stats(audio=audio, sr=int(sr))
        grouped.setdefault(int(row.target_genre_idx), []).append(stats)
    out: Dict[int, AudioStatSummary] = {}
    for genre_idx, stats_list in grouped.items():
        out[int(genre_idx)] = AudioStatSummary(
            centroid_hz=float(np.mean([s.centroid_hz for s in stats_list])),
            hf_ratio=float(np.mean([s.hf_ratio for s in stats_list])),
            lf_ratio=float(np.mean([s.lf_ratio for s in stats_list])),
            dynamic_range_db=float(np.mean([s.dynamic_range_db for s in stats_list])),
        )
    return out


def extract_mert_embeddings(
    mert: FrozenMERT,
    audio_list: Sequence[np.ndarray],
    sr: int,
    batch_size: int = 8,
) -> np.ndarray:
    rows: List[np.ndarray] = []
    converted: List[np.ndarray] = []
    for y in audio_list:
        converted.append(FrozenMERT.resample_audio(np.asarray(y, dtype=np.float32), sr_from=int(sr), sr_to=int(mert.cfg.sample_rate)))
    for start in range(0, len(converted), int(max(1, batch_size))):
        batch = converted[start : start + int(max(1, batch_size))]
        rows.append(mert.extract_features_batch(batch))
    if not rows:
        raise ValueError("No audio provided for MERT embedding extraction.")
    return np.concatenate(rows, axis=0).astype(np.float32)


def compute_mert_fad(
    mert: FrozenMERT,
    fake_audio: Sequence[np.ndarray],
    real_audio: Sequence[np.ndarray],
    sr: int,
    batch_size: int = 8,
) -> float:
    fake_emb = extract_mert_embeddings(mert=mert, audio_list=fake_audio, sr=int(sr), batch_size=int(batch_size))
    real_emb = extract_mert_embeddings(mert=mert, audio_list=real_audio, sr=int(sr), batch_size=int(batch_size))
    mu_f = np.mean(fake_emb, axis=0)
    mu_r = np.mean(real_emb, axis=0)
    cov_f = np.cov(fake_emb, rowvar=False) + 1e-6 * np.eye(fake_emb.shape[1], dtype=np.float64)
    cov_r = np.cov(real_emb, rowvar=False) + 1e-6 * np.eye(real_emb.shape[1], dtype=np.float64)
    return frechet_distance(mu_f.astype(np.float64), cov_f, mu_r.astype(np.float64), cov_r)


def evaluate_realism_metrics(
    plan: pd.DataFrame,
    fake_audio: Sequence[np.ndarray],
    target_ref_audio: Sequence[np.ndarray],
    sr: int,
    mert: FrozenMERT,
    mert_batch_size: int = 8,
) -> Dict[str, float]:
    target_profiles = build_target_profiles(plan=plan, target_ref_audio=target_ref_audio, sr=int(sr))
    centroid_mae: List[float] = []
    hf_mae: List[float] = []
    lf_mae: List[float] = []
    dr_mae: List[float] = []
    for row, fake in zip(plan.itertuples(index=False), fake_audio):
        tgt_idx = int(row.target_genre_idx)
        ref = target_profiles[tgt_idx]
        stats = summarize_audio_stats(audio=fake, sr=int(sr))
        centroid_mae.append(abs(stats.centroid_hz - ref.centroid_hz) / (ref.centroid_hz + 1e-8))
        hf_mae.append(abs(stats.hf_ratio - ref.hf_ratio))
        lf_mae.append(abs(stats.lf_ratio - ref.lf_ratio))
        dr_mae.append(abs(stats.dynamic_range_db - ref.dynamic_range_db))

    return {
        "fad_mert": float(
            compute_mert_fad(
                mert=mert,
                fake_audio=fake_audio,
                real_audio=target_ref_audio,
                sr=int(sr),
                batch_size=int(mert_batch_size),
            )
        ),
        "target_centroid_mae_norm": float(np.mean(centroid_mae)) if centroid_mae else float("nan"),
        "target_hf_mae": float(np.mean(hf_mae)) if hf_mae else float("nan"),
        "target_lf_mae": float(np.mean(lf_mae)) if lf_mae else float("nan"),
        "target_dynamic_range_mae_db": float(np.mean(dr_mae)) if dr_mae else float("nan"),
    }


def apply_realism_gate(row: Dict[str, float], gate: RealismGate) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}

    def _check_max(name: str, metric: str, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        val = float(row.get(metric, float("nan")))
        checks[name] = bool(np.isfinite(val) and val <= float(threshold))

    def _check_min(name: str, metric: str, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        val = float(row.get(metric, float("nan")))
        checks[name] = bool(np.isfinite(val) and val >= float(threshold))

    _check_max("fad_mert", "fad_mert", gate.max_fad_mert)
    _check_max("target_centroid", "target_centroid_mae_norm", gate.max_target_centroid_mae_norm)
    _check_max("target_hf", "target_hf_mae", gate.max_target_hf_mae)
    _check_max("target_lf", "target_lf_mae", gate.max_target_lf_mae)
    _check_max("target_dynamic_range", "target_dynamic_range_mae_db", gate.max_target_dynamic_range_mae_db)
    _check_min("mps", "mps", gate.min_mps)
    _check_min("style_target_acc", "style_target_acc", gate.min_style_target_acc)
    _check_min("style_target_cos", "style_target_cos", gate.min_style_target_cos)
    checks["all"] = bool(all(checks.values())) if checks else True
    return checks


def composite_realism_score(row: Dict[str, float]) -> float:
    """Lower is better. Weighted toward realism, with light penalties for transfer collapse."""
    fad = float(row.get("fad_mert", float("inf")))
    centroid = float(row.get("target_centroid_mae_norm", 0.0))
    hf = float(row.get("target_hf_mae", 0.0))
    lf = float(row.get("target_lf_mae", 0.0))
    dr = float(row.get("target_dynamic_range_mae_db", 0.0))
    mps = float(row.get("mps", 0.0))
    style_acc = float(row.get("style_target_acc", 0.0))
    style_cos = float(row.get("style_target_cos", 0.0))
    style_floor_acc = max(0.0, 0.18 - style_acc)
    style_floor_cos = max(0.0, 0.0 - style_cos)
    return (
        fad
        + 8.0 * centroid
        + 40.0 * hf
        + 20.0 * lf
        + 0.35 * dr
        + 25.0 * style_floor_acc
        + 6.0 * style_floor_cos
        - 0.75 * mps
        - 2.0 * style_acc
        - 0.75 * style_cos
    )


def rank_realism_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["realism_score"] = [composite_realism_score(r) for r in out.to_dict(orient="records")]
    sort_cols = []
    ascending = []
    if "pass_all" in out.columns:
        sort_cols.append("pass_all")
        ascending.append(False)
    sort_cols.extend(["realism_score", "fad_mert", "target_hf_mae", "target_dynamic_range_mae_db"])
    ascending.extend([True, True, True, True])
    out = out.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def save_plan(plan: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(path, index=False)
