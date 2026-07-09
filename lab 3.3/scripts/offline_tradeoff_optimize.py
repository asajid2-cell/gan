from __future__ import annotations

import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB31_SCRIPTS = REPO_ROOT / "lab 3.1" / "scripts"
if str(LAB31_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB31_SCRIPTS))

from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_diffusion_data import load_diffusion_cache
from dggr.lab3_mert_bridge import FrozenMERT

import diffusion_downloads_batch as ddb
from diffusion_longform_settings_sweep import DiffusionSettingsSweepConfig
from run_bestpt_targeted_downloads_sweep import targeted_bestpt_settings_panel


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _safe_load_audio(path: Path, *, sr: int = 22050, offset: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True, offset=float(offset), duration=duration)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return y.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _audio_diagnostics(y: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    if y.size == 0:
        return {
            "rms_mean": 0.0,
            "dynamic_range_db": 0.0,
            "zcr": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "hf_ratio": 0.0,
            "vocal_ratio": 0.0,
            "hf_roughness": 0.0,
            "clip_frac": 0.0,
            "onset_strength_mean": 0.0,
            "onset_density": 0.0,
            "start_spectral_flatness": 0.0,
            "start_hf_ratio": 0.0,
            "start_hf_roughness": 0.0,
            "start_clip_frac": 0.0,
            "start_peak_to_rms": 0.0,
            "start_onset_burst": 0.0,
        }

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-8
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = float(np.mean(S))
    hf_mask = freqs >= 6000.0
    vocal_mask = (freqs >= 250.0) & (freqs <= 4000.0)

    hf_env = np.mean(S[hf_mask], axis=0) if np.any(hf_mask) else np.zeros(S.shape[1], dtype=np.float32)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512).squeeze()
    rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=1.0)
    onset = librosa.onset.onset_strength(y=y, sr=sr)

    start_n = min(len(y), int(sr * 1.5))
    y_start = y[:start_n]
    if y_start.size >= 1024:
        S_start = np.abs(librosa.stft(y_start, n_fft=2048, hop_length=512)) + 1e-8
        start_total = float(np.mean(S_start))
        start_hf_env = np.mean(S_start[hf_mask], axis=0) if np.any(hf_mask) else np.zeros(S_start.shape[1], dtype=np.float32)
        start_onset = librosa.onset.onset_strength(y=y_start, sr=sr)
        start_rms = librosa.feature.rms(y=y_start, frame_length=2048, hop_length=512).squeeze()
        start_rms_mean = float(np.mean(start_rms)) if start_rms.size else 0.0
        start_peak = float(np.max(np.abs(y_start))) if y_start.size else 0.0
        start_flatness = float(librosa.feature.spectral_flatness(S=S_start).mean())
        start_hf_ratio = float(np.mean(S_start[hf_mask]) / start_total) if np.any(hf_mask) else 0.0
        start_hf_roughness = float(np.mean(np.abs(np.diff(start_hf_env, n=2))) / (np.mean(np.abs(start_hf_env)) + 1e-8)) if start_hf_env.size > 2 else 0.0
        start_clip_frac = float(np.mean(np.abs(y_start) >= 0.98))
        start_peak_to_rms = float(start_peak / (start_rms_mean + 1e-6))
        start_onset_burst = float(np.max(start_onset) / (float(np.mean(start_onset)) + 1e-6)) if start_onset.size else 0.0
    else:
        start_flatness = 0.0
        start_hf_ratio = 0.0
        start_hf_roughness = 0.0
        start_clip_frac = 0.0
        start_peak_to_rms = 0.0
        start_onset_burst = 0.0

    return {
        "rms_mean": float(np.mean(rms)) if rms.size else 0.0,
        "dynamic_range_db": float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)) if rms_db.size else 0.0,
        "zcr": float(librosa.feature.zero_crossing_rate(y).mean()),
        "spectral_centroid": float(librosa.feature.spectral_centroid(S=S, sr=sr).mean()),
        "spectral_flatness": float(librosa.feature.spectral_flatness(S=S).mean()),
        "hf_ratio": float(np.mean(S[hf_mask]) / total) if np.any(hf_mask) else 0.0,
        "vocal_ratio": float(np.mean(S[vocal_mask]) / total) if np.any(vocal_mask) else 0.0,
        "hf_roughness": float(np.mean(np.abs(np.diff(hf_env, n=2))) / (np.mean(np.abs(hf_env)) + 1e-8)) if hf_env.size > 2 else 0.0,
        "clip_frac": float(np.mean(np.abs(y) >= 0.98)),
        "onset_strength_mean": float(np.mean(onset)) if onset.size else 0.0,
        "onset_density": float(np.mean(onset > (float(np.mean(onset)) + float(np.std(onset))))) if onset.size else 0.0,
        "start_spectral_flatness": start_flatness,
        "start_hf_ratio": start_hf_ratio,
        "start_hf_roughness": start_hf_roughness,
        "start_clip_frac": start_clip_frac,
        "start_peak_to_rms": start_peak_to_rms,
        "start_onset_burst": start_onset_burst,
    }


def _paired_audio_metrics(y_src: np.ndarray, y_gen: np.ndarray, sr: int = 22050) -> Dict[str, float]:
    n = min(len(y_src), len(y_gen))
    if n <= 2048:
        return {"chroma_cos": 0.0, "onset_corr": 0.0}
    y_src = y_src[:n]
    y_gen = y_gen[:n]
    c_src = librosa.feature.chroma_cqt(y=y_src, sr=sr)
    c_gen = librosa.feature.chroma_cqt(y=y_gen, sr=sr)
    frames = min(c_src.shape[1], c_gen.shape[1])
    c_src = c_src[:, :frames]
    c_gen = c_gen[:, :frames]
    frame_cos = [_cosine_similarity(c_src[:, i], c_gen[:, i]) for i in range(frames)] if frames else [0.0]
    o_src = librosa.onset.onset_strength(y=y_src, sr=sr)
    o_gen = librosa.onset.onset_strength(y=y_gen, sr=sr)
    f = min(len(o_src), len(o_gen))
    onset_corr = _corrcoef(o_src[:f], o_gen[:f]) if f > 1 else 0.0
    return {"chroma_cos": float(np.mean(frame_cos)) if frame_cos else 0.0, "onset_corr": float(onset_corr)}


REALISM_FEATURES = [
    "boundary_mel_mse_mean",
    "boundary_disc_db_mean",
    "gen_spectral_flatness",
    "gen_hf_roughness",
    "gen_clip_frac",
    "gen_start_spectral_flatness",
    "gen_start_hf_roughness",
    "gen_start_clip_frac",
    "gen_start_peak_to_rms",
    "gen_start_onset_burst",
    "dynamic_range_drift",
    "hf_ratio_drift",
    "vocal_ratio_drift",
    "chroma_cos",
    "onset_corr",
]


@dataclass
class OptimizeConfig:
    desktop_root: Path = Path.home() / "Desktop" / "dggr_offline_optimization"
    sweep_manifest: Path = REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_bestpt_targeted_sweep" / "20260328_102330" / "manifest.csv"
    sweep_settings: Path = REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_bestpt_targeted_sweep" / "20260328_102330" / "settings_panel.json"
    bad_roots: Tuple[Path, ...] = (
        REPO_ROOT / "lab 3.3" / "outputs" / "diffusion_downloads_best_finetune" / "run_20260328_185852" / "epoch_samples",
        REPO_ROOT / "lab 3.3" / "outputs" / "diffusion_stage1_warble_mixed" / "run_20260330_044057" / "epoch_samples",
    )
    cache_dir: Path = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    checkpoint_path: Path = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt"
    lab1_checkpoint: Path = REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    target_genre: str = "baroque_classical"
    source_seconds: float = 36.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    ddim_steps: int = 50
    n_validation_songs: int = 2
    seed: int = 328


def _rank_norm(values: Sequence[float]) -> Dict[int, float]:
    if not values:
        return {}
    order = np.argsort(np.asarray(values))
    out: Dict[int, float] = {}
    denom = max(1, len(values) - 1)
    for rank, idx in enumerate(order):
        out[int(idx)] = float(rank) / float(denom)
    return out


def _load_settings_map(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["label"]): row for row in data}


def _build_mert_centroids(cfg: OptimizeConfig, out_dir: Path, seconds: float = 5.0, per_genre: int = 24) -> Dict[str, np.ndarray]:
    cache_path = out_dir / "mert_genre_centroids.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    index_df, arrays, genre_to_idx, _ = load_diffusion_cache(cfg.cache_dir, mmap=True)
    mert = FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=seconds, device="auto", layer=-1)
    genre_idx_arr = np.asarray(arrays["genre_idx"], dtype=np.int64)
    centroids: Dict[str, np.ndarray] = {}
    rng = np.random.default_rng(cfg.seed)
    for genre_name, genre_idx in sorted(genre_to_idx.items(), key=lambda kv: kv[1]):
        rows = np.flatnonzero(genre_idx_arr == int(genre_idx))
        if rows.size == 0:
            continue
        choose = rows if rows.size <= per_genre else rng.choice(rows, size=per_genre, replace=False)
        feats: List[np.ndarray] = []
        for ridx in choose.tolist():
            row = index_df.iloc[int(ridx)]
            y = load_audio_chunk(
                path=Path(str(row["path"])),
                sample_rate=int(mert.cfg.sample_rate),
                seconds=seconds,
                start_sec=float(row.get("start_sec", 0.0)),
            ).astype(np.float32)
            feats.append(mert.extract_features(y))
        centroid = np.mean(np.stack(feats, axis=0), axis=0).astype(np.float32)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        centroids[str(genre_name)] = centroid
    np.savez(cache_path, **centroids)
    return centroids


def _mert_feat_for_audio(mert: FrozenMERT, path: Path, *, duration: float = 8.0) -> np.ndarray:
    y = _safe_load_audio(path, sr=int(mert.cfg.sample_rate), offset=0.0, duration=duration)
    feat = mert.extract_features(y)
    feat = feat / (np.linalg.norm(feat) + 1e-8)
    return feat.astype(np.float32)


def _analyze_candidate(
    *,
    source_wav: Path,
    generated_wav: Path,
    target_genre: str,
    mert: FrozenMERT,
    genre_centroids: Dict[str, np.ndarray],
    coherence_json: Optional[Path] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y_src = _safe_load_audio(source_wav, sr=22050)
    y_gen = _safe_load_audio(generated_wav, sr=22050)
    src_diag = _audio_diagnostics(y_src)
    gen_diag = _audio_diagnostics(y_gen)
    pair = _paired_audio_metrics(y_src, y_gen)
    metrics: Dict[str, Any] = {}
    if coherence_json and coherence_json.exists():
        metrics.update(json.loads(coherence_json.read_text(encoding="utf-8")))
    else:
        metrics["boundary_mel_mse_mean"] = 0.0
        metrics["boundary_disc_db_mean"] = 0.0

    src_feat = _mert_feat_for_audio(mert, source_wav)
    gen_feat = _mert_feat_for_audio(mert, generated_wav)
    tgt_centroid = genre_centroids[target_genre]
    style_target_cos_src = _cosine_similarity(src_feat, tgt_centroid)
    style_target_cos_gen = _cosine_similarity(gen_feat, tgt_centroid)
    src_gen_cos = _cosine_similarity(src_feat, gen_feat)
    movement = 1.0 - src_gen_cos

    out = dict(meta or {})
    out.update(metrics)
    out.update({f"src_{k}": v for k, v in src_diag.items()})
    out.update({f"gen_{k}": v for k, v in gen_diag.items()})
    out.update(pair)
    out["dynamic_range_drift"] = abs(float(gen_diag["dynamic_range_db"]) - float(src_diag["dynamic_range_db"]))
    out["hf_ratio_drift"] = abs(float(gen_diag["hf_ratio"]) - float(src_diag["hf_ratio"]))
    out["vocal_ratio_drift"] = abs(float(gen_diag["vocal_ratio"]) - float(src_diag["vocal_ratio"]))
    out["style_target_cos_src"] = style_target_cos_src
    out["style_target_cos_gen"] = style_target_cos_gen
    out["style_gain"] = style_target_cos_gen - style_target_cos_src
    out["src_gen_cos"] = src_gen_cos
    out["movement"] = movement
    return out


def _collect_bad_examples(cfg: OptimizeConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in cfg.bad_roots:
        if not root.exists():
            continue
        for gen_path in root.rglob("*_to_*.wav"):
            if gen_path.name.endswith("_source.wav"):
                continue
            prefix = gen_path.name.split("_to_")[0]
            source_path = gen_path.with_name(f"{prefix}_source.wav")
            if not source_path.exists():
                continue
            target_genre = gen_path.stem.split("_to_")[-1]
            rows.append(
                {
                    "label_group": "bad_finetune",
                    "setting_label": "bad_finetune",
                    "source_wav": source_path,
                    "generated_wav": gen_path,
                    "target_genre": target_genre,
                    "metrics_json": None,
                }
            )
    return rows


def _collect_sweep_examples(cfg: OptimizeConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with cfg.sweep_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "label_group": "bestpt_sweep",
                    "setting_label": str(row["setting_label"]),
                    "setting_note": str(row.get("setting_note", "")),
                    "source_wav": Path(str(row["source_wav"])),
                    "generated_wav": Path(str(row["generated_wav"])),
                    "target_genre": str(row["target_genre"]),
                    "metrics_json": Path(str(row["metrics_json"])),
                    "source_audio": str(row["source_audio"]),
                    "job_idx": int(row["job_idx"]),
                }
            )
    return rows


def _fit_realism_classifier(df: pd.DataFrame) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    use = df.dropna(subset=REALISM_FEATURES + ["label"])
    X = use[REALISM_FEATURES].to_numpy(dtype=np.float32)
    y = use["label"].to_numpy(dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=328, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=2,
        random_state=328,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return clf, {"holdout_auc": float(auc)}


def _apply_tradeoff_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    style_raw = df["style_gain"].to_numpy(dtype=np.float32) + 0.15 * df["movement"].to_numpy(dtype=np.float32)
    content_raw = 0.7 * df["chroma_cos"].to_numpy(dtype=np.float32) + 0.3 * np.clip(df["onset_corr"].to_numpy(dtype=np.float32), 0.0, 1.0)
    warble_raw = (
        1.20 * df["gen_start_hf_roughness"].to_numpy(dtype=np.float32)
        + 1.00 * df["gen_start_spectral_flatness"].to_numpy(dtype=np.float32)
        + 0.80 * df["gen_hf_roughness"].to_numpy(dtype=np.float32)
        + 0.50 * df["gen_spectral_flatness"].to_numpy(dtype=np.float32)
        + 0.45 * df["hf_ratio_drift"].to_numpy(dtype=np.float32)
        + 0.35 * df["vocal_ratio_drift"].to_numpy(dtype=np.float32)
    )
    style_norm = _rank_norm(style_raw.tolist())
    content_norm = _rank_norm(content_raw.tolist())
    warble_norm = _rank_norm(warble_raw.tolist())
    df["style_term"] = [style_norm[i] for i in range(len(df))]
    df["content_term"] = [content_norm[i] for i in range(len(df))]
    df["warble_penalty"] = [warble_norm[i] for i in range(len(df))]
    df["tradeoff_score"] = (
        0.60 * df["realism_prob"].to_numpy(dtype=np.float32)
        + 0.25 * df["style_term"].to_numpy(dtype=np.float32)
        + 0.15 * df["content_term"].to_numpy(dtype=np.float32)
        - 0.10 * df["warble_penalty"].to_numpy(dtype=np.float32)
    )
    return df


def _build_validation_jobs(cfg: OptimizeConfig, seen_sources: Sequence[str]) -> List[Dict[str, Any]]:
    rows = ddb.discover_download_audio(Path.home() / "Downloads")
    seen = {str(Path(x)).lower() for x in seen_sources}
    usable: List[Path] = []
    for row in rows:
        path = Path(str(row["path"]))
        if str(path).lower() in seen:
            continue
        if float(row.get("duration_seconds") or 0.0) < float(cfg.source_seconds + 2.0):
            continue
        usable.append(path)
    rng = random.Random(cfg.seed + 17)
    rng.shuffle(usable)
    jobs: List[Dict[str, Any]] = []
    for i, path in enumerate(usable[: int(cfg.n_validation_songs)]):
        duration = librosa.get_duration(path=str(path))
        start = max(0.0, min(duration - cfg.source_seconds - 0.25, duration * 0.35))
        jobs.append(
            {
                "job_idx": i,
                "source_audio": str(path),
                "source_genre": str(ddb.infer_source_genre(path)),
                "target_genre": cfg.target_genre,
                "start_sec": round(float(start), 3),
            }
        )
    return jobs


def _run_longform_job(
    *,
    cfg: OptimizeConfig,
    setting: Dict[str, Any],
    job: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    cmd = [
        "python",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(cfg.cache_dir),
        "--checkpoint", str(cfg.checkpoint_path),
        "--lab1-checkpoint", str(cfg.lab1_checkpoint),
        "--source-audio", str(job["source_audio"]),
        "--source-genre", str(job["source_genre"]),
        "--target-genre", str(job["target_genre"]),
        "--source-start-sec", str(job["start_sec"]),
        "--source-seconds", str(cfg.source_seconds),
        "--out-dir", str(out_dir),
        "--chunk-seconds", str(cfg.chunk_seconds),
        "--overlap-seconds", str(cfg.overlap_seconds),
        "--n-frames", str(cfg.n_frames),
        "--ddim-steps", str(cfg.ddim_steps),
        "--assemble-domain", "mel",
        "--device", "auto",
        "--seed", str(cfg.seed + int(job["job_idx"])),
        "--t-start", str(setting["t_start"]),
        "--t-start-end", str(setting["t_start_end"]),
        "--reanchor-every", str(setting["reanchor_every"]),
        "--reanchor-t-start", str(setting["reanchor_t_start"]),
        "--guidance-scale", str(setting["guidance_scale"]),
        "--style-strength", str(setting["style_strength"]),
        "--prefix-blend", str(setting["prefix_blend"]),
        "--source-prefix-blend", str(setting["source_prefix_blend"]),
        "--source-mel-blend", str(setting["source_mel_blend"]),
        "--hf-source-blend", str(setting["hf_source_blend"]),
        "--hf-start-bin", str(setting["hf_start_bin"]),
        "--mel-time-smooth", str(setting["mel_time_smooth"]),
        "--mel-freq-smooth", str(setting["mel_freq_smooth"]),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Longform job failed: {' '.join(cmd)}")
    return {
        "setting_label": setting["label"],
        "source_audio": job["source_audio"],
        "source_genre": job["source_genre"],
        "target_genre": job["target_genre"],
        "start_sec": job["start_sec"],
        "output_dir": out_dir,
        "source_wav": out_dir / "source.wav",
        "generated_wav": out_dir / "longform_coherent.wav",
        "metrics_json": out_dir / "coherence_metrics.json",
    }


def _make_refinement_panel(base_setting: Dict[str, Any]) -> List[Dict[str, Any]]:
    base = dict(base_setting)
    panel = [dict(base)]

    guard = dict(base)
    guard["label"] = f"{base['label']}_guard"
    guard["note"] = "Slightly stronger HF/source guard around the winner."
    guard["guidance_scale"] = max(1.6, float(base["guidance_scale"]) - 0.05)
    guard["style_strength"] = max(0.50, float(base["style_strength"]) - 0.01)
    guard["source_prefix_blend"] = min(0.60, float(base["source_prefix_blend"]) + 0.01)
    guard["source_mel_blend"] = min(0.16, float(base["source_mel_blend"]) + 0.01)
    guard["hf_source_blend"] = min(0.30, float(base["hf_source_blend"]) + 0.02)
    guard["hf_start_bin"] = int(max(48, min(int(base["hf_start_bin"]), 54)))
    panel.append(guard)

    style_edge = dict(base)
    style_edge["label"] = f"{base['label']}_style"
    style_edge["note"] = "Slightly more style around the winner."
    style_edge["guidance_scale"] = min(2.10, float(base["guidance_scale"]) + 0.05)
    style_edge["style_strength"] = min(0.80, float(base["style_strength"]) + 0.02)
    style_edge["source_prefix_blend"] = max(0.35, float(base["source_prefix_blend"]) - 0.01)
    style_edge["source_mel_blend"] = max(0.05, float(base["source_mel_blend"]) - 0.01)
    style_edge["hf_source_blend"] = max(0.12, float(base["hf_source_blend"]) - 0.01)
    panel.append(style_edge)

    bridge = dict(base)
    bridge["label"] = f"{base['label']}_bridge"
    bridge["note"] = "Middle bridge between the winner and extra guard."
    bridge["guidance_scale"] = float(base["guidance_scale"])
    bridge["style_strength"] = min(0.80, float(base["style_strength"]) + 0.01)
    bridge["source_prefix_blend"] = min(0.60, float(base["source_prefix_blend"]) + 0.005)
    bridge["source_mel_blend"] = min(0.16, float(base["source_mel_blend"]) + 0.005)
    bridge["hf_source_blend"] = min(0.30, float(base["hf_source_blend"]) + 0.01)
    panel.append(bridge)

    dewarble = dict(base)
    dewarble["label"] = f"{base['label']}_dewarble"
    dewarble["note"] = "Slightly more source/HF anchoring to shave robotic warble."
    dewarble["guidance_scale"] = max(1.75, float(base["guidance_scale"]) - 0.05)
    dewarble["style_strength"] = max(0.58, float(base["style_strength"]) - 0.02)
    dewarble["source_prefix_blend"] = min(0.60, float(base["source_prefix_blend"]) + 0.02)
    dewarble["source_mel_blend"] = min(0.16, float(base["source_mel_blend"]) + 0.01)
    dewarble["hf_source_blend"] = min(0.30, float(base["hf_source_blend"]) + 0.03)
    dewarble["hf_start_bin"] = int(max(50, min(int(base["hf_start_bin"]), 54)))
    panel.append(dewarble)

    dewarble_soft = dict(base)
    dewarble_soft["label"] = f"{base['label']}_dewarble_soft"
    dewarble_soft["note"] = "Slightly softer edit depth with the same long-form shape."
    dewarble_soft["t_start"] = max(245, int(base["t_start"]) - 5)
    dewarble_soft["t_start_end"] = max(186, int(base["t_start_end"]) - 4)
    dewarble_soft["guidance_scale"] = max(1.75, float(base["guidance_scale"]) - 0.05)
    dewarble_soft["style_strength"] = max(0.60, float(base["style_strength"]) - 0.02)
    dewarble_soft["source_prefix_blend"] = min(0.60, float(base["source_prefix_blend"]) + 0.015)
    dewarble_soft["source_mel_blend"] = min(0.16, float(base["source_mel_blend"]) + 0.01)
    dewarble_soft["hf_source_blend"] = min(0.30, float(base["hf_source_blend"]) + 0.02)
    dewarble_soft["hf_start_bin"] = int(max(50, min(int(base["hf_start_bin"]), 54)))
    panel.append(dewarble_soft)

    dewarble_hf = dict(base)
    dewarble_hf["label"] = f"{base['label']}_dewarble_hf"
    dewarble_hf["note"] = "Sharper HF guard while preserving most of the winner's movement."
    dewarble_hf["guidance_scale"] = max(1.80, float(base["guidance_scale"]) - 0.02)
    dewarble_hf["style_strength"] = max(0.60, float(base["style_strength"]) - 0.01)
    dewarble_hf["source_prefix_blend"] = min(0.60, float(base["source_prefix_blend"]) + 0.01)
    dewarble_hf["source_mel_blend"] = min(0.16, float(base["source_mel_blend"]) + 0.01)
    dewarble_hf["hf_source_blend"] = min(0.30, float(base["hf_source_blend"]) + 0.04)
    dewarble_hf["hf_start_bin"] = 52
    panel.append(dewarble_hf)
    return panel


def main() -> None:
    cfg = OptimizeConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.desktop_root / f"offline_opt_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str), encoding="utf-8")

    settings_map = _load_settings_map(cfg.sweep_settings)
    sweep_rows = _collect_sweep_examples(cfg)
    bad_rows = _collect_bad_examples(cfg)

    mert = FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=8.0, device="auto", layer=-1)
    centroids = _build_mert_centroids(cfg, out_dir=out_dir, seconds=5.0, per_genre=24)

    analyzed: List[Dict[str, Any]] = []
    for row in sweep_rows + bad_rows:
        analyzed.append(
            _analyze_candidate(
                source_wav=Path(row["source_wav"]),
                generated_wav=Path(row["generated_wav"]),
                target_genre=str(row["target_genre"]),
                mert=mert,
                genre_centroids=centroids,
                coherence_json=Path(row["metrics_json"]) if row.get("metrics_json") else None,
                meta=row,
            )
        )
    df = pd.DataFrame(analyzed)
    df["label"] = (df["label_group"] == "bestpt_sweep").astype(int)
    clf, clf_metrics = _fit_realism_classifier(df)
    df["realism_prob"] = clf.predict_proba(df[REALISM_FEATURES].to_numpy(dtype=np.float32))[:, 1]
    df["split"] = np.where(df["label"] == 1, "sweep", "bad")

    sweep_df = _apply_tradeoff_score(df[df["label"] == 1].reset_index(drop=True))
    bad_df = df[df["label"] == 0].copy()

    label_means = (
        sweep_df.groupby("setting_label")[["tradeoff_score", "realism_prob", "style_gain", "movement", "chroma_cos", "onset_corr"]]
        .mean()
        .sort_values("tradeoff_score", ascending=False)
        .reset_index()
    )
    baseline_label = "best_mid_230_a"
    aggressive_label = "best_mid_275_b"
    top_labels = label_means["setting_label"].head(3).tolist()
    compare_labels = []
    for label in [baseline_label, aggressive_label] + top_labels:
        if label not in compare_labels:
            compare_labels.append(label)

    seen_sources = [str(row["source_audio"]) for row in sweep_rows]
    validation_jobs = _build_validation_jobs(cfg, seen_sources=seen_sources)
    validation_root = out_dir / "validation_generations"
    validation_rows: List[Dict[str, Any]] = []
    for label in compare_labels:
        setting = settings_map[label]
        for job in validation_jobs:
            job_tag = f"{int(job['job_idx']):02d}_{_slug(Path(str(job['source_audio'])).stem)[:42]}__to__{_slug(str(job['target_genre']))}"
            out_subdir = validation_root / label / job_tag
            row = _run_longform_job(cfg=cfg, setting=setting, job=job, out_dir=out_subdir)
            validation_rows.append(row)

    val_analyzed: List[Dict[str, Any]] = []
    for row in validation_rows:
        meta = {
            "label_group": "validation",
            "setting_label": row["setting_label"],
            "source_audio": row["source_audio"],
            "source_genre": row["source_genre"],
            "target_genre": row["target_genre"],
            "start_sec": row["start_sec"],
            "output_dir": str(row["output_dir"]),
        }
        val_analyzed.append(
            _analyze_candidate(
                source_wav=Path(row["source_wav"]),
                generated_wav=Path(row["generated_wav"]),
                target_genre=str(row["target_genre"]),
                mert=mert,
                genre_centroids=centroids,
                coherence_json=Path(row["metrics_json"]),
                meta=meta,
            )
        )
    val_df = pd.DataFrame(val_analyzed)
    val_df["realism_prob"] = clf.predict_proba(val_df[REALISM_FEATURES].to_numpy(dtype=np.float32))[:, 1]
    val_df = _apply_tradeoff_score(val_df.reset_index(drop=True))
    val_means = (
        val_df.groupby("setting_label")[["tradeoff_score", "realism_prob", "style_gain", "movement", "chroma_cos", "onset_corr"]]
        .mean()
        .sort_values("tradeoff_score", ascending=False)
        .reset_index()
    )

    winner_label = str(val_means.iloc[0]["setting_label"])
    baseline_metrics = val_means[val_means["setting_label"] == baseline_label].iloc[0].to_dict()
    winner_metrics = val_means[val_means["setting_label"] == winner_label].iloc[0].to_dict()

    refinement_panel = _make_refinement_panel(settings_map[winner_label])
    refinement_root = out_dir / "refinement_generations"
    refinement_rows: List[Dict[str, Any]] = []
    for setting in refinement_panel:
        for job in validation_jobs:
            job_tag = f"{int(job['job_idx']):02d}_{_slug(Path(str(job['source_audio'])).stem)[:42]}__to__{_slug(str(job['target_genre']))}"
            out_subdir = refinement_root / str(setting["label"]) / job_tag
            row = _run_longform_job(cfg=cfg, setting=setting, job=job, out_dir=out_subdir)
            refinement_rows.append(row)

    ref_analyzed: List[Dict[str, Any]] = []
    for row in refinement_rows:
        meta = {
            "label_group": "refinement",
            "setting_label": row["setting_label"],
            "source_audio": row["source_audio"],
            "source_genre": row["source_genre"],
            "target_genre": row["target_genre"],
            "start_sec": row["start_sec"],
            "output_dir": str(row["output_dir"]),
        }
        ref_analyzed.append(
            _analyze_candidate(
                source_wav=Path(row["source_wav"]),
                generated_wav=Path(row["generated_wav"]),
                target_genre=str(row["target_genre"]),
                mert=mert,
                genre_centroids=centroids,
                coherence_json=Path(row["metrics_json"]),
                meta=meta,
            )
        )
    ref_df = pd.DataFrame(ref_analyzed)
    ref_df["realism_prob"] = clf.predict_proba(ref_df[REALISM_FEATURES].to_numpy(dtype=np.float32))[:, 1]
    ref_df = _apply_tradeoff_score(ref_df.reset_index(drop=True))
    ref_means = (
        ref_df.groupby("setting_label")[["tradeoff_score", "realism_prob", "style_gain", "movement", "chroma_cos", "onset_corr"]]
        .mean()
        .sort_values("tradeoff_score", ascending=False)
        .reset_index()
    )
    final_winner_label = str(ref_means.iloc[0]["setting_label"])
    final_winner_metrics = ref_means.iloc[0].to_dict()

    summary = {
        "classifier_metrics": clf_metrics,
        "positive_count": int((df["label"] == 1).sum()),
        "negative_count": int((df["label"] == 0).sum()),
        "sweep_mean_realism_prob": float(sweep_df["realism_prob"].mean()),
        "bad_mean_realism_prob": float(bad_df["realism_prob"].mean()) if not bad_df.empty else None,
        "global_label_ranking": label_means.to_dict(orient="records"),
        "validation_jobs": validation_jobs,
        "validation_label_ranking": val_means.to_dict(orient="records"),
        "baseline_label": baseline_label,
        "aggressive_label": aggressive_label,
        "winner_label": winner_label,
        "refinement_label_ranking": ref_means.to_dict(orient="records"),
        "final_winner_label": final_winner_label,
        "winner_vs_baseline": {
            "tradeoff_delta": float(winner_metrics["tradeoff_score"] - baseline_metrics["tradeoff_score"]),
            "realism_delta": float(winner_metrics["realism_prob"] - baseline_metrics["realism_prob"]),
            "style_gain_delta": float(winner_metrics["style_gain"] - baseline_metrics["style_gain"]),
            "movement_delta": float(winner_metrics["movement"] - baseline_metrics["movement"]),
            "chroma_cos_delta": float(winner_metrics["chroma_cos"] - baseline_metrics["chroma_cos"]),
            "onset_corr_delta": float(winner_metrics["onset_corr"] - baseline_metrics["onset_corr"]),
        },
        "final_winner_vs_baseline": {
            "tradeoff_delta": float(final_winner_metrics["tradeoff_score"] - baseline_metrics["tradeoff_score"]),
            "realism_delta": float(final_winner_metrics["realism_prob"] - baseline_metrics["realism_prob"]),
            "style_gain_delta": float(final_winner_metrics["style_gain"] - baseline_metrics["style_gain"]),
            "movement_delta": float(final_winner_metrics["movement"] - baseline_metrics["movement"]),
            "chroma_cos_delta": float(final_winner_metrics["chroma_cos"] - baseline_metrics["chroma_cos"]),
            "onset_corr_delta": float(final_winner_metrics["onset_corr"] - baseline_metrics["onset_corr"]),
        },
        "selected_outputs": ref_df[ref_df["setting_label"] == final_winner_label][["source_audio", "output_dir", "tradeoff_score", "realism_prob", "style_gain"]].to_dict(orient="records"),
    }

    df.to_csv(out_dir / "judge_training_rows.csv", index=False)
    sweep_df.to_csv(out_dir / "sweep_ranked.csv", index=False)
    label_means.to_csv(out_dir / "sweep_label_ranking.csv", index=False)
    val_df.to_csv(out_dir / "validation_ranked.csv", index=False)
    val_means.to_csv(out_dir / "validation_label_ranking.csv", index=False)
    ref_df.to_csv(out_dir / "refinement_ranked.csv", index=False)
    ref_means.to_csv(out_dir / "refinement_label_ranking.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    winner_config = next((row for row in refinement_panel if str(row["label"]) == final_winner_label), settings_map[winner_label])
    (out_dir / "winner_config.json").write_text(json.dumps(winner_config, indent=2), encoding="utf-8")

    notes = [
        f"Judge holdout AUC: {clf_metrics['holdout_auc']:.4f}",
        f"Mean realism prob, sweep positives: {float(sweep_df['realism_prob'].mean()):.4f}",
        f"Mean realism prob, bad fine-tunes: {float(bad_df['realism_prob'].mean()):.4f}" if not bad_df.empty else "No bad fine-tune pool.",
        f"Global top setting on prior sweep: {label_means.iloc[0]['setting_label']}",
        f"Validated winner on fresh songs: {winner_label}",
        f"Final refined winner: {final_winner_label}",
        f"Final winner vs baseline tradeoff delta: {summary['final_winner_vs_baseline']['tradeoff_delta']:.4f}",
        f"Final winner vs baseline style_gain delta: {summary['final_winner_vs_baseline']['style_gain_delta']:.4f}",
        f"Final winner vs baseline realism delta: {summary['final_winner_vs_baseline']['realism_delta']:.4f}",
    ]
    (out_dir / "notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
