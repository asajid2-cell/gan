from __future__ import annotations

import csv
import json
import os
import pickle
import random
import subprocess
import sys
from dataclasses import field, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB31_SCRIPTS = REPO_ROOT / "lab 3.1" / "scripts"
if str(LAB31_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB31_SCRIPTS))

import diffusion_downloads_batch as ddb
import diffusion_longform_compare as dlc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _safe_load_audio(
    path: Path,
    *,
    sr: int = 22050,
    offset: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    y, _ = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        offset=float(offset or 0.0),
        duration=duration,
    )
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
        start_hf_env = (
            np.mean(S_start[hf_mask], axis=0) if np.any(hf_mask) else np.zeros(S_start.shape[1], dtype=np.float32)
        )
        start_onset = librosa.onset.onset_strength(y=y_start, sr=sr)
        start_rms = librosa.feature.rms(y=y_start, frame_length=2048, hop_length=512).squeeze()
        start_rms_mean = float(np.mean(start_rms)) if start_rms.size else 0.0
        start_peak = float(np.max(np.abs(y_start))) if y_start.size else 0.0
        start_flatness = float(librosa.feature.spectral_flatness(S=S_start).mean())
        start_hf_ratio = float(np.mean(S_start[hf_mask]) / start_total) if np.any(hf_mask) else 0.0
        start_hf_roughness = (
            float(np.mean(np.abs(np.diff(start_hf_env, n=2))) / (np.mean(np.abs(start_hf_env)) + 1e-8))
            if start_hf_env.size > 2
            else 0.0
        )
        start_clip_frac = float(np.mean(np.abs(y_start) >= 0.98))
        start_peak_to_rms = float(start_peak / (start_rms_mean + 1e-6))
        start_onset_burst = (
            float(np.max(start_onset) / (float(np.mean(start_onset)) + 1e-6)) if start_onset.size else 0.0
        )
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
        "hf_roughness": float(np.mean(np.abs(np.diff(hf_env, n=2))) / (np.mean(np.abs(hf_env)) + 1e-8))
        if hf_env.size > 2
        else 0.0,
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

    return {
        "chroma_cos": float(np.mean(frame_cos)) if frame_cos else 0.0,
        "onset_corr": float(onset_corr),
    }


@dataclass
class PolicyConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.2" / "outputs" / "policy_runs")
    training_roots: List[Path] = field(
        default_factory=lambda: [
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_settings_sweep",
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_vocal_noise_panel",
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_run_d002_low_noise_vocal",
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_run_d002_midground",
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_run_d002_style_guard",
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_vocal_iso_d_repro",
        ]
    )
    downloads_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    n_demo_songs: int = 3
    source_seconds: float = 45.0
    candidate_checkpoint_paths: List[Path] = field(
        default_factory=lambda: [
            REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_vocal_crackle_retool" / "run_20260324_221729" / "checkpoints" / "epoch_001.pt",
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt",
        ]
    )
    baseline_safe: Tuple[str, str] = ("epoch_001.pt", "vocal_iso_d")
    baseline_style: Tuple[str, str] = ("epoch_006.pt", "style_guard_c")
    seed: int = 328

    def materialize(self) -> "PolicyConfig":
        self.output_root = Path(self.output_root)
        self.training_roots = [Path(p) for p in self.training_roots]
        self.downloads_dir = Path(self.downloads_dir)
        self.candidate_checkpoint_paths = [Path(p) for p in self.candidate_checkpoint_paths]
        return self


def _normalize_manifest_row(row: Dict[str, Any], manifest_path: Path) -> Optional[Dict[str, Any]]:
    output_dir = Path(row.get("output_dir", "")) if row.get("output_dir") else None
    generated_wav = Path(row.get("generated_wav", "")) if row.get("generated_wav") else None
    metrics_json = Path(row.get("metrics_json", "")) if row.get("metrics_json") else None

    if output_dir is None or not output_dir.exists():
        return None
    if generated_wav is None or not generated_wav.exists():
        maybe = output_dir / "longform_coherent.wav"
        if not maybe.exists():
            return None
        generated_wav = maybe
    if metrics_json is None or not metrics_json.exists():
        maybe = output_dir / "coherence_metrics.json"
        if not maybe.exists():
            return None
        metrics_json = maybe

    coherence_config = output_dir / "coherence_config.json"
    if not coherence_config.exists():
        return None

    source_audio = row.get("source_audio")
    if not source_audio:
        return None

    setting_label = row.get("setting_label") or output_dir.parent.name
    checkpoint_path = row.get("checkpoint_path") or row.get("checkpoint")
    if not checkpoint_path:
        cfg_json = json.loads(coherence_config.read_text(encoding="utf-8"))
        checkpoint_path = cfg_json.get("checkpoint")
    if not checkpoint_path:
        return None

    return {
        "manifest_path": manifest_path,
        "output_dir": output_dir,
        "generated_wav": generated_wav,
        "metrics_json": metrics_json,
        "coherence_config": coherence_config,
        "source_audio": str(source_audio),
        "target_genre": row.get("target_genre"),
        "source_genre": row.get("source_genre"),
        "start_sec": float(row.get("start_sec", 0.0)),
        "source_seconds": float(row.get("source_seconds", 45.0)),
        "setting_label": str(setting_label),
        "checkpoint_path": str(checkpoint_path),
    }


def discover_training_examples(cfg: PolicyConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in cfg.training_roots:
        if not root.exists():
            continue
        for manifest in root.rglob("manifest.csv"):
            try:
                for row in csv.DictReader(manifest.open(encoding="utf-8")):
                    out = _normalize_manifest_row(row, manifest)
                    if out:
                        rows.append(out)
            except Exception:
                continue
        for manifest in root.rglob("manifest.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for row in data:
                if not isinstance(row, dict):
                    continue
                out = _normalize_manifest_row(row, manifest)
                if out:
                    rows.append(out)
    return rows


def _setting_style_intent(cfg_json: Dict[str, Any]) -> float:
    return (
        1.10 * float(cfg_json.get("style_strength", 0.0))
        + 0.0023 * float(cfg_json.get("t_start", 0.0))
        + 0.18 * float(cfg_json.get("guidance_scale", 0.0))
        - 0.55 * float(cfg_json.get("source_prefix_blend", 0.0))
        - 0.45 * float(cfg_json.get("source_mel_blend", 0.0))
        - 0.40 * float(cfg_json.get("hf_source_blend", 0.0))
        - 0.02 * float(cfg_json.get("reanchor_every", 0.0))
    )


def build_training_table(cfg: PolicyConfig) -> pd.DataFrame:
    examples = discover_training_examples(cfg)
    audio_cache: Dict[Tuple[str, float, float], np.ndarray] = {}
    gen_cache: Dict[str, np.ndarray] = {}
    records: List[Dict[str, Any]] = []

    for row in examples:
        coherence = json.loads(Path(row["coherence_config"]).read_text(encoding="utf-8"))
        metrics = json.loads(Path(row["metrics_json"]).read_text(encoding="utf-8"))
        source_key = (
            row["source_audio"],
            float(coherence.get("source_start_sec", row["start_sec"])),
            float(coherence.get("source_seconds", row["source_seconds"])),
        )
        if source_key not in audio_cache:
            audio_cache[source_key] = _safe_load_audio(
                Path(row["source_audio"]),
                offset=float(source_key[1]),
                duration=float(source_key[2]),
            )
        y_src = audio_cache[source_key]

        gen_key = str(row["generated_wav"])
        if gen_key not in gen_cache:
            gen_cache[gen_key] = _safe_load_audio(Path(row["generated_wav"]))
        y_gen = gen_cache[gen_key]

        src_diag = _audio_diagnostics(y_src)
        gen_diag = _audio_diagnostics(y_gen)
        paired = _paired_audio_metrics(y_src, y_gen)

        checkpoint_path = Path(row["checkpoint_path"])
        checkpoint_id = f"{checkpoint_path.parent.parent.name}::{checkpoint_path.name}"
        record: Dict[str, Any] = {
            "source_audio": row["source_audio"],
            "target_genre": row["target_genre"],
            "source_genre": row["source_genre"] or ddb.infer_source_genre(Path(row["source_audio"])),
            "setting_label": row["setting_label"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_id": checkpoint_id,
            "generated_wav": str(row["generated_wav"]),
            "metrics_json": str(row["metrics_json"]),
            "output_dir": str(row["output_dir"]),
            "source_start_sec": float(coherence.get("source_start_sec", row["start_sec"])),
            "source_seconds": float(coherence.get("source_seconds", row["source_seconds"])),
            "boundary_mel_mse_mean": float(metrics.get("boundary_mel_mse_mean", 0.0)),
            "boundary_disc_db_mean": float(metrics.get("boundary_disc_db_mean", 0.0)),
            "style_intent": _setting_style_intent(coherence),
        }
        for key in [
            "t_start",
            "t_start_end",
            "reanchor_every",
            "reanchor_t_start",
            "guidance_scale",
            "style_strength",
            "prefix_blend",
            "source_prefix_blend",
            "source_mel_blend",
            "hf_source_blend",
            "hf_start_bin",
            "mel_time_smooth",
            "mel_freq_smooth",
        ]:
            record[key] = float(coherence.get(key, 0.0))

        for key, value in src_diag.items():
            record[f"src_{key}"] = float(value)
        for key, value in gen_diag.items():
            record[f"gen_{key}"] = float(value)
        for key, value in paired.items():
            record[key] = float(value)

        record["hf_ratio_delta"] = abs(record["gen_hf_ratio"] - record["src_hf_ratio"])
        record["vocal_ratio_delta"] = abs(record["gen_vocal_ratio"] - record["src_vocal_ratio"])
        record["dynamic_range_delta"] = abs(record["gen_dynamic_range_db"] - record["src_dynamic_range_db"])
        record["start_hf_ratio_delta"] = abs(record["gen_start_hf_ratio"] - record["src_start_hf_ratio"])
        record["start_peak_to_rms_delta"] = abs(record["gen_start_peak_to_rms"] - record["src_start_peak_to_rms"])
        record["start_onset_burst_delta"] = abs(record["gen_start_onset_burst"] - record["src_start_onset_burst"])
        record["source_key"] = f"{row['source_audio']}::{record['source_start_sec']:.3f}::{row['target_genre']}"
        records.append(record)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No long-form training examples found.")

    higher_better = ["chroma_cos", "onset_corr", "style_intent"]
    lower_better = [
        "boundary_mel_mse_mean",
        "boundary_disc_db_mean",
        "gen_hf_roughness",
        "gen_spectral_flatness",
        "gen_clip_frac",
        "hf_ratio_delta",
        "vocal_ratio_delta",
        "dynamic_range_delta",
        "gen_start_hf_roughness",
        "gen_start_spectral_flatness",
        "gen_start_clip_frac",
        "gen_start_peak_to_rms",
        "start_hf_ratio_delta",
        "start_onset_burst_delta",
    ]
    for col in higher_better:
        df[f"rank_{col}"] = df[col].rank(method="average", pct=True)
    for col in lower_better:
        df[f"rank_{col}"] = 1.0 - df[col].rank(method="average", pct=True)

    df["policy_score"] = (
        0.12 * df["rank_chroma_cos"]
        + 0.06 * df["rank_onset_corr"]
        + 0.09 * df["rank_style_intent"]
        + 0.06 * df["rank_boundary_mel_mse_mean"]
        + 0.05 * df["rank_boundary_disc_db_mean"]
        + 0.11 * df["rank_gen_hf_roughness"]
        + 0.09 * df["rank_gen_spectral_flatness"]
        + 0.07 * df["rank_gen_clip_frac"]
        + 0.05 * df["rank_hf_ratio_delta"]
        + 0.04 * df["rank_vocal_ratio_delta"]
        + 0.03 * df["rank_dynamic_range_delta"]
        + 0.08 * df["rank_gen_start_hf_roughness"]
        + 0.06 * df["rank_gen_start_spectral_flatness"]
        + 0.05 * df["rank_gen_start_clip_frac"]
        + 0.05 * df["rank_gen_start_peak_to_rms"]
        + 0.04 * df["rank_start_hf_ratio_delta"]
        + 0.05 * df["rank_start_onset_burst_delta"]
    )
    return df.sort_values(["source_key", "policy_score"], ascending=[True, False]).reset_index(drop=True)


def train_policy_model(df: pd.DataFrame, out_dir: Path) -> Dict[str, Any]:
    feature_cols = [
        "src_rms_mean",
        "src_dynamic_range_db",
        "src_zcr",
        "src_spectral_centroid",
        "src_spectral_flatness",
        "src_hf_ratio",
        "src_vocal_ratio",
        "src_hf_roughness",
        "src_onset_strength_mean",
        "src_onset_density",
        "src_start_spectral_flatness",
        "src_start_hf_ratio",
        "src_start_hf_roughness",
        "src_start_clip_frac",
        "src_start_peak_to_rms",
        "src_start_onset_burst",
        "t_start",
        "t_start_end",
        "reanchor_every",
        "reanchor_t_start",
        "guidance_scale",
        "style_strength",
        "style_intent",
        "prefix_blend",
        "source_prefix_blend",
        "source_mel_blend",
        "hf_source_blend",
        "hf_start_bin",
        "mel_time_smooth",
        "mel_freq_smooth",
    ]
    cat_cols = ["target_genre", "source_genre", "checkpoint_id", "setting_label"]

    model_df = pd.get_dummies(df[feature_cols + cat_cols], columns=cat_cols, dtype=float)
    y = df["policy_score"].astype(float)
    groups = df["source_key"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=328)
    train_idx, test_idx = next(splitter.split(model_df, y, groups=groups))

    X_train = model_df.iloc[train_idx]
    X_test = model_df.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=2,
        random_state=328,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "r2": float(r2_score(y_test, pred_test)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_groups_train": int(len(set(groups.iloc[train_idx]))),
        "n_groups_test": int(len(set(groups.iloc[test_idx]))),
    }

    train_group_df = df.iloc[train_idx].copy()
    combo_means = (
        train_group_df.groupby(["checkpoint_id", "setting_label"], as_index=False)["policy_score"]
        .mean()
        .sort_values("policy_score", ascending=False)
    )
    baseline_combo = combo_means.iloc[0].to_dict()
    style_combo = (
        train_group_df[train_group_df["style_intent"] >= train_group_df["style_intent"].quantile(0.70)]
        .groupby(["checkpoint_id", "setting_label"], as_index=False)["policy_score"]
        .mean()
        .sort_values("policy_score", ascending=False)
        .iloc[0]
        .to_dict()
    )

    test_eval = df.iloc[test_idx].copy()
    test_eval["pred_score"] = pred_test
    policy_rows = test_eval.sort_values("pred_score", ascending=False).groupby("source_key", as_index=False).head(1)
    oracle_rows = test_eval.sort_values("policy_score", ascending=False).groupby("source_key", as_index=False).head(1)
    baseline_rows = test_eval[
        (test_eval["checkpoint_id"] == baseline_combo["checkpoint_id"])
        & (test_eval["setting_label"] == baseline_combo["setting_label"])
    ]
    style_rows = test_eval[
        (test_eval["checkpoint_id"] == style_combo["checkpoint_id"])
        & (test_eval["setting_label"] == style_combo["setting_label"])
    ]

    ranking_summary = {
        "policy_mean_true_score": float(policy_rows["policy_score"].mean()),
        "oracle_mean_true_score": float(oracle_rows["policy_score"].mean()),
        "baseline_mean_true_score": float(baseline_rows["policy_score"].mean()) if not baseline_rows.empty else None,
        "style_baseline_mean_true_score": float(style_rows["policy_score"].mean()) if not style_rows.empty else None,
        "baseline_combo": baseline_combo,
        "style_combo": style_combo,
    }

    bundle = {
        "feature_columns": list(model_df.columns),
        "model": model,
        "feature_numeric": feature_cols,
        "feature_categorical": cat_cols,
        "baseline_combo": baseline_combo,
        "style_combo": style_combo,
    }
    with (out_dir / "policy_model.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    combo_means.to_csv(out_dir / "combo_means.csv", index=False)
    (out_dir / "policy_metrics.json").write_text(
        json.dumps({"fit_metrics": metrics, "ranking_summary": ranking_summary}, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return {"bundle": bundle, "fit_metrics": metrics, "ranking_summary": ranking_summary}


def _candidate_setting_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "setting_label",
        "t_start",
        "t_start_end",
        "reanchor_every",
        "reanchor_t_start",
        "guidance_scale",
        "style_strength",
        "style_intent",
        "prefix_blend",
        "source_prefix_blend",
        "source_mel_blend",
        "hf_source_blend",
        "hf_start_bin",
        "mel_time_smooth",
        "mel_freq_smooth",
    ]
    return df[cols].drop_duplicates("setting_label").reset_index(drop=True)


def _checkpoint_id_from_path(path: Path) -> str:
    return f"{path.parent.parent.name}::{path.name}"


def _build_candidate_rows(
    bundle: Dict[str, Any],
    training_df: pd.DataFrame,
    *,
    source_audio: Path,
    source_genre: str,
    target_genre: str,
    start_sec: float,
    source_seconds: float,
    checkpoint_paths: Sequence[Path],
) -> pd.DataFrame:
    y_src = _safe_load_audio(source_audio, offset=start_sec, duration=source_seconds)
    src_diag = _audio_diagnostics(y_src)
    settings_df = _candidate_setting_table(training_df)
    rows: List[Dict[str, Any]] = []
    for ckpt in checkpoint_paths:
        checkpoint_id = _checkpoint_id_from_path(ckpt)
        for _, setting in settings_df.iterrows():
            row = {
                "source_audio": str(source_audio),
                "source_genre": source_genre,
                "target_genre": target_genre,
                "source_start_sec": float(start_sec),
                "source_seconds": float(source_seconds),
                "checkpoint_path": str(ckpt),
                "checkpoint_id": checkpoint_id,
                "setting_label": setting["setting_label"],
            }
            for key, value in src_diag.items():
                row[f"src_{key}"] = float(value)
            for key in settings_df.columns:
                if key != "setting_label":
                    row[key] = float(setting[key])
            rows.append(row)
    candidate_df = pd.DataFrame(rows)
    model_input = pd.get_dummies(
        candidate_df[bundle["feature_numeric"] + bundle["feature_categorical"]],
        columns=bundle["feature_categorical"],
        dtype=float,
    )
    model_input = model_input.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    candidate_df["pred_score"] = bundle["model"].predict(model_input)
    return candidate_df.sort_values("pred_score", ascending=False).reset_index(drop=True)


def _pick_fresh_demo_songs(training_df: pd.DataFrame, cfg: PolicyConfig) -> List[Dict[str, Any]]:
    seen = set(training_df["source_audio"].astype(str))
    rows = ddb.discover_download_audio(cfg.downloads_dir)
    rows = [
        row
        for row in rows
        if (row["duration_seconds"] or 0.0) >= float(cfg.source_seconds) + 10.0
        and row["size_bytes"] >= 10_000_000
        and str(row["path"]) not in seen
        and all(ord(ch) < 128 for ch in str(row["path"]))
    ]
    rng = random.Random(cfg.seed + 99)
    selected = rng.sample(rows, k=min(int(cfg.n_demo_songs), len(rows)))
    jobs: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected):
        path = Path(row["path"])
        duration = float(row["duration_seconds"] or 0.0)
        max_start = max(0.0, duration - float(cfg.source_seconds) - 0.1)
        start_sec = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        source_genre = ddb.infer_source_genre(path)
        target_genre = "baroque_classical" if source_genre != "baroque_classical" else "lofi_hh_lfbb"
        jobs.append(
            {
                "job_idx": idx,
                "source_audio": path,
                "source_genre": source_genre,
                "target_genre": target_genre,
                "start_sec": round(float(start_sec), 3),
                "source_seconds": float(cfg.source_seconds),
                "duration_seconds": duration,
                "size_bytes": int(row["size_bytes"]),
            }
        )
    return jobs


def _checkpoint_path_for_id(checkpoint_id: str, candidates: Sequence[Path]) -> Path:
    for path in candidates:
        if _checkpoint_id_from_path(path) == checkpoint_id:
            return path
    raise KeyError(f"Checkpoint id not found in candidate list: {checkpoint_id}")


def _render_longform(
    *,
    out_dir: Path,
    checkpoint_path: Path,
    source_audio: Path,
    source_genre: str,
    target_genre: str,
    start_sec: float,
    source_seconds: float,
    setting_row: Dict[str, Any],
    seed: int,
) -> None:
    if (out_dir / "longform_coherent.wav").exists() and (out_dir / "coherence_metrics.json").exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"),
        "--checkpoint", str(checkpoint_path),
        "--lab1-checkpoint", str(REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"),
        "--source-audio", str(source_audio),
        "--source-genre", str(source_genre),
        "--target-genre", str(target_genre),
        "--source-start-sec", str(start_sec),
        "--source-seconds", str(source_seconds),
        "--out-dir", str(out_dir),
        "--chunk-seconds", "3.0",
        "--overlap-seconds", "0.5",
        "--n-frames", "256",
        "--ddim-steps", "50",
        "--assemble-domain", "mel",
        "--device", "auto",
        "--seed", str(seed),
        "--t-start", str(int(round(float(setting_row["t_start"])))),
        "--t-start-end", str(int(round(float(setting_row["t_start_end"])))),
        "--reanchor-every", str(int(round(float(setting_row["reanchor_every"])))),
        "--reanchor-t-start", str(int(round(float(setting_row["reanchor_t_start"])))),
        "--guidance-scale", str(float(setting_row["guidance_scale"])),
        "--style-strength", str(float(setting_row["style_strength"])),
        "--prefix-blend", str(float(setting_row["prefix_blend"])),
        "--source-prefix-blend", str(float(setting_row["source_prefix_blend"])),
        "--source-mel-blend", str(float(setting_row["source_mel_blend"])),
        "--hf-source-blend", str(float(setting_row["hf_source_blend"])),
        "--hf-start-bin", str(int(round(float(setting_row["hf_start_bin"])))),
        "--mel-time-smooth", str(int(round(float(setting_row["mel_time_smooth"])))),
        "--mel-freq-smooth", str(int(round(float(setting_row["mel_freq_smooth"])))),
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    log_path = out_dir / "policy_render.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Long-form render failed with exit code {code}: {' '.join(cmd)}")


def run_demo(cfg: PolicyConfig, training_df: pd.DataFrame, bundle: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    jobs = _pick_fresh_demo_songs(training_df, cfg)
    (out_dir / "demo_jobs.json").write_text(json.dumps(jobs, indent=2, default=_json_default), encoding="utf-8")

    baseline_safe_id, baseline_safe_setting = cfg.baseline_safe
    baseline_style_id, baseline_style_setting = cfg.baseline_style
    demo_rows: List[Dict[str, Any]] = []
    for job in jobs:
        candidates = _build_candidate_rows(
            bundle,
            training_df,
            source_audio=job["source_audio"],
            source_genre=job["source_genre"],
            target_genre=job["target_genre"],
            start_sec=float(job["start_sec"]),
            source_seconds=float(job["source_seconds"]),
            checkpoint_paths=cfg.candidate_checkpoint_paths,
        )
        top_policy = candidates.iloc[0].to_dict()
        presets = [("policy_top1", top_policy)]

        safe_rows = candidates[
            (candidates["checkpoint_id"].str.endswith(baseline_safe_id))
            & (candidates["setting_label"] == baseline_safe_setting)
        ]
        if not safe_rows.empty:
            presets.append(("baseline_safe", safe_rows.iloc[0].to_dict()))
        style_rows = candidates[
            (candidates["checkpoint_id"].str.endswith(baseline_style_id))
            & (candidates["setting_label"] == baseline_style_setting)
        ]
        if not style_rows.empty:
            presets.append(("baseline_style", style_rows.iloc[0].to_dict()))

        candidate_csv = out_dir / "candidate_rankings" / f"{_slug(job['source_audio'].stem)}.csv"
        candidate_csv.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidate_csv, index=False)

        for variant_label, row in presets:
            checkpoint_path = _checkpoint_path_for_id(str(row["checkpoint_id"]), cfg.candidate_checkpoint_paths)
            render_dir = (
                out_dir
                / "demo_generations"
                / variant_label
                / _slug(job["source_audio"].stem)[:40]
                / _slug(f"{Path(checkpoint_path).stem}_{row['setting_label']}_{job['target_genre']}")
            )
            _render_longform(
                out_dir=render_dir,
                checkpoint_path=checkpoint_path,
                source_audio=job["source_audio"],
                source_genre=job["source_genre"],
                target_genre=job["target_genre"],
                start_sec=float(job["start_sec"]),
                source_seconds=float(job["source_seconds"]),
                setting_row=row,
                seed=cfg.seed + int(job["job_idx"]),
            )
            demo_rows.append(
                {
                    "variant_label": variant_label,
                    "source_audio": str(job["source_audio"]),
                    "target_genre": job["target_genre"],
                    "checkpoint_id": row["checkpoint_id"],
                    "setting_label": row["setting_label"],
                    "pred_score": float(row["pred_score"]),
                    "output_dir": str(render_dir),
                    "generated_wav": str(render_dir / "longform_coherent.wav"),
                    "metrics_json": str(render_dir / "coherence_metrics.json"),
                }
            )

    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(out_dir / "demo_manifest.csv", index=False)
    return {
        "demo_jobs_path": out_dir / "demo_jobs.json",
        "demo_manifest_path": out_dir / "demo_manifest.csv",
        "n_demo_rows": int(len(demo_df)),
    }


def run_full_pipeline(cfg: PolicyConfig) -> Dict[str, Any]:
    cfg = cfg.materialize()
    run_dir = cfg.output_root / cfg.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    training_df = build_training_table(cfg)
    training_df.to_csv(run_dir / "training_table.csv", index=False)

    bundle_info = train_policy_model(training_df, run_dir)
    demo_info = run_demo(cfg, training_df, bundle_info["bundle"], run_dir)

    summary = {
        "tag": cfg.tag,
        "run_dir": run_dir,
        "training_table_path": run_dir / "training_table.csv",
        "policy_model_path": run_dir / "policy_model.pkl",
        "policy_metrics_path": run_dir / "policy_metrics.json",
        **bundle_info["fit_metrics"],
        "ranking_summary": bundle_info["ranking_summary"],
        **demo_info,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def main() -> None:
    cfg = PolicyConfig()
    summary = run_full_pipeline(cfg)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
