from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_hybrid_vocal_push_compare import (  # noqa: E402
    HybridPushConfig,
    _json_default,
    _make_mix,
    _resolve_stems,
    _run_longform,
    _shift_wave,
    _slug,
    picked_songs,
    settings_panel,
)


TARGETS = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb"]

BASE_LABEL_BY_TARGET = {
    "baroque_classical": "hybrid_push_guard",
    "hiphop_xtc": "hybrid_push_guard",
    "lofi_hh_lfbb": "hybrid_push_a",
}

CANDIDATES = [
    {"candidate_label": "baseline_none", "backing_timing_mode": "none", "backing_source_blend": 0.0, "vocal_delay_ms": 0.0},
    {"candidate_label": "warp_blend10_delay160", "backing_timing_mode": "warp_to_source", "backing_source_blend": 0.10, "vocal_delay_ms": 160.0},
    {"candidate_label": "warp_blend20_delay160", "backing_timing_mode": "warp_to_source", "backing_source_blend": 0.20, "vocal_delay_ms": 160.0},
    {"candidate_label": "dtw_blend20_delay160", "backing_timing_mode": "dtw_to_source", "backing_source_blend": 0.20, "vocal_delay_ms": 160.0},
    {"candidate_label": "phrasegrid_blend10_delay160", "backing_timing_mode": "phrasegrid_to_source", "backing_source_blend": 0.10, "vocal_delay_ms": 160.0},
    {"candidate_label": "phrasegrid_blend20_delay160", "backing_timing_mode": "phrasegrid_to_source", "backing_source_blend": 0.20, "vocal_delay_ms": 160.0},
    {"candidate_label": "phrasegrid_blend20_delay80", "backing_timing_mode": "phrasegrid_to_source", "backing_source_blend": 0.20, "vocal_delay_ms": 80.0},
    {"candidate_label": "phrasegrid_blend20_delay0", "backing_timing_mode": "phrasegrid_to_source", "backing_source_blend": 0.20, "vocal_delay_ms": 0.0},
    {"candidate_label": "anchorgrid_blend20_delay0", "backing_timing_mode": "anchorgrid_to_source", "backing_source_blend": 0.20, "backing_percussive_blend": 0.0, "vocal_delay_ms": 0.0},
    {"candidate_label": "anchorgrid_perc15_blend20_delay0", "backing_timing_mode": "anchorgrid_perc_to_source", "backing_source_blend": 0.20, "backing_percussive_blend": 0.15, "vocal_delay_ms": 0.0},
    {"candidate_label": "anchorgrid_perc25_blend20_delay0", "backing_timing_mode": "anchorgrid_perc_to_source", "backing_source_blend": 0.20, "backing_percussive_blend": 0.25, "vocal_delay_ms": 0.0},
]


def _load_tradeoff_module():
    path = THIS_DIR / "offline_tradeoff_optimize.py"
    spec = importlib.util.spec_from_file_location("hybrid_tradeoff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_tradeoff"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_mono(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _xcorr_profile(a: np.ndarray, b: np.ndarray, max_lag_frames: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    vals: List[float] = []
    lags = np.arange(-max_lag_frames, max_lag_frames + 1, dtype=np.int32)
    for lag in lags:
        if lag >= 0:
            x = a[: len(a) - lag]
            y = b[lag : lag + len(x)]
        else:
            shift = -lag
            x = a[shift:]
            y = b[: len(x)]
        if len(x) < 8 or len(y) < 8:
            vals.append(0.0)
            continue
        x = x - float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
        vals.append(float(np.dot(x, y) / denom))
    return lags.astype(np.float32), np.asarray(vals, dtype=np.float32)


def _onset_env(y: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).astype(np.float32)


def _phase_signature(v_env: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    delta = 0.2 * float(np.max(v_env) + 1e-6)
    peak_frames = librosa.util.peak_pick(v_env, pre_max=2, post_max=2, pre_avg=4, post_avg=4, delta=delta, wait=2)
    peak_frames = np.asarray(peak_frames, dtype=np.int64)
    if peak_frames.size == 0 or beat_frames.size < 2:
        return np.zeros(16, dtype=np.float32)
    residuals: List[float] = []
    for p in peak_frames.tolist():
        idx = int(np.searchsorted(beat_frames, p))
        prev_b = int(beat_frames[max(0, idx - 1)])
        next_b = int(beat_frames[min(len(beat_frames) - 1, idx)])
        if next_b <= prev_b:
            continue
        interval = max(1, next_b - prev_b)
        residuals.append(float((p - prev_b) / interval))
    if not residuals:
        return np.zeros(16, dtype=np.float32)
    hist, _ = np.histogram(np.asarray(residuals, dtype=np.float32), bins=16, range=(-0.5, 1.5), density=True)
    return hist.astype(np.float32)


def _source_relation(src_vocals: np.ndarray, src_accomp: np.ndarray, sr: int, hop: int = 512) -> Dict[str, Any]:
    v_env = _onset_env(src_vocals, sr, hop=hop)
    a_env = _onset_env(src_accomp, sr, hop=hop)
    n = min(len(v_env), len(a_env))
    v_env = v_env[:n]
    a_env = a_env[:n]
    lags, profile = _xcorr_profile(v_env, a_env)
    best_idx = int(np.argmax(profile))
    beat_frames = librosa.beat.beat_track(y=src_accomp, sr=sr, hop_length=hop, units="frames")[1]
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    return {
        "v_env": v_env,
        "a_env": a_env,
        "lags": lags,
        "profile": profile,
        "best_lag_frames": float(lags[best_idx]),
        "beat_frames": beat_frames,
        "hop": hop,
    }


def _local_backing_metrics(src_accomp: np.ndarray, backing: np.ndarray, sr: int, hop: int = 512, win_frames: int = 64, step_frames: int = 32) -> Dict[str, float]:
    src_env = _onset_env(src_accomp, sr, hop=hop)
    gen_env = _onset_env(backing, sr, hop=hop)
    m = min(len(src_env), len(gen_env))
    src_env = src_env[:m]
    gen_env = gen_env[:m]
    lags: List[float] = []
    cors: List[float] = []
    for st in range(0, max(0, m - win_frames), step_frames):
        a = src_env[st : st + win_frames] - float(np.mean(src_env[st : st + win_frames]))
        best = (0, -1e18)
        for lag in range(-16, 17):
            if lag >= 0:
                x = a[: win_frames - lag]
                b = gen_env[st + lag : st + lag + len(x)] - float(np.mean(gen_env[st + lag : st + lag + len(x)]))
            else:
                shift = -lag
                x = a[shift:win_frames]
                b = gen_env[st : st + len(x)] - float(np.mean(gen_env[st : st + len(x)]))
            if len(x) < 8:
                continue
            denom = float(np.linalg.norm(x) * np.linalg.norm(b)) + 1e-8
            score = float(np.dot(x, b) / denom)
            if score > best[1]:
                best = (lag, score)
        lags.append(float(best[0] * hop / sr))
        cors.append(float(best[1]))
    lag_arr = np.asarray(lags, dtype=np.float32) if lags else np.zeros(1, dtype=np.float32)
    corr_arr = np.asarray(cors, dtype=np.float32) if cors else np.zeros(1, dtype=np.float32)
    abs_lag = np.abs(lag_arr)
    start_idx = max(1, int(0.2 * len(lag_arr)))
    end_idx = max(start_idx + 1, int(0.8 * len(lag_arr)))
    x = np.arange(len(lag_arr), dtype=np.float32)
    if len(lag_arr) > 1:
        lag_trend = float(np.polyfit(x, lag_arr.astype(np.float64), 1)[0])
    else:
        lag_trend = 0.0
    return {
        "local_mean_abs_lag_sec": float(np.mean(abs_lag)),
        "local_std_lag_sec": float(np.std(lag_arr)),
        "local_max_abs_lag_sec": float(np.max(abs_lag)),
        "local_mean_corr": float(np.mean(corr_arr)),
        "mid_mean_abs_lag_sec": float(np.mean(np.abs(lag_arr[start_idx:end_idx]))),
        "lag_trend_sec_per_window": lag_trend,
    }


def _sync_score(source_relation: Dict[str, Any], vocals: np.ndarray, backing: np.ndarray, sr: int) -> Dict[str, float]:
    hop = int(source_relation["hop"])
    v_env = _onset_env(vocals, sr, hop=hop)
    a_env = _onset_env(backing, sr, hop=hop)
    n = min(len(v_env), len(a_env))
    v_env = v_env[:n]
    a_env = a_env[:n]
    lags, profile = _xcorr_profile(v_env, a_env)
    best_idx = int(np.argmax(profile))
    best_lag = float(lags[best_idx])
    src_profile = source_relation["profile"]
    m = min(len(src_profile), len(profile))
    profile_cos = 0.0
    if m > 0:
        x = src_profile[:m]
        y = profile[:m]
        denom = float(np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
        profile_cos = float(np.dot(x, y) / denom)
    cand_phase = _phase_signature(v_env, np.asarray(source_relation["beat_frames"], dtype=np.int64))
    src_phase = _phase_signature(source_relation["v_env"], np.asarray(source_relation["beat_frames"], dtype=np.int64))
    phase_l1 = float(np.mean(np.abs(cand_phase - src_phase)))
    lag_error = abs(best_lag - float(source_relation["best_lag_frames"]))
    peak_corr = float(profile[best_idx]) if profile.size else 0.0
    return {
        "profile_cos": profile_cos,
        "peak_corr": peak_corr,
        "lag_error_frames": lag_error,
        "phase_l1": phase_l1,
        "best_lag_frames": best_lag,
        "sync_profile_score": float(profile_cos + 0.5 * peak_corr - 0.08 * lag_error - 0.7 * phase_l1),
    }


def _classify_drift(local_metrics: Dict[str, float]) -> str:
    if local_metrics["local_mean_abs_lag_sec"] < 0.035 and local_metrics["local_std_lag_sec"] < 0.08:
        return "well_aligned"
    if local_metrics["local_std_lag_sec"] < 0.05 and local_metrics["local_mean_abs_lag_sec"] >= 0.035:
        return "constant_like_offset"
    if abs(local_metrics["lag_trend_sec_per_window"]) > 0.006:
        return "monotonic_drift"
    return "piecewise_phrase_drift"


def main() -> None:
    oto = _load_tradeoff_module()
    cfg = HybridPushConfig()
    out_root = Path.home() / "Desktop" / "dggr_hybrid_sync_optimize" / f"sync_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    mert_cfg = oto.OptimizeConfig()
    mert = oto.FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=5.0, device="auto", layer=-1)
    centroids = oto._build_mert_centroids(mert_cfg, out_root)
    settings_map = {row["label"]: row for row in settings_panel()}
    songs = picked_songs()

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "candidates.json").write_text(json.dumps(CANDIDATES, indent=2), encoding="utf-8")

    raw_root = out_root / "raw_generations"
    cand_root = out_root / "candidate_mixes"
    raw_root.mkdir(parents=True, exist_ok=True)
    cand_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, Any]] = []
    by_target: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TARGETS}

    for song in songs:
        stems = _resolve_stems(cfg, song)
        src_vocals, sr_v = _load_mono(stems["vocals"])
        src_accomp, sr_a = _load_mono(stems["accompaniment"])
        if sr_v != sr_a:
            raise RuntimeError("Stem sample rate mismatch")
        relation = _source_relation(src_vocals, src_accomp, sr_v)
        for target in TARGETS:
            label = BASE_LABEL_BY_TARGET[target]
            base_setting = dict(settings_map[label])
            raw_dir = raw_root / label / f"{_slug(Path(song['path']).stem)}__to__{target}"
            raw_dir.mkdir(parents=True, exist_ok=True)
            if not (raw_dir / "source.wav").exists():
                sf.write(str(raw_dir / "source.wav"), _load_mono(stems["source_clip"])[0], sr_v)
            _run_longform(cfg, base_setting, stems["accompaniment"], song["source_genre"], target, raw_dir, cfg.seed + len(manifest_rows))

            target_rows: List[Dict[str, Any]] = []
            for candidate in CANDIDATES:
                setting = dict(base_setting)
                setting.update(candidate)
                render_dir = cand_root / target / candidate["candidate_label"] / f"{_slug(Path(song['path']).stem)}"
                render_dir.mkdir(parents=True, exist_ok=True)
                for src_name, dst_name in [
                    ("source_clip", "source.wav"),
                    ("vocals", "vocals.wav"),
                    ("accompaniment", "source_accompaniment.wav"),
                ]:
                    src_path = stems[src_name] if src_name in stems else stems["source_clip"]
                    if not (render_dir / dst_name).exists():
                        (render_dir / dst_name).write_bytes(Path(src_path).read_bytes())
                if not (render_dir / "longform_coherent.wav").exists():
                    (render_dir / "longform_coherent.wav").write_bytes((raw_dir / "longform_coherent.wav").read_bytes())

                final_mix = _make_mix(setting, stems, render_dir)
                backing_path = render_dir / "backing_fixed.wav"
                backing, sr_b = _load_mono(backing_path)
                if sr_b != sr_v:
                    raise RuntimeError("Backing sample rate mismatch")
                local_metrics = _local_backing_metrics(src_accomp, backing, sr_v)
                shifted_vocals = _shift_wave(src_vocals, int(round(float(candidate["vocal_delay_ms"]) * sr_v / 1000.0)))
                sync_metrics = _sync_score(relation, shifted_vocals, backing, sr_v)
                style_metrics = oto._analyze_candidate(
                    source_wav=stems["accompaniment"],
                    generated_wav=backing_path,
                    target_genre=target,
                    mert=mert,
                    genre_centroids=centroids,
                    coherence_json=render_dir / "coherence_metrics.json",
                    meta={
                        "source_audio": str(song["path"]),
                        "target_genre": target,
                        "setting_label": label,
                        "candidate_label": candidate["candidate_label"],
                        "output_dir": str(render_dir),
                        "final_mix_wav": str(final_mix),
                    },
                )
                row = {
                    **style_metrics,
                    **local_metrics,
                    **sync_metrics,
                    "drift_type": _classify_drift(local_metrics),
                }
                row["overall_score"] = float(
                    1.60 * row["sync_profile_score"]
                    + 0.90 * row["local_mean_corr"]
                    - 2.20 * row["local_mean_abs_lag_sec"]
                    - 1.10 * row["local_std_lag_sec"]
                    - 0.45 * row["local_max_abs_lag_sec"]
                    + 0.10 * row["style_gain"]
                    + 0.05 * row["movement"]
                    - 0.15 * row["gen_start_hf_roughness"]
                )
                target_rows.append(row)
                manifest_rows.append(row)

            target_rows.sort(key=lambda r: r["overall_score"], reverse=True)
            by_target[target].append(target_rows[0])

    manifest_path = out_root / "manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    recommended: Dict[str, Dict[str, Any]] = {}
    group_rows: List[Dict[str, Any]] = []
    for target, rows in by_target.items():
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in [r for r in manifest_rows if r["target_genre"] == target]:
            grouped.setdefault(str(row["candidate_label"]), []).append(row)
        scored = []
        for cand_label, cand_rows in grouped.items():
            avg = {
                "candidate_label": cand_label,
                "target_genre": target,
                "n_rows": len(cand_rows),
                "mean_overall_score": float(np.mean([r["overall_score"] for r in cand_rows])),
                "mean_sync_profile_score": float(np.mean([r["sync_profile_score"] for r in cand_rows])),
                "mean_local_mean_abs_lag_sec": float(np.mean([r["local_mean_abs_lag_sec"] for r in cand_rows])),
                "mean_local_std_lag_sec": float(np.mean([r["local_std_lag_sec"] for r in cand_rows])),
                "mean_local_mean_corr": float(np.mean([r["local_mean_corr"] for r in cand_rows])),
                "mean_style_gain": float(np.mean([r["style_gain"] for r in cand_rows])),
                "mean_warble": float(np.mean([r["gen_start_hf_roughness"] for r in cand_rows])),
            }
            scored.append(avg)
            group_rows.append(avg)
        scored.sort(key=lambda r: r["mean_overall_score"], reverse=True)
        winner = scored[0]
        cand_cfg = next(row for row in CANDIDATES if row["candidate_label"] == winner["candidate_label"])
        recommended[target] = {
            "candidate_label": winner["candidate_label"],
            "backing_timing_mode": cand_cfg["backing_timing_mode"],
            "backing_source_blend": float(cand_cfg["backing_source_blend"]),
            "vocal_delay_ms": float(cand_cfg["vocal_delay_ms"]),
            "score_summary": winner,
        }

    (out_root / "group_summary.json").write_text(json.dumps(group_rows, indent=2), encoding="utf-8")
    (out_root / "recommended_map.json").write_text(json.dumps(recommended, indent=2), encoding="utf-8")
    diagnosis = {
        "root_cause": "piecewise_phrase_drift_in_generated_backing_against_source_vocal_grid",
        "insufficient_fix": "constant_delay_or_single_global_backing_warp",
        "winning_family_by_target": {k: v["candidate_label"] for k, v in recommended.items()},
    }
    (out_root / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    print(json.dumps({"out_root": str(out_root), "recommended": recommended}, indent=2))


if __name__ == "__main__":
    main()
