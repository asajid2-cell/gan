from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf


def _mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        return y.mean(axis=1).astype(np.float32)
    return y.astype(np.float32)


def _load_wav(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    return _mono(y), int(sr)


def _shift_wave(y: np.ndarray, lag_samples: int) -> np.ndarray:
    out = np.zeros_like(y, dtype=np.float32)
    if lag_samples == 0:
        return y.astype(np.float32, copy=True)
    if lag_samples > 0:
        n = max(0, len(y) - lag_samples)
        if n > 0:
            out[lag_samples : lag_samples + n] = y[:n]
    else:
        shift = -lag_samples
        n = max(0, len(y) - shift)
        if n > 0:
            out[:n] = y[shift : shift + n]
    return out


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
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return env.astype(np.float32)


def _source_relation(src_vocals: np.ndarray, src_accomp: np.ndarray, sr: int, hop: int = 512) -> Dict[str, Any]:
    v_env = _onset_env(src_vocals, sr, hop=hop)
    a_env = _onset_env(src_accomp, sr, hop=hop)
    n = min(len(v_env), len(a_env))
    v_env = v_env[:n]
    a_env = a_env[:n]
    lags, profile = _xcorr_profile(v_env, a_env)
    best_idx = int(np.argmax(profile))
    beat_frames = librosa.beat.beat_track(y=src_accomp, sr=sr, hop_length=hop, units="frames")[1]
    if isinstance(beat_frames, tuple):
        beat_frames = beat_frames[1]
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


def _phase_signature(v_env: np.ndarray, beat_frames: np.ndarray) -> np.ndarray:
    peak_frames = librosa.util.peak_pick(v_env, pre_max=2, post_max=2, pre_avg=4, post_avg=4, delta=0.2 * float(np.max(v_env) + 1e-6), wait=2)
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
        if abs(p - prev_b) <= abs(next_b - p):
            base = prev_b
            nxt = next_b
        else:
            base = prev_b
            nxt = next_b
        interval = max(1, nxt - base)
        residuals.append(float((p - base) / interval))
    if not residuals:
        return np.zeros(16, dtype=np.float32)
    hist, _ = np.histogram(np.asarray(residuals, dtype=np.float32), bins=16, range=(-0.5, 1.5), density=True)
    return hist.astype(np.float32)


def _score_variant(source_relation: Dict[str, Any], vocals: np.ndarray, backing: np.ndarray, sr: int) -> Dict[str, float]:
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
    score = float(profile_cos + 0.5 * peak_corr - 0.08 * lag_error - 0.7 * phase_l1)
    return {
        "profile_cos": profile_cos,
        "peak_corr": peak_corr,
        "lag_error_frames": lag_error,
        "phase_l1": phase_l1,
        "best_lag_frames": best_lag,
        "sync_score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score vocal offset sweep variants against source vocal/backing timing relationship.")
    parser.add_argument("--stem-dir", required=True)
    parser.add_argument("--sweep-dir", required=True)
    args = parser.parse_args()

    stem_dir = Path(args.stem_dir)
    sweep_dir = Path(args.sweep_dir)
    src_vocals, sr_v = _load_wav(stem_dir / "vocals.wav")
    src_accomp, sr_a = _load_wav(stem_dir / "accompaniment.wav")
    if sr_v != sr_a:
        raise RuntimeError("Sample rate mismatch")
    relation = _source_relation(src_vocals, src_accomp, sr_v)

    rows: List[Dict[str, Any]] = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        backing, sr_b = _load_wav(d / "backing_fixed.wav")
        if sr_b != sr_v:
            raise RuntimeError("Sample rate mismatch")
        meta_path = d / "variant_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"label": d.name}
        delay_samples = int(meta.get("delay_samples", 0))
        vocals = _shift_wave(src_vocals, delay_samples)
        row = {"label": d.name, **meta}
        row.update(_score_variant(relation, vocals, backing, sr_v))
        rows.append(row)

    rows.sort(key=lambda r: r["sync_score"], reverse=True)
    out_path = sweep_dir / "sync_scores.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "winner": rows[0] if rows else None,
        "rows": rows,
    }
    (sweep_dir / "sync_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
