from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf

from run_hybrid_vocal_push_compare import (
    HybridPushConfig,
    _debleed_vocals_with_source_accomp,
    _json_default,
)


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _ensure_file_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copyfile(src, dst)


def _onset_corr(a: np.ndarray, b: np.ndarray, sr: int, hop: int = 512) -> float:
    env_a = librosa.onset.onset_strength(y=a, sr=sr, hop_length=hop)
    env_b = librosa.onset.onset_strength(y=b, sr=sr, hop_length=hop)
    m = min(len(env_a), len(env_b))
    if m < 8:
        return 0.0
    xa = env_a[:m] - float(np.mean(env_a[:m]))
    xb = env_b[:m] - float(np.mean(env_b[:m]))
    denom = float(np.linalg.norm(xa) * np.linalg.norm(xb)) + 1e-8
    return float(np.dot(xa, xb) / denom)


def _hf_roughness(y: np.ndarray, sr: int) -> float:
    y = y[: min(len(y), sr * 8)]
    if len(y) < 2048:
        return 0.0
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    hf = S[freqs >= 6000]
    if hf.size == 0 or hf.shape[1] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(hf, axis=1))))


def _mel_cos(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    ma = librosa.feature.melspectrogram(y=a, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    mb = librosa.feature.melspectrogram(y=b, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    m = min(ma.shape[1], mb.shape[1])
    if m < 4:
        return 0.0
    va = np.log1p(ma[:, :m]).reshape(-1)
    vb = np.log1p(mb[:, :m]).reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
    return float(np.dot(va, vb) / denom)


def _score_variant(
    original_vocals: np.ndarray,
    processed_vocals: np.ndarray,
    src_accomp: np.ndarray,
    backing_fixed: np.ndarray,
    hybrid: np.ndarray,
    longform: np.ndarray,
    sr: int,
) -> Dict[str, float]:
    bleed_corr = _onset_corr(processed_vocals, src_accomp, sr)
    vocal_preserve = _mel_cos(original_vocals, processed_vocals, sr)
    hybrid_backing_corr = _onset_corr(hybrid, backing_fixed, sr)
    hybrid_longform_corr = _onset_corr(hybrid, longform, sr)
    hf_gap = abs(_hf_roughness(hybrid, sr) - _hf_roughness(longform, sr))
    flat_h = float(librosa.feature.spectral_flatness(y=hybrid).mean())
    flat_l = float(librosa.feature.spectral_flatness(y=longform).mean())
    flat_gap = abs(flat_h - flat_l)
    score = float(
        0.60 * vocal_preserve
        - 0.55 * bleed_corr
        + 0.45 * hybrid_backing_corr
        + 0.20 * hybrid_longform_corr
        - 0.35 * hf_gap
        - 0.25 * flat_gap
    )
    return {
        "score": score,
        "bleed_corr": bleed_corr,
        "vocal_preserve": vocal_preserve,
        "hybrid_backing_corr": hybrid_backing_corr,
        "hybrid_longform_corr": hybrid_longform_corr,
        "hf_gap": hf_gap,
        "flat_gap": flat_gap,
    }


def _mix_with_existing_backing(
    original_vocals: np.ndarray,
    src_accomp: np.ndarray,
    backing_fixed: np.ndarray,
    sr: int,
    out_wav: Path,
    debleed_strength: float,
    debleed_floor: float,
    vocal_mix_gain: float = 0.95,
    accomp_mix_gain: float = 0.93,
) -> np.ndarray:
    vocals = original_vocals.astype(np.float32, copy=False)
    if debleed_strength > 0.0:
        vocals = _debleed_vocals_with_source_accomp(
            vocals,
            src_accomp.astype(np.float32, copy=False),
            sr,
            strength=float(debleed_strength),
            floor=float(debleed_floor),
        )
    n = min(len(vocals), len(backing_fixed))
    mix = float(vocal_mix_gain) * vocals[:n] + float(accomp_mix_gain) * backing_fixed[:n]
    peak = float(np.max(np.abs(mix))) + 1e-8
    if peak > 0:
        mix = (mix / peak * 0.95).astype(np.float32)
    sf.write(str(out_wav), mix, sr)
    return vocals[:n].astype(np.float32, copy=False)


def compare_cases() -> List[Dict[str, Any]]:
    home = Path.home()
    return [
        {
            "label": "sza_baroque",
            "stems_dir": home / "Desktop" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems" / "sza_f2f",
            "render_dir": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__baroque_classical",
            "target_genre": "baroque_classical",
        },
        {
            "label": "sza_lofi",
            "stems_dir": home / "Desktop" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems" / "sza_f2f",
            "render_dir": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__lofi_hh_lfbb",
            "target_genre": "lofi_hh_lfbb",
        },
        {
            "label": "bea_baroque",
            "stems_dir": home / "Desktop" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems" / "beabadoobee_fairy_song",
            "render_dir": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_120007" / "clips" / "anchor_style_pull_c" / "beabadoobee_fairy_song__to__baroque_classical",
            "target_genre": "baroque_classical",
        },
        {
            "label": "bea_hiphop",
            "stems_dir": home / "Desktop" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems" / "beabadoobee_fairy_song",
            "render_dir": home / "Desktop" / "dggr_hybrid_vocal_auto_best" / "hybrid_auto_best_20260331_112800_stylepush" / "bea" / "clips" / "style_pull_a" / "001_beabadoobee_fairy_song__to__hiphop_xtc",
            "target_genre": "hiphop_xtc",
        },
    ]


def main() -> None:
    out_root = Path.home() / "Desktop" / "dggr_hybrid_debleed_compare" / f"debleed_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = HybridPushConfig()
    strengths = [0.0, 0.15, 0.25, 0.35, 0.45]

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "strengths.json").write_text(json.dumps(strengths, indent=2), encoding="utf-8")

    all_rows: List[Dict[str, Any]] = []
    winners: Dict[str, Dict[str, Any]] = {}
    for case in compare_cases():
        stems = {
            "source_clip": case["stems_dir"] / "source_clip.wav",
            "vocals": case["stems_dir"] / "vocals.wav",
            "accompaniment": case["stems_dir"] / "accompaniment.wav",
        }
        base_render = case["render_dir"]
        for p in [stems["source_clip"], stems["vocals"], stems["accompaniment"], base_render / "longform_coherent.wav", base_render / "backing_fixed.wav", base_render / "coherence_metrics.json"]:
            if not p.exists():
                raise FileNotFoundError(f"Missing required input: {p}")

        original_vocals, sr = _load_mono(stems["vocals"])
        src_accomp, _ = _load_mono(stems["accompaniment"])
        longform, _ = _load_mono(base_render / "longform_coherent.wav")
        base_backing, _ = _load_mono(base_render / "backing_fixed.wav")

        rows: List[Dict[str, Any]] = []
        for strength in strengths:
            variant_name = f"debleed_{int(round(strength * 100)):02d}"
            variant_dir = out_root / case["label"] / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)
            for name in ["longform_coherent.wav", "backing_fixed.wav", "coherence_metrics.json", "source.wav"]:
                src = base_render / name
                if src.exists():
                    _ensure_file_link(src, variant_dir / name)
            processed_vocals = _mix_with_existing_backing(
                original_vocals=original_vocals,
                src_accomp=src_accomp,
                backing_fixed=base_backing,
                sr=sr,
                out_wav=variant_dir / "hybrid_longform_coherent.wav",
                debleed_strength=float(strength),
                debleed_floor=0.18,
                vocal_mix_gain=0.95,
                accomp_mix_gain=0.93,
            )
            (variant_dir / "hybrid_mix_meta.json").write_text(
                json.dumps(
                    {
                        "mix_method": "existing_backing_plus_debleeded_vocals",
                        "base_backing_fixed": str(base_render / "backing_fixed.wav"),
                        "vocal_debleed_strength": float(strength),
                        "vocal_debleed_floor": 0.18,
                        "vocal_mix_gain": 0.95,
                        "accomp_mix_gain": 0.93,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            hybrid, _ = _load_mono(variant_dir / "hybrid_longform_coherent.wav")
            metrics = _score_variant(original_vocals, processed_vocals, src_accomp, base_backing, hybrid, longform, sr)
            row = {
                "case_label": case["label"],
                "target_genre": case["target_genre"],
                "variant_name": variant_name,
                "vocal_debleed_strength": float(strength),
                "variant_dir": str(variant_dir),
                **metrics,
            }
            rows.append(row)
            all_rows.append(row)

        rows.sort(key=lambda r: r["score"], reverse=True)
        winners[case["label"]] = rows[0]

    manifest_path = out_root / "manifest.csv"
    if all_rows:
        import csv

        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summary = {
        "output_root": str(out_root),
        "winners": winners,
        "n_rows": len(all_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
