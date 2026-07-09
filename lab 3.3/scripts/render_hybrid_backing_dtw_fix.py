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


def _load_audio(path: Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    return _mono(y), int(sr)


def _write(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y.astype(np.float32), sr)


def _dtw_warp_generated_to_source(src: np.ndarray, gen: np.ndarray, sr: int, hop: int = 512) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = min(len(src), len(gen))
    src = src[:n]
    gen = gen[:n]

    src_chroma = librosa.feature.chroma_cqt(y=src, sr=sr, hop_length=hop)
    gen_chroma = librosa.feature.chroma_cqt(y=gen, sr=sr, hop_length=hop)
    src_on = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)[None, :]
    gen_on = librosa.onset.onset_strength(y=gen, sr=sr, hop_length=hop)[None, :]

    src_feat = np.vstack([src_chroma * 0.7, src_on * 0.3]).astype(np.float32)
    gen_feat = np.vstack([gen_chroma * 0.7, gen_on * 0.3]).astype(np.float32)
    _, wp = librosa.sequence.dtw(X=src_feat, Y=gen_feat, metric="cosine")
    wp = np.asarray(wp[::-1], dtype=np.int64)

    src_frames = wp[:, 0].astype(np.float32)
    gen_frames = wp[:, 1].astype(np.float32)
    keep = np.concatenate(([True], np.diff(src_frames) > 0))
    src_frames = src_frames[keep]
    gen_frames = gen_frames[keep]
    if len(src_frames) < 2:
        return gen.astype(np.float32), {"warp_method": "none", "reason": "dtw_path_too_short"}

    target_samples = np.arange(n, dtype=np.float32)
    src_samples = src_frames * float(hop)
    gen_samples = gen_frames * float(hop)
    if src_samples[0] > 0:
        src_samples = np.concatenate(([0.0], src_samples))
        gen_samples = np.concatenate(([0.0], gen_samples))
    if src_samples[-1] < n - 1:
        src_samples = np.concatenate((src_samples, [float(n - 1)]))
        gen_samples = np.concatenate((gen_samples, [float(n - 1)]))
    mapped_gen_positions = np.interp(target_samples, src_samples, gen_samples)
    mapped_gen_positions = np.clip(mapped_gen_positions, 0.0, float(len(gen) - 1))
    warped = np.interp(mapped_gen_positions, np.arange(len(gen), dtype=np.float32), gen.astype(np.float32)).astype(np.float32)
    return warped, {
        "warp_method": "dtw_interp",
        "hop": hop,
        "path_points": int(len(src_frames)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render DTW-fixed backing variants for an existing hybrid render.")
    parser.add_argument("--stem-dir", required=True)
    parser.add_argument("--rendered-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    stem_dir = Path(args.stem_dir)
    rendered_dir = Path(args.rendered_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source, sr0 = _load_audio(stem_dir / "source_clip.wav")
    vocals, sr1 = _load_audio(stem_dir / "vocals.wav")
    src_accomp, sr2 = _load_audio(stem_dir / "accompaniment.wav")
    gen_accomp, sr3 = _load_audio(rendered_dir / "longform_coherent.wav")
    if not (sr0 == sr1 == sr2 == sr3):
        raise RuntimeError("Sample rate mismatch")

    warped, warp_meta = _dtw_warp_generated_to_source(src_accomp, gen_accomp, sr3)
    variants: List[Dict[str, Any]] = [
        {"label": "dtw_only", "source_blend": 0.00},
        {"label": "dtw_blend10", "source_blend": 0.10},
        {"label": "dtw_blend20", "source_blend": 0.20},
        {"label": "baseline_none", "source_blend": -1.0},
    ]

    manifest = []
    for variant in variants:
        d = out_dir / variant["label"]
        d.mkdir(parents=True, exist_ok=True)
        if variant["source_blend"] < 0.0:
            backing = gen_accomp.astype(np.float32, copy=True)
        else:
            n = min(len(warped), len(src_accomp))
            backing = ((1.0 - variant["source_blend"]) * warped[:n] + float(variant["source_blend"]) * src_accomp[:n]).astype(np.float32)
        m = min(len(vocals), len(backing))
        mix = 0.95 * vocals[:m] + 0.91 * backing[:m]
        peak = float(np.max(np.abs(mix))) + 1e-8
        mix = (mix / peak * 0.95).astype(np.float32)
        _write(d / "source.wav", source, sr3)
        _write(d / "backing_fixed.wav", backing, sr3)
        _write(d / "hybrid_longform_coherent.wav", mix, sr3)
        meta: Dict[str, Any] = {
            "label": variant["label"],
            "source_blend": float(variant["source_blend"]),
            **warp_meta,
        }
        (d / "variant_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        manifest.append({"label": variant["label"], "variant_dir": str(d), "output_wav": str(d / "hybrid_longform_coherent.wav")})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_variants": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
