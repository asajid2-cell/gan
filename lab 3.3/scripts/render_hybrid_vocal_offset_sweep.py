from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf


def _mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        return y.mean(axis=1).astype(np.float32)
    return y.astype(np.float32)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render vocal offset variants on top of an existing accompaniment.")
    parser.add_argument("--stem-dir", required=True)
    parser.add_argument("--accompaniment-wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--delays-ms", default="-160,-120,-80,-40,0,40,80,120,160")
    parser.add_argument("--vocal-mix-gain", type=float, default=0.95)
    parser.add_argument("--accomp-mix-gain", type=float, default=0.91)
    args = parser.parse_args()

    stem_dir = Path(args.stem_dir)
    accompaniment_wav = Path(args.accompaniment_wav)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocals, sr_v = sf.read(str(stem_dir / "vocals.wav"), dtype="float32")
    source, sr_s = sf.read(str(stem_dir / "source_clip.wav"), dtype="float32")
    accomp, sr_a = sf.read(str(accompaniment_wav), dtype="float32")
    if not (sr_v == sr_s == sr_a):
        raise RuntimeError("Sample rate mismatch")

    vocals = _mono(vocals)
    source = _mono(source)
    accomp = _mono(accomp)
    delays: List[float] = [float(part.strip()) for part in args.delays_ms.split(",") if part.strip()]

    manifest = []
    for delay_ms in delays:
        delay_samples = int(round(delay_ms * sr_a / 1000.0))
        variant_dir = out_dir / f"delay_{delay_ms:+.0f}ms".replace("+", "p").replace("-", "m")
        variant_dir.mkdir(parents=True, exist_ok=True)
        shifted_vocals = _shift_wave(vocals, delay_samples)
        n = min(len(shifted_vocals), len(accomp))
        mix = float(args.vocal_mix_gain) * shifted_vocals[:n] + float(args.accomp_mix_gain) * accomp[:n]
        peak = float(np.max(np.abs(mix))) + 1e-8
        mix = (mix / peak * 0.95).astype(np.float32)
        sf.write(str(variant_dir / "source.wav"), source, sr_a)
        sf.write(str(variant_dir / "backing_fixed.wav"), accomp, sr_a)
        sf.write(str(variant_dir / "hybrid_longform_coherent.wav"), mix, sr_a)
        (variant_dir / "variant_meta.json").write_text(
            json.dumps(
                {
                    "delay_ms": delay_ms,
                    "delay_samples": delay_samples,
                    "vocal_mix_gain": float(args.vocal_mix_gain),
                    "accomp_mix_gain": float(args.accomp_mix_gain),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "label": variant_dir.name,
                "delay_ms": delay_ms,
                "output_wav": str(variant_dir / "hybrid_longform_coherent.wav"),
            }
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_variants": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
