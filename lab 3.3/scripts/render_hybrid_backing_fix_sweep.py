from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf

from run_hybrid_vocal_push_compare import _beat_warp_wave, _json_default


def _mono(y: np.ndarray) -> np.ndarray:
    if y.ndim > 1:
        return y.mean(axis=1).astype(np.float32)
    return y.astype(np.float32)


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    sf.write(str(path), y.astype(np.float32), sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render backing-fix variants from an existing hybrid render.")
    parser.add_argument("--stem-dir", required=True, help="Directory containing source_clip.wav, vocals.wav, accompaniment.wav")
    parser.add_argument("--rendered-dir", required=True, help="Directory containing longform_coherent.wav")
    parser.add_argument("--out-dir", required=True, help="Directory to store backing-fix variants")
    args = parser.parse_args()

    stem_dir = Path(args.stem_dir)
    rendered_dir = Path(args.rendered_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_clip, sr0 = sf.read(str(stem_dir / "source_clip.wav"), dtype="float32")
    vocals, sr1 = sf.read(str(stem_dir / "vocals.wav"), dtype="float32")
    src_accomp, sr2 = sf.read(str(stem_dir / "accompaniment.wav"), dtype="float32")
    gen_accomp, sr3 = sf.read(str(rendered_dir / "longform_coherent.wav"), dtype="float32")
    if not (sr0 == sr1 == sr2 == sr3):
        raise RuntimeError("Sample rate mismatch")

    src_clip = _mono(src_clip)
    vocals = _mono(vocals)
    src_accomp = _mono(src_accomp)
    gen_accomp = _mono(gen_accomp)

    warped_accomp, warp_meta = _beat_warp_wave(gen_accomp, gen_accomp, src_accomp, sr3)
    variants: List[Dict[str, Any]] = [
        {"label": "backing_none", "use_warp": False, "source_blend": 0.00},
        {"label": "backing_warp", "use_warp": True, "source_blend": 0.00},
        {"label": "backing_warp_blend10", "use_warp": True, "source_blend": 0.10},
        {"label": "backing_warp_blend20", "use_warp": True, "source_blend": 0.20},
        {"label": "backing_blend10", "use_warp": False, "source_blend": 0.10},
    ]

    manifest = []
    for variant in variants:
        d = out_dir / variant["label"]
        d.mkdir(parents=True, exist_ok=True)
        _write_wav(d / "source.wav", src_clip, sr3)
        base_accomp = warped_accomp if variant["use_warp"] else gen_accomp
        n = min(len(base_accomp), len(src_accomp))
        fixed_accomp = (1.0 - float(variant["source_blend"])) * base_accomp[:n] + float(variant["source_blend"]) * src_accomp[:n]
        m = min(len(vocals), len(fixed_accomp))
        mix = 0.95 * vocals[:m] + 0.91 * fixed_accomp[:m]
        peak = float(np.max(np.abs(mix))) + 1e-8
        mix = (mix / peak * 0.95).astype(np.float32)
        _write_wav(d / "backing_fixed.wav", fixed_accomp, sr3)
        _write_wav(d / "hybrid_longform_coherent.wav", mix, sr3)
        meta = {
            "label": variant["label"],
            "use_warp": bool(variant["use_warp"]),
            "source_blend": float(variant["source_blend"]),
            "warp_meta": warp_meta,
        }
        (d / "variant_meta.json").write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        manifest.append({"label": variant["label"], "variant_dir": str(d), "output_wav": str(d / "hybrid_longform_coherent.wav")})

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
