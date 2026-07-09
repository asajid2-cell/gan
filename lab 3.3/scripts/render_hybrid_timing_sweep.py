from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import soundfile as sf

from run_hybrid_vocal_push_compare import _json_default, _make_mix


def main() -> None:
    parser = argparse.ArgumentParser(description="Render hybrid vocal timing variants from an existing longform accompaniment render.")
    parser.add_argument("--stem-dir", required=True, help="Directory containing source_clip.wav, vocals.wav, accompaniment.wav")
    parser.add_argument("--rendered-dir", required=True, help="Directory containing longform_coherent.wav")
    parser.add_argument("--out-dir", required=True, help="Directory to store timing variants")
    args = parser.parse_args()

    stem_dir = Path(args.stem_dir)
    rendered_dir = Path(args.rendered_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = {
        "source_clip": stem_dir / "source_clip.wav",
        "vocals": stem_dir / "vocals.wav",
        "accompaniment": stem_dir / "accompaniment.wav",
    }
    for path in stems.values():
        if not path.exists():
            raise FileNotFoundError(path)
    if not (rendered_dir / "longform_coherent.wav").exists():
        raise FileNotFoundError(rendered_dir / "longform_coherent.wav")

    variants: List[Dict[str, Any]] = [
        {"label": "timing_none", "vocal_mix_gain": 0.95, "accomp_mix_gain": 0.91, "vocal_timing_mode": "none", "vocal_delay_ms": 0.0},
        {"label": "timing_delay_80ms", "vocal_mix_gain": 0.95, "accomp_mix_gain": 0.91, "vocal_timing_mode": "none", "vocal_delay_ms": 80.0},
        {"label": "timing_delay_120ms", "vocal_mix_gain": 0.95, "accomp_mix_gain": 0.91, "vocal_timing_mode": "none", "vocal_delay_ms": 120.0},
        {"label": "timing_delay_160ms", "vocal_mix_gain": 0.95, "accomp_mix_gain": 0.91, "vocal_timing_mode": "none", "vocal_delay_ms": 160.0},
        {"label": "timing_beatwarp", "vocal_mix_gain": 0.95, "accomp_mix_gain": 0.91, "vocal_timing_mode": "beatwarp", "vocal_delay_ms": 0.0},
    ]

    manifest = []
    for variant in variants:
        variant_dir = out_dir / variant["label"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(variant_dir / "source.wav"), *sf.read(str(stems["source_clip"]), dtype="float32"))
        sf.write(str(variant_dir / "longform_coherent.wav"), *sf.read(str(rendered_dir / "longform_coherent.wav"), dtype="float32"))
        final_mix = _make_mix(variant, stems, variant_dir)
        manifest.append(
            {
                "label": variant["label"],
                "variant_dir": str(variant_dir),
                "output_wav": str(final_mix),
            }
        )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "n_variants": len(variants)}, indent=2))


if __name__ == "__main__":
    main()
