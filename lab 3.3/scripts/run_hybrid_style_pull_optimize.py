from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_hybrid_vocal_push_compare import (  # noqa: E402
    HybridPushConfig,
    TARGET_GENRES,
    _json_default,
    _make_mix,
    _resolve_stems,
    _run_longform,
    _slug,
)

import importlib.util  # noqa: E402


def _load_tradeoff_module():
    path = THIS_DIR / "offline_tradeoff_optimize.py"
    spec = importlib.util.spec_from_file_location("hybrid_tradeoff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_tradeoff"] = mod
    spec.loader.exec_module(mod)
    return mod


def picked_song(song_filter: str) -> Dict[str, Any]:
    base = Path.home() / "Downloads"
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in sorted(base.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    pat = song_filter.strip().lower()
    for p in files:
        if pat in p.name.lower():
            return {"path": p, "source_genre": "cc0_other"}
    raise FileNotFoundError(f"No Downloads song matched {song_filter!r}")


def style_pull_settings() -> List[Dict[str, Any]]:
    return [
        {
            "label": "hybrid_base",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 2.00,
            "style_strength": 0.74,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.38,
            "source_mel_blend": 0.04,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.12,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.90,
        },
        {
            "label": "hybrid_push_a",
            "t_start": 290,
            "t_start_end": 210,
            "reanchor_every": 3,
            "reanchor_t_start": 176,
            "guidance_scale": 2.12,
            "style_strength": 0.80,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.34,
            "source_mel_blend": 0.02,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.08,
            "hf_start_bin": 58,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.92,
        },
        {
            "label": "style_pull_a",
            "t_start": 300,
            "t_start_end": 214,
            "reanchor_every": 3,
            "reanchor_t_start": 178,
            "guidance_scale": 2.18,
            "style_strength": 0.86,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.30,
            "source_mel_blend": 0.01,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.07,
            "hf_start_bin": 58,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.94,
        },
        {
            "label": "style_pull_b",
            "t_start": 308,
            "t_start_end": 220,
            "reanchor_every": 3,
            "reanchor_t_start": 182,
            "guidance_scale": 2.24,
            "style_strength": 0.90,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.28,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.06,
            "hf_start_bin": 58,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.95,
        },
        {
            "label": "style_pull_guard",
            "t_start": 300,
            "t_start_end": 214,
            "reanchor_every": 2,
            "reanchor_t_start": 176,
            "guidance_scale": 2.16,
            "style_strength": 0.84,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.31,
            "source_mel_blend": 0.015,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.09,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.93,
        },
    ]


def _score_row(row: Dict[str, Any]) -> float:
    return float(
        row["style_gain"]
        + 0.22 * row["movement"]
        + 0.05 * row["chroma_cos"]
        + 0.05 * row["onset_corr"]
        - 0.18 * row["gen_start_hf_roughness"]
        - 0.08 * row["boundary_disc_db_mean"] / 10.0
        - 0.08 * row["boundary_mel_mse_mean"] * 10.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MERT-guided hybrid style-pull optimization.")
    parser.add_argument("--song-filter", type=str, default="beabadoobee - fairy song")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    cfg = HybridPushConfig()
    out_root = Path(args.out_dir) if args.out_dir.strip() else (Path.home() / "Desktop" / "dggr_hybrid_style_pull" / f"style_pull_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)

    oto = _load_tradeoff_module()
    mert = oto.FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=5.0, device="auto", layer=-1)
    centroids = oto._build_mert_centroids(oto.OptimizeConfig(), out_root)

    song = picked_song(args.song_filter)
    stems = _resolve_stems(cfg, song)
    settings = style_pull_settings()

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "song.json").write_text(json.dumps({"path": str(song["path"]), "source_genre": song["source_genre"]}, indent=2), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    job_idx = 0
    for setting in settings:
        for target_genre in TARGET_GENRES:
            tag = f"{job_idx:03d}_{_slug(Path(song['path']).stem)}__to__{target_genre}"
            out_dir = out_root / "clips" / setting["label"] / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            if not (out_dir / "source.wav").exists():
                (out_dir / "source.wav").write_bytes((stems["source_clip"]).read_bytes())
            _run_longform(cfg, setting, stems["accompaniment"], song["source_genre"], target_genre, out_dir, cfg.seed + job_idx)
            final_mix = _make_mix(setting, stems, out_dir)
            row = oto._analyze_candidate(
                source_wav=stems["accompaniment"],
                generated_wav=out_dir / "longform_coherent.wav",
                target_genre=target_genre,
                mert=mert,
                genre_centroids=centroids,
                coherence_json=out_dir / "coherence_metrics.json",
                meta={
                    "setting_label": setting["label"],
                    "target_genre": target_genre,
                    "source_audio": str(song["path"]),
                    "output_dir": str(out_dir),
                    "final_mix_wav": str(final_mix),
                },
            )
            row["style_pull_score"] = _score_row(row)
            rows.append(row)
            job_idx += 1

    rows_sorted = sorted(rows, key=lambda r: (r["target_genre"], -r["style_pull_score"]))
    by_genre: Dict[str, List[Dict[str, Any]]] = {}
    best_by_genre: Dict[str, Dict[str, Any]] = {}
    for genre in TARGET_GENRES:
        subset = sorted([r for r in rows if r["target_genre"] == genre], key=lambda r: r["style_pull_score"], reverse=True)
        by_genre[genre] = subset
        if subset:
            best_by_genre[genre] = {
                "winner": subset[0]["setting_label"],
                "score": subset[0]["style_pull_score"],
                "style_gain": subset[0]["style_gain"],
                "movement": subset[0]["movement"],
                "warble": subset[0]["gen_start_hf_roughness"],
                "output_dir": subset[0]["output_dir"],
                "final_mix_wav": subset[0]["final_mix_wav"],
            }

    (out_root / "ranking.json").write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
    (out_root / "best_by_genre.json").write_text(json.dumps(best_by_genre, indent=2), encoding="utf-8")
    print(json.dumps({"output_root": str(out_root), "best_by_genre": best_by_genre}, indent=2))


if __name__ == "__main__":
    main()
