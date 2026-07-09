from __future__ import annotations

import argparse
import json
import shutil
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
    picked_songs,
    settings_panel,
)


# Derived from the completed fixed-song comparisons:
# - baroque_classical: push_guard very slightly beat push_a on mean score
# - hiphop_xtc: push_guard best
# - lofi_hh_lfbb: push_a best
# - cc0_other: baseline best
BEST_LABEL_BY_TARGET = {
    "baroque_classical": "style_pull_c",
    "hiphop_xtc": "style_pull_a",
    "lofi_hh_lfbb": "style_pull_c",
    "cc0_other": "style_pull_b",
}

BEST_BACKING_FIX_BY_TARGET = {
    "baroque_classical": {
        "backing_timing_mode": "anchorgrid_perc_to_source",
        "backing_source_blend": 0.06,
        "backing_percussive_blend": 0.08,
        "backing_post_mode": "genre_separate",
        "backing_post_strength": 0.36,
        "backing_dewarble_strength": 0.32,
        "accomp_mix_gain": 1.00,
    },
    "hiphop_xtc": {
        "backing_timing_mode": "anchorgrid_perc_to_source",
        "backing_source_blend": 0.00,
        "backing_percussive_blend": 0.00,
        "backing_post_mode": "genre_accent",
        "backing_post_strength": 0.78,
        "backing_dewarble_strength": 0.18,
        "accomp_mix_gain": 1.05,
    },
    "lofi_hh_lfbb": {
        "backing_timing_mode": "anchorgrid_perc_to_source",
        "backing_source_blend": 0.00,
        "backing_percussive_blend": 0.00,
        "backing_post_mode": "genre_texture",
        "backing_post_strength": 0.78,
        "backing_dewarble_strength": 0.18,
        "accomp_mix_gain": 1.04,
    },
    "cc0_other": {
        "backing_timing_mode": "anchorgrid_perc_to_source",
        "backing_source_blend": 0.00,
        "backing_percussive_blend": 0.00,
        "backing_post_mode": "genre_accent",
        "backing_post_strength": 0.66,
        "backing_dewarble_strength": 0.18,
        "accomp_mix_gain": 1.04,
    },
}

BEST_VOCAL_OFFSET_BY_TARGET = {
    "baroque_classical": {"vocal_delay_ms": 0.0},
    "hiphop_xtc": {"vocal_delay_ms": 0.0},
    "lofi_hh_lfbb": {"vocal_delay_ms": 0.0},
    "cc0_other": {"vocal_delay_ms": 0.0},
}

BEST_VOCAL_DEBLEED_BY_TARGET = {
    "baroque_classical": {"vocal_debleed_strength": 0.35, "vocal_debleed_floor": 0.18},
    "hiphop_xtc": {"vocal_debleed_strength": 0.35, "vocal_debleed_floor": 0.18},
    "lofi_hh_lfbb": {"vocal_debleed_strength": 0.45, "vocal_debleed_floor": 0.18},
    "cc0_other": {"vocal_debleed_strength": 0.0, "vocal_debleed_floor": 0.18},
}


def _picked_settings_map() -> Dict[str, Dict[str, Any]]:
    return {row["label"]: row for row in settings_panel()}


def _songs_from_downloads(root: Path, limit: int, song_filter: str = "") -> List[Dict[str, Any]]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if song_filter.strip():
        pat = song_filter.strip().lower()
        files = [p for p in files if pat in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No audio files found in {root}")
    picked = files[:limit]
    return [{"path": p, "source_genre": "cc0_other"} for p in picked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid vocal-preserve remaster runner with target-specific best settings.")
    parser.add_argument("--out-dir", type=str, default="", help="Existing or desired output directory.")
    parser.add_argument("--downloads-dir", type=str, default=str(Path.home() / "Downloads"), help="Folder containing source songs.")
    parser.add_argument("--limit", type=int, default=2, help="How many songs to render from downloads.")
    parser.add_argument("--song-filter", type=str, default="", help="Case-insensitive substring filter on source filename.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint override for the longform diffusion model.")
    parser.add_argument("--cache-dir", type=str, default="", help="Optional diffusion cache override.")
    parser.add_argument("--stem-cache-root", type=str, default="", help="Optional stem cache root override.")
    parser.add_argument("--use-picked-songs", action="store_true", help="Use the fixed picked compare songs instead of scanning downloads.")
    args = parser.parse_args()

    cfg = HybridPushConfig()
    if args.checkpoint.strip():
        cfg.checkpoint = Path(args.checkpoint)
    if args.cache_dir.strip():
        cfg.cache_dir = Path(args.cache_dir)
    if args.stem_cache_root.strip():
        cfg.stem_cache_root = Path(args.stem_cache_root)
    out_root = Path(args.out_dir) if args.out_dir.strip() else cfg.output_root / f"hybrid_auto_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    settings_map = _picked_settings_map()
    if bool(args.use_picked_songs):
        songs = picked_songs()
    else:
        downloads_dir = Path(args.downloads_dir)
        songs = _songs_from_downloads(downloads_dir, limit=max(1, int(args.limit)), song_filter=args.song_filter)

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "best_label_by_target.json").write_text(json.dumps(BEST_LABEL_BY_TARGET, indent=2), encoding="utf-8")
    (out_root / "best_backing_fix_by_target.json").write_text(json.dumps(BEST_BACKING_FIX_BY_TARGET, indent=2), encoding="utf-8")
    (out_root / "best_vocal_offset_by_target.json").write_text(json.dumps(BEST_VOCAL_OFFSET_BY_TARGET, indent=2), encoding="utf-8")
    (out_root / "best_vocal_debleed_by_target.json").write_text(json.dumps(BEST_VOCAL_DEBLEED_BY_TARGET, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(row["path"]), "source_genre": row["source_genre"]} for row in songs], indent=2), encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("job_idx,setting_label,source_audio,target_genre,output_dir,generated_wav,final_mix_wav\n")

    job_idx = 0
    for song in songs:
        stems = _resolve_stems(cfg, song)
        for target_genre in TARGET_GENRES:
            label = BEST_LABEL_BY_TARGET[target_genre]
            setting = dict(settings_map[label])
            setting.update(BEST_BACKING_FIX_BY_TARGET[target_genre])
            setting.update(BEST_VOCAL_OFFSET_BY_TARGET[target_genre])
            setting.update(BEST_VOCAL_DEBLEED_BY_TARGET[target_genre])
            setting["target_genre"] = target_genre
            base_name = f"{job_idx:03d}_{_slug(Path(song['path']).stem)}__to__{target_genre}"
            render_dir = out_root / "clips" / label / base_name
            render_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stems["source_clip"], render_dir / "source.wav")
            _run_longform(
                cfg,
                setting,
                stems["accompaniment"],
                song["source_genre"],
                target_genre,
                render_dir,
                seed=cfg.seed + job_idx,
            )
            final_mix = _make_mix(setting, stems, render_dir)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                f.write(
                    f"{job_idx},{label},{song['path']},{target_genre},{render_dir},"
                    f"{render_dir / 'longform_coherent.wav'},{final_mix}\n"
                )
            job_idx += 1

    summary = {
        "output_root": str(out_root),
        "n_songs": len(songs),
        "n_jobs": job_idx,
        "best_label_by_target": BEST_LABEL_BY_TARGET,
        "best_backing_fix_by_target": BEST_BACKING_FIX_BY_TARGET,
        "best_vocal_offset_by_target": BEST_VOCAL_OFFSET_BY_TARGET,
        "best_vocal_debleed_by_target": BEST_VOCAL_DEBLEED_BY_TARGET,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
