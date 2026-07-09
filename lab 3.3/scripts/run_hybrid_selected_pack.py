from __future__ import annotations

import csv
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

from run_hybrid_vocal_auto_best import (  # noqa: E402
    BEST_BACKING_FIX_BY_TARGET,
    BEST_LABEL_BY_TARGET,
    BEST_VOCAL_DEBLEED_BY_TARGET,
    BEST_VOCAL_OFFSET_BY_TARGET,
    _json_default,
)
from run_hybrid_vocal_push_compare import (  # noqa: E402
    HybridPushConfig,
    TARGET_GENRES,
    _make_mix,
    _resolve_stems,
    _run_longform,
    _slug,
    picked_songs,
    settings_panel,
)


def _settings_map() -> Dict[str, Dict[str, Any]]:
    return {row["label"]: row for row in settings_panel()}


def main() -> None:
    cfg = HybridPushConfig(source_seconds=30.0)
    out_root = Path.home() / "Desktop" / "dggr_hybrid_selected_pack" / f"selected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    settings_map = _settings_map()
    songs = picked_songs()

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(row["path"]), "source_genre": row["source_genre"]} for row in songs], indent=2), encoding="utf-8")
    (out_root / "best_label_by_target.json").write_text(json.dumps(BEST_LABEL_BY_TARGET, indent=2), encoding="utf-8")
    (out_root / "best_backing_fix_by_target.json").write_text(json.dumps(BEST_BACKING_FIX_BY_TARGET, indent=2), encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job_idx", "song", "setting_label", "target_genre", "output_dir", "generated_wav", "final_mix_wav"])

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
            render_dir = out_root / "clips" / label / f"{job_idx:03d}_{_slug(Path(song['path']).stem)}__to__{target_genre}"
            render_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stems["source_clip"], render_dir / "source.wav")
            _run_longform(cfg, setting, stems["accompaniment"], song["source_genre"], target_genre, render_dir, cfg.seed + job_idx)
            final_mix = _make_mix(setting, stems, render_dir)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    job_idx,
                    str(song["path"]),
                    label,
                    target_genre,
                    str(render_dir),
                    str(render_dir / "longform_coherent.wav"),
                    str(final_mix),
                ])
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
