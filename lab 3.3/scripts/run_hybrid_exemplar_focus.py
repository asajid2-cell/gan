from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from offline_tradeoff_optimize import FrozenMERT, OptimizeConfig, _analyze_candidate, _build_mert_centroids, _mert_feat_for_audio
from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _json_default, _make_mix, _resolve_stems, _run_longform, picked_songs, settings_panel


FOCUS_LABELS = [
    "style_pull_d_sep",
    "style_pull_c_exfar",
    "style_pull_d_exfar",
    "style_pull_hybrid_exfar",
]


def _settings_map() -> Dict[str, Dict[str, Any]]:
    return {row["label"]: row for row in settings_panel()}


def _picked_songs() -> List[Dict[str, Any]]:
    songs = picked_songs()
    return songs[:2]


def main() -> None:
    cfg = HybridPushConfig(source_seconds=30.0)
    out_root = Path.home() / "Desktop" / "dggr_hybrid_exemplar_focus" / f"focus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    settings_map = _settings_map()
    settings = [dict(settings_map[label]) for label in FOCUS_LABELS]
    songs = _picked_songs()
    mert = FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=5.0, device="auto", layer=-1)
    centroids = _build_mert_centroids(OptimizeConfig(), out_root)

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(row["path"]), "source_genre": row["source_genre"]} for row in songs], indent=2), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "song",
            "setting_label",
            "target_genre",
            "output_dir",
            "backing_fixed_wav",
            "final_mix_wav",
            "style_target_cos_gen",
            "style_gain",
            "src_gen_cos",
            "movement",
            "boundary_disc_db_mean",
            "gen_start_hf_roughness",
        ])

    job_idx = 0
    for song in songs:
        stems = _resolve_stems(cfg, song)
        for setting in settings:
            for target in TARGET_GENRES:
                run_setting = dict(setting)
                run_setting["target_genre"] = target
                render_dir = out_root / "clips" / setting["label"] / f"{Path(song['path']).stem}__to__{target}"
                render_dir.mkdir(parents=True, exist_ok=True)
                _run_longform(cfg, run_setting, stems["accompaniment"], song["source_genre"], target, render_dir, cfg.seed + job_idx)
                final_mix = _make_mix(run_setting, stems, render_dir)
                backing = render_dir / "backing_fixed.wav"
                row = _analyze_candidate(
                    source_wav=stems["accompaniment"],
                    generated_wav=backing,
                    target_genre=target,
                    mert=mert,
                    genre_centroids=centroids,
                    coherence_json=render_dir / "coherence_metrics.json",
                    meta={
                        "song": str(song["path"]),
                        "setting_label": setting["label"],
                        "target_genre": target,
                        "output_dir": str(render_dir),
                        "final_mix_wav": str(final_mix),
                    },
                )
                feat = _mert_feat_for_audio(mert, backing)
                other_cos = [
                    float(np.dot(feat, centroids[g]) / (np.linalg.norm(feat) * np.linalg.norm(centroids[g]) + 1e-8))
                    for g in TARGET_GENRES
                    if g != target and g in centroids
                ]
                row["target_margin"] = float(row["style_target_cos_gen"] - max(other_cos)) if other_cos else float(row["style_target_cos_gen"])
                row["overall_score"] = float(
                    1.70 * row["target_margin"]
                    + 0.80 * row["style_gain"]
                    + 0.60 * row["movement"]
                    - 0.34 * row["gen_start_hf_roughness"]
                    - 0.12 * row["boundary_disc_db_mean"]
                )
                rows.append(row)
                with manifest_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        row["song"],
                        row["setting_label"],
                        row["target_genre"],
                        row["output_dir"],
                        str(backing),
                        str(final_mix),
                        row["style_target_cos_gen"],
                        row["style_gain"],
                        row["src_gen_cos"],
                        row["movement"],
                        row.get("boundary_disc_db_mean", 0.0),
                        row["gen_start_hf_roughness"],
                    ])
                job_idx += 1

    summary_rows: List[Dict[str, Any]] = []
    winner_map: Dict[str, Dict[str, Any]] = {}
    for target in TARGET_GENRES:
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if str(row["target_genre"]) == target:
                by_label.setdefault(str(row["setting_label"]), []).append(row)
        scored: List[Dict[str, Any]] = []
        for label, group in by_label.items():
            scored.append(
                {
                    "target_genre": target,
                    "setting_label": label,
                    "n_rows": len(group),
                    "mean_overall_score": float(np.mean([float(r["overall_score"]) for r in group])),
                    "mean_target_margin": float(np.mean([float(r["target_margin"]) for r in group])),
                    "mean_style_gain": float(np.mean([float(r["style_gain"]) for r in group])),
                    "mean_movement": float(np.mean([float(r["movement"]) for r in group])),
                    "mean_style_target_cos_gen": float(np.mean([float(r["style_target_cos_gen"]) for r in group])),
                    "mean_warble": float(np.mean([float(r["gen_start_hf_roughness"]) for r in group])),
                    "mean_boundary_disc_db_mean": float(np.mean([float(r.get("boundary_disc_db_mean", 0.0)) for r in group])),
                }
            )
        scored.sort(key=lambda x: x["mean_overall_score"], reverse=True)
        summary_rows.extend(scored)
        if scored:
            winner_map[target] = scored[0]

    (out_root / "summary_rows.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (out_root / "winner_map.json").write_text(json.dumps(winner_map, indent=2), encoding="utf-8")
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "output_root": str(out_root),
                "n_jobs": len(rows),
                "songs": [str(row["path"]) for row in songs],
                "labels": FOCUS_LABELS,
                "winner_map": winner_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_root": str(out_root), "n_jobs": len(rows), "winner_map": winner_map}, indent=2))


if __name__ == "__main__":
    main()
