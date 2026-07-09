from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf


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
)


ACTIVE_TARGETS = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]


def _load_tradeoff_module():
    path = THIS_DIR / "offline_tradeoff_optimize.py"
    spec = importlib.util.spec_from_file_location("hybrid_tradeoff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hybrid_tradeoff"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def candidate_settings() -> List[Dict[str, Any]]:
    common = {
        "prefix_blend": 1.0,
        "vocal_source_blend": 0.0,
        "vocal_start_bin": 10,
        "vocal_end_bin": 42,
        "mel_time_smooth": 3,
        "mel_freq_smooth": 0,
        "vocal_mix_gain": 0.95,
        "backing_timing_mode": "anchorgrid_perc_to_source",
        "vocal_delay_ms": 0.0,
    }
    return [
        {
            "label": "anchor_sep_d_baseline",
            **common,
            "t_start": 336,
            "t_start_end": 236,
            "reanchor_every": 4,
            "reanchor_t_start": 192,
            "guidance_scale": 2.55,
            "style_strength": 1.14,
            "source_prefix_blend": 0.10,
            "source_mel_blend": 0.00,
            "hf_source_blend": 0.02,
            "hf_start_bin": 60,
            "accomp_mix_gain": 1.00,
            "backing_source_blend": 0.08,
            "backing_percussive_blend": 0.10,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.30,
            "backing_dewarble_strength": 0.25,
        },
        {
            "label": "anchor_c_exfar",
            **common,
            "t_start": 332,
            "t_start_end": 232,
            "reanchor_every": 4,
            "reanchor_t_start": 190,
            "guidance_scale": 2.46,
            "style_strength": 1.18,
            "style_cond_mode": "farthest_exemplar",
            "source_prefix_blend": 0.08,
            "source_mel_blend": 0.00,
            "hf_source_blend": 0.02,
            "hf_start_bin": 60,
            "accomp_mix_gain": 1.00,
            "backing_source_blend": 0.04,
            "backing_percussive_blend": 0.06,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.42,
            "backing_dewarble_strength": 0.34,
        },
        {
            "label": "anchor_d_exfar",
            **common,
            "t_start": 344,
            "t_start_end": 238,
            "reanchor_every": 4,
            "reanchor_t_start": 194,
            "guidance_scale": 2.60,
            "style_strength": 1.24,
            "style_cond_mode": "farthest_exemplar",
            "source_prefix_blend": 0.05,
            "source_mel_blend": 0.00,
            "hf_source_blend": 0.01,
            "hf_start_bin": 60,
            "accomp_mix_gain": 1.02,
            "backing_source_blend": 0.02,
            "backing_percussive_blend": 0.04,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.46,
            "backing_dewarble_strength": 0.36,
        },
        {
            "label": "anchor_hybrid_exfar",
            **common,
            "t_start": 336,
            "t_start_end": 236,
            "reanchor_every": 4,
            "reanchor_t_start": 192,
            "guidance_scale": 2.52,
            "style_strength": 1.20,
            "style_cond_mode": "hybrid_exemplar",
            "style_exemplar_weight": 0.75,
            "source_prefix_blend": 0.06,
            "source_mel_blend": 0.00,
            "hf_source_blend": 0.01,
            "hf_start_bin": 60,
            "accomp_mix_gain": 1.01,
            "backing_source_blend": 0.03,
            "backing_percussive_blend": 0.05,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.44,
            "backing_dewarble_strength": 0.35,
        },
        {
            "label": "anchor_d_exfar_loud",
            **common,
            "t_start": 352,
            "t_start_end": 242,
            "reanchor_every": 5,
            "reanchor_t_start": 198,
            "guidance_scale": 2.72,
            "style_strength": 1.30,
            "style_cond_mode": "farthest_exemplar",
            "source_prefix_blend": 0.03,
            "source_mel_blend": 0.00,
            "hf_source_blend": 0.00,
            "hf_start_bin": 60,
            "accomp_mix_gain": 1.03,
            "backing_source_blend": 0.00,
            "backing_percussive_blend": 0.03,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.50,
            "backing_dewarble_strength": 0.32,
        },
    ]


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _local_sync(src: np.ndarray, cand: np.ndarray, sr: int, hop: int = 512) -> Dict[str, float]:
    src_env = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)
    cand_env = librosa.onset.onset_strength(y=cand, sr=sr, hop_length=hop)
    m = min(len(src_env), len(cand_env))
    src_env = src_env[:m]
    cand_env = cand_env[:m]
    win = 64
    step = 32
    lags: List[float] = []
    cors: List[float] = []
    for st in range(0, max(0, m - win), step):
        a = src_env[st : st + win] - float(np.mean(src_env[st : st + win]))
        best = (0, -1e18)
        for lag in range(-16, 17):
            if lag >= 0:
                x = a[: win - lag]
                b = cand_env[st + lag : st + lag + len(x)] - float(np.mean(cand_env[st + lag : st + lag + len(x)]))
            else:
                shift = -lag
                x = a[shift:win]
                b = cand_env[st : st + len(x)] - float(np.mean(cand_env[st : st + len(x)]))
            if len(x) < 8:
                continue
            denom = float(np.linalg.norm(x) * np.linalg.norm(b)) + 1e-8
            score = float(np.dot(x, b) / denom)
            if score > best[1]:
                best = (lag, score)
        lags.append(float(best[0] * hop / sr))
        cors.append(float(best[1]))
    lag_arr = np.asarray(lags, dtype=np.float32) if lags else np.zeros(1, dtype=np.float32)
    return {
        "local_mean_abs_lag_sec": float(np.mean(np.abs(lag_arr))),
        "local_mean_corr": float(np.mean(cors)) if cors else 0.0,
    }


def _mean_logmel(path: Path) -> np.ndarray:
    y, sr = _load_mono(path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return np.mean(np.log1p(mel), axis=1).astype(np.float32)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--resume-latest":
        parent = Path.home() / "Desktop" / "dggr_hybrid_exemplar_pull"
        runs = sorted([p for p in parent.iterdir() if p.is_dir() and p.name.startswith("genre_sep_")], key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            raise FileNotFoundError(f"No existing runs under {parent}")
        out_root = runs[0]
    else:
        out_root = Path.home() / "Desktop" / "dggr_hybrid_exemplar_pull" / f"genre_sep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out_root.mkdir(parents=True, exist_ok=True)

    oto = _load_tradeoff_module()
    cfg = HybridPushConfig(source_seconds=30.0)
    mert = oto.FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=5.0, device="auto", layer=-1)
    centroids = oto._build_mert_centroids(oto.OptimizeConfig(), out_root)
    settings = candidate_settings()
    songs = picked_songs()
    songs = songs[:2]

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(r["path"]), "source_genre": r["source_genre"]} for r in songs], indent=2), encoding="utf-8")

    manifest_path = out_root / "manifest.csv"
    rows: List[Dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    for song in songs:
        stems = _resolve_stems(cfg, song)
        src_accomp, sr = _load_mono(stems["accompaniment"])
        for setting in settings:
            for target in ACTIVE_TARGETS:
                run_setting = dict(setting)
                run_setting["target_genre"] = target
                render_dir = out_root / "clips" / setting["label"] / f"{_slug(Path(song['path']).stem)}__to__{target}"
                render_dir.mkdir(parents=True, exist_ok=True)
                _run_longform(cfg, run_setting, stems["accompaniment"], song["source_genre"], target, render_dir, cfg.seed + len(rows))
                final_mix = _make_mix(run_setting, stems, render_dir)
                backing = render_dir / "backing_fixed.wav"
                existing = None
                for row in rows:
                    if str(row.get("output_dir")) == str(render_dir):
                        existing = row
                        break
                if existing is not None:
                    continue
                row = oto._analyze_candidate(
                    source_wav=stems["accompaniment"],
                    generated_wav=backing,
                    target_genre=target,
                    mert=mert,
                    genre_centroids=centroids,
                    coherence_json=render_dir / "coherence_metrics.json",
                    meta={
                        "source_audio": str(song["path"]),
                        "setting_label": setting["label"],
                        "target_genre": target,
                        "output_dir": str(render_dir),
                        "final_mix_wav": str(final_mix),
                    },
                )
                backing_y, _ = _load_mono(backing)
                row.update(_local_sync(src_accomp, backing_y, sr))
                row["target_genre"] = target
                row["setting_label"] = setting["label"]
                feat = oto._mert_feat_for_audio(mert, backing)
                target_cos = float(row["style_target_cos_gen"])
                other_cos = [
                    float(np.dot(feat, centroids[g]) / (np.linalg.norm(feat) * np.linalg.norm(centroids[g]) + 1e-8))
                    for g in ACTIVE_TARGETS
                    if g != target and g in centroids
                ]
                row["target_margin"] = float(target_cos - max(other_cos)) if other_cos else target_cos
                rows.append(row)
                with manifest_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

    if not rows:
        raise RuntimeError("No completed rows found")

    # Pairwise target separation per source + setting.
    by_group: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["source_audio"]), str(row["setting_label"]))
        by_group.setdefault(key, []).append(row)
    sep_bonus: Dict[tuple[str, str, str], float] = {}
    spectral_sep_bonus: Dict[tuple[str, str, str], float] = {}
    for key, group_rows in by_group.items():
        feat_map: Dict[str, np.ndarray] = {}
        mel_map: Dict[str, np.ndarray] = {}
        for row in group_rows:
            target = str(row["target_genre"])
            feat_map[target] = oto._mert_feat_for_audio(mert, Path(str(row["output_dir"])) / "backing_fixed.wav")
            mel_map[target] = _mean_logmel(Path(str(row["output_dir"])) / "backing_fixed.wav")
        targets = sorted(feat_map)
        pair_dists: Dict[str, float] = {t: 0.0 for t in targets}
        pair_counts: Dict[str, int] = {t: 0 for t in targets}
        mel_pair_dists: Dict[str, float] = {t: 0.0 for t in targets}
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                a = targets[i]
                b = targets[j]
                cos = float(np.dot(feat_map[a], feat_map[b]) / (np.linalg.norm(feat_map[a]) * np.linalg.norm(feat_map[b]) + 1e-8))
                dist = 1.0 - cos
                mel_dist = float(np.mean(np.abs(mel_map[a] - mel_map[b])))
                pair_dists[a] += dist
                pair_dists[b] += dist
                mel_pair_dists[a] += mel_dist
                mel_pair_dists[b] += mel_dist
                pair_counts[a] += 1
                pair_counts[b] += 1
        for target in targets:
            bonus = pair_dists[target] / max(1, pair_counts[target])
            mel_bonus = mel_pair_dists[target] / max(1, pair_counts[target])
            sep_bonus[(key[0], key[1], target)] = float(bonus)
            spectral_sep_bonus[(key[0], key[1], target)] = float(mel_bonus)

    for row in rows:
        row["separation_bonus"] = sep_bonus[(str(row["source_audio"]), str(row["setting_label"]), str(row["target_genre"]))]
        row["spectral_sep_bonus"] = spectral_sep_bonus[(str(row["source_audio"]), str(row["setting_label"]), str(row["target_genre"]))]
        row["overall_score"] = float(
            1.55 * row["target_margin"]
            + 0.75 * row["style_gain"]
            + 0.50 * row["movement"]
            + 1.10 * row["separation_bonus"]
            + 0.70 * row["spectral_sep_bonus"]
            + 0.15 * row["local_mean_corr"]
            - 0.20 * row["local_mean_abs_lag_sec"]
            - 0.32 * row["gen_start_hf_roughness"]
        )

    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    winners: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for target in ACTIVE_TARGETS:
        subset = [r for r in rows if r["target_genre"] == target]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in subset:
            grouped.setdefault(str(row["setting_label"]), []).append(row)
        scored: List[Dict[str, Any]] = []
        for label, group_rows in grouped.items():
            scored.append(
                {
                    "target_genre": target,
                    "setting_label": label,
                    "n_rows": len(group_rows),
                    "mean_overall_score": float(np.mean([float(r["overall_score"]) for r in group_rows])),
                    "mean_target_margin": float(np.mean([float(r["target_margin"]) for r in group_rows])),
                    "mean_style_gain": float(np.mean([float(r["style_gain"]) for r in group_rows])),
                    "mean_movement": float(np.mean([float(r["movement"]) for r in group_rows])),
                    "mean_separation_bonus": float(np.mean([float(r["separation_bonus"]) for r in group_rows])),
                    "mean_spectral_sep_bonus": float(np.mean([float(r["spectral_sep_bonus"]) for r in group_rows])),
                    "mean_warble": float(np.mean([float(r["gen_start_hf_roughness"]) for r in group_rows])),
                }
            )
        scored.sort(key=lambda r: r["mean_overall_score"], reverse=True)
        summary_rows.extend(scored)
        winners[target] = scored[0]

    (out_root / "summary_rows.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (out_root / "winner_map.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    summary = {
        "output_root": str(out_root),
        "winner_map": winners,
        "n_rows": len(rows),
        "targets": ACTIVE_TARGETS,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
