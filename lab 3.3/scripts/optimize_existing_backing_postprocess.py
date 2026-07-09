from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_hybrid_vocal_push_compare import (  # noqa: E402
    _json_default,
    _make_mix,
)


ACTIVE_TARGETS = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]
VDEBLEED = {
    "baroque_classical": 0.35,
    "hiphop_xtc": 0.35,
    "lofi_hh_lfbb": 0.45,
    "cc0_other": 0.0,
}


def _load_tradeoff_module():
    path = THIS_DIR / "offline_tradeoff_optimize.py"
    spec = importlib.util.spec_from_file_location("existing_backing_tradeoff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["existing_backing_tradeoff"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def _mean_logmel(path: Path) -> np.ndarray:
    y, sr = _load_mono(path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80)
    return np.mean(np.log1p(mel), axis=1).astype(np.float32)


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


def candidate_variants() -> List[Dict[str, Any]]:
    common = {
        "vocal_mix_gain": 0.95,
        "vocal_timing_mode": "none",
        "vocal_delay_ms": 0.0,
        "vocal_debleed_floor": 0.18,
        "backing_timing_mode": "anchorgrid_perc_to_source",
    }
    return [
        {
            "label": "base_sync",
            "accomp_mix_gain": 0.93,
            "backing_source_blend": 0.20,
            "backing_percussive_blend": 0.25,
            "backing_post_mode": "none",
            "backing_post_strength": 0.0,
            "backing_dewarble_strength": 0.0,
            **common,
        },
        {
            "label": "sep_a",
            "accomp_mix_gain": 0.97,
            "backing_source_blend": 0.12,
            "backing_percussive_blend": 0.12,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.22,
            "backing_dewarble_strength": 0.22,
            **common,
        },
        {
            "label": "sep_b",
            "accomp_mix_gain": 0.99,
            "backing_source_blend": 0.08,
            "backing_percussive_blend": 0.10,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.30,
            "backing_dewarble_strength": 0.25,
            **common,
        },
        {
            "label": "sep_c",
            "accomp_mix_gain": 0.99,
            "backing_source_blend": 0.10,
            "backing_percussive_blend": 0.08,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.26,
            "backing_dewarble_strength": 0.28,
            **common,
        },
        {
            "label": "sep_d",
            "accomp_mix_gain": 1.00,
            "backing_source_blend": 0.06,
            "backing_percussive_blend": 0.08,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.36,
            "backing_dewarble_strength": 0.32,
            **common,
        },
    ]


def cases() -> List[Dict[str, Any]]:
    home = Path.home()
    stems_root = home / "Desktop" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems"
    return [
        {
            "song_label": "sza",
            "stems_dir": stems_root / "sza_f2f",
            "targets": {
                "baroque_classical": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__baroque_classical",
                "hiphop_xtc": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__hiphop_xtc",
                "lofi_hh_lfbb": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__lofi_hh_lfbb",
                "cc0_other": home / "Desktop" / "dggr_hybrid_genre_separation" / "genre_sep_20260331_114413" / "clips" / "anchor_style_pull_c" / "sza_f2f__to__cc0_other",
            },
        },
        {
            "song_label": "bea",
            "stems_dir": stems_root / "beabadoobee_fairy_song",
            "targets": {
                "baroque_classical": home / "Desktop" / "dggr_hybrid_vocal_auto_best" / "hybrid_auto_best_20260331_112800_stylepush" / "bea" / "clips" / "style_pull_a" / "000_beabadoobee_fairy_song__to__baroque_classical",
                "hiphop_xtc": home / "Desktop" / "dggr_hybrid_vocal_auto_best" / "hybrid_auto_best_20260331_112800_stylepush" / "bea" / "clips" / "style_pull_a" / "001_beabadoobee_fairy_song__to__hiphop_xtc",
                "lofi_hh_lfbb": home / "Desktop" / "dggr_hybrid_vocal_auto_best" / "hybrid_auto_best_20260331_112800_stylepush" / "bea" / "clips" / "style_pull_a" / "002_beabadoobee_fairy_song__to__lofi_hh_lfbb",
                "cc0_other": home / "Desktop" / "dggr_hybrid_vocal_auto_best" / "hybrid_auto_best_20260331_112800_stylepush" / "bea" / "clips" / "style_pull_b" / "003_beabadoobee_fairy_song__to__cc0_other",
            },
        },
        {
            "song_label": "mag",
            "stems_dir": stems_root / "magdalena_bay_imaginal_disk_01_06_fear_sex",
            "targets": {
                "baroque_classical": home / "Desktop" / "dggr_vocal_preserve_compare" / "vocal_compare_20260330_143610" / "clips" / "a_style" / "004_magdalena_bay_imaginal_disk_01_06_fear_sex__to__baroque_classical",
                "hiphop_xtc": home / "Desktop" / "dggr_vocal_preserve_compare" / "vocal_compare_20260330_143610" / "clips" / "a_style" / "005_magdalena_bay_imaginal_disk_01_06_fear_sex__to__hiphop_xtc",
                "lofi_hh_lfbb": home / "Desktop" / "dggr_vocal_preserve_compare" / "vocal_compare_20260330_143610" / "clips" / "a_style" / "006_magdalena_bay_imaginal_disk_01_06_fear_sex__to__lofi_hh_lfbb",
                "cc0_other": home / "Desktop" / "dggr_vocal_preserve_compare" / "vocal_compare_20260330_143610" / "clips" / "a_style" / "007_magdalena_bay_imaginal_disk_01_06_fear_sex__to__cc0_other",
            },
        },
    ]


def _ensure_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        import shutil

        shutil.copyfile(src, dst)


def main() -> None:
    out_root = Path.home() / "Desktop" / "dggr_hybrid_backing_post_opt" / f"postopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    oto = _load_tradeoff_module()
    mert = oto.FrozenMERT(model_id="m-a-p/MERT-v1-95M", chunk_seconds=5.0, device="auto", layer=-1)
    centroids = oto._build_mert_centroids(oto.OptimizeConfig(), out_root)
    variants = candidate_variants()
    cases_list = cases()
    (out_root / "variants.json").write_text(json.dumps(variants, indent=2), encoding="utf-8")
    (out_root / "cases.json").write_text(json.dumps([{k: str(v) if isinstance(v, Path) else v for k, v in case.items() if k != "targets"} | {"targets": {tg: str(path) for tg, path in case["targets"].items()}} for case in cases_list], indent=2, default=_json_default), encoding="utf-8")

    rows: List[Dict[str, Any]] = []
    for case in cases_list:
        stems = {
            "source_clip": case["stems_dir"] / "source_clip.wav",
            "vocals": case["stems_dir"] / "vocals.wav",
            "accompaniment": case["stems_dir"] / "accompaniment.wav",
        }
        src_accomp, sr = _load_mono(stems["accompaniment"])
        for target, base_render in case["targets"].items():
            for need in [stems["source_clip"], stems["vocals"], stems["accompaniment"], base_render / "longform_coherent.wav", base_render / "coherence_metrics.json"]:
                if not need.exists():
                    raise FileNotFoundError(f"Missing input: {need}")
            for variant in variants:
                render_dir = out_root / "clips" / variant["label"] / f"{case['song_label']}__to__{target}"
                render_dir.mkdir(parents=True, exist_ok=True)
                for name in ["source.wav", "longform_coherent.wav", "coherence_metrics.json"]:
                    src = stems["source_clip"] if name == "source.wav" else base_render / name
                    _ensure_link(src, render_dir / name)
                setting = dict(variant)
                setting["target_genre"] = target
                setting["vocal_debleed_strength"] = VDEBLEED[target]
                final_mix = _make_mix(setting, stems, render_dir)
                backing = render_dir / "backing_fixed.wav"
                row = oto._analyze_candidate(
                    source_wav=stems["accompaniment"],
                    generated_wav=backing,
                    target_genre=target,
                    mert=mert,
                    genre_centroids=centroids,
                    coherence_json=render_dir / "coherence_metrics.json",
                    meta={
                        "song_label": case["song_label"],
                        "target_genre": target,
                        "setting_label": variant["label"],
                        "output_dir": str(render_dir),
                        "final_mix_wav": str(final_mix),
                    },
                )
                backing_y, _ = _load_mono(backing)
                row.update(_local_sync(src_accomp, backing_y, sr))
                row["song_label"] = case["song_label"]
                row["target_genre"] = target
                row["setting_label"] = variant["label"]
                rows.append(row)

    feat_map: Dict[tuple[str, str, str], np.ndarray] = {}
    mel_map: Dict[tuple[str, str, str], np.ndarray] = {}
    for row in rows:
        key = (str(row["song_label"]), str(row["setting_label"]), str(row["target_genre"]))
        out_dir = Path(str(row["output_dir"])) / "backing_fixed.wav"
        feat_map[key] = oto._mert_feat_for_audio(mert, out_dir)
        mel_map[key] = _mean_logmel(out_dir)

    for row in rows:
        song_key = str(row["song_label"])
        setting_key = str(row["setting_label"])
        target = str(row["target_genre"])
        target_feat = feat_map[(song_key, setting_key, target)]
        other_cos = []
        for g in ACTIVE_TARGETS:
            if g == target or g not in centroids:
                continue
            other_cos.append(float(np.dot(target_feat, centroids[g]) / (np.linalg.norm(target_feat) * np.linalg.norm(centroids[g]) + 1e-8)))
        row["target_margin"] = float(row["style_target_cos_gen"] - max(other_cos)) if other_cos else float(row["style_target_cos_gen"])
        sep_vals: List[float] = []
        mel_vals: List[float] = []
        for other in ACTIVE_TARGETS:
            if other == target:
                continue
            other_feat = feat_map[(song_key, setting_key, other)]
            sep_vals.append(1.0 - float(np.dot(target_feat, other_feat) / (np.linalg.norm(target_feat) * np.linalg.norm(other_feat) + 1e-8)))
            mel_vals.append(float(np.mean(np.abs(mel_map[(song_key, setting_key, target)] - mel_map[(song_key, setting_key, other)]))))
        row["separation_bonus"] = float(np.mean(sep_vals)) if sep_vals else 0.0
        row["spectral_sep_bonus"] = float(np.mean(mel_vals)) if mel_vals else 0.0
        row["overall_score"] = float(
            1.55 * row["target_margin"]
            + 0.75 * row["style_gain"]
            + 0.50 * row["movement"]
            + 1.10 * row["separation_bonus"]
            + 0.80 * row["spectral_sep_bonus"]
            + 0.15 * row["local_mean_corr"]
            - 0.20 * row["local_mean_abs_lag_sec"]
            - 0.35 * row["gen_start_hf_roughness"]
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
        for label, group in grouped.items():
            scored.append(
                {
                    "target_genre": target,
                    "setting_label": label,
                    "n_rows": len(group),
                    "mean_overall_score": float(np.mean([float(r["overall_score"]) for r in group])),
                    "mean_target_margin": float(np.mean([float(r["target_margin"]) for r in group])),
                    "mean_style_gain": float(np.mean([float(r["style_gain"]) for r in group])),
                    "mean_movement": float(np.mean([float(r["movement"]) for r in group])),
                    "mean_separation_bonus": float(np.mean([float(r["separation_bonus"]) for r in group])),
                    "mean_spectral_sep_bonus": float(np.mean([float(r["spectral_sep_bonus"]) for r in group])),
                    "mean_warble": float(np.mean([float(r["gen_start_hf_roughness"]) for r in group])),
                }
            )
        scored.sort(key=lambda r: r["mean_overall_score"], reverse=True)
        summary_rows.extend(scored)
        winners[target] = scored[0]

    (out_root / "summary_rows.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    (out_root / "winner_map.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")
    summary = {"output_root": str(out_root), "winner_map": winners, "n_rows": len(rows)}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
