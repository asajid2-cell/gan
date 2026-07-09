from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .lab3_bridge import load_audio_chunk
from .lab3_diffusion_data import DIFFUSION_SR, extract_bigvgan_mel_np, load_diffusion_cache, pad_or_trim
from .lab3_diffusion_train import load_bigvgan_robust
from .real_music_transfer import (
    RetrievalFusionUNet,
    _device_from_arg,
    build_track_bank,
    choose_donor_track,
    generate_longform,
)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(x), len(y))
    if n < 4:
        return 0.0
    x = x[:n]
    y = y[:n]
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n]
    y = y[:n]
    return float(np.dot(x, y) / ((np.linalg.norm(x) + 1e-8) * (np.linalg.norm(y) + 1e-8)))


def load_audio_mono(path: Path, seconds: float = 0.0) -> np.ndarray:
    y, _ = librosa.load(
        str(path),
        sr=DIFFUSION_SR,
        mono=True,
        duration=float(seconds) if float(seconds) > 0 else None,
        dtype=np.float32,
        res_type="soxr_hq",
    )
    if len(y) == 0:
        return np.zeros((1,), dtype=np.float32)
    return librosa.util.normalize(y).astype(np.float32)


def audio_metrics(audio: np.ndarray, sr: int = DIFFUSION_SR) -> Dict[str, float]:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(y) < 1024:
        y = np.pad(y, (0, 1024 - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=2.0)
    mel_db = librosa.power_to_db(mel + 1e-8, ref=np.max)
    low = float(np.mean(mel[:18, :]))
    mid = float(np.mean(mel[18:56, :]))
    high = float(np.mean(mel[56:, :]))
    total = low + mid + high + 1e-8
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo = librosa.feature.tempo(y=y, sr=sr)
    return {
        "duration_sec": float(len(y) / float(sr)),
        "tempo": float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.0,
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "centroid_mean": float(np.mean(centroid)),
        "centroid_std": float(np.std(centroid)),
        "spectral_flatness": float(np.mean(librosa.feature.spectral_flatness(y=y))),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        "low_ratio": float(low / total),
        "mid_ratio": float(mid / total),
        "high_ratio": float(high / total),
        "dynamic_range_db": float(np.percentile(mel_db, 95) - np.percentile(mel_db, 5)),
        "warble": float(np.mean(np.abs(np.diff(centroid))) / (np.mean(centroid) + 1e-8)),
        "fullness": float(np.clip((low + mid) / total, 0.0, 1.0)),
        "onset_mean": float(np.mean(onset)),
        "chroma_mean": chroma.mean(axis=1).astype(float).tolist(),
        "onset_env": onset.astype(float).tolist(),
        "rms_env": rms.astype(float).tolist(),
    }


def _compact_metrics(m: Dict[str, Any]) -> np.ndarray:
    vals = [
        float(m["tempo"]),
        float(m["rms_mean"]),
        float(m["centroid_mean"]),
        float(m["spectral_flatness"]),
        float(m["zcr"]),
        float(m["low_ratio"]),
        float(m["mid_ratio"]),
        float(m["high_ratio"]),
        float(m["dynamic_range_db"]),
        float(m["onset_mean"]),
    ]
    vals.extend(float(x) for x in m["chroma_mean"])
    return np.asarray(vals, dtype=np.float32)


def build_reference_profiles(cache_dir: Path, *, max_per_genre: int = 96, seconds: float = 12.0, seed: int = 328) -> Dict[str, Any]:
    index_df, _arrays, genre_to_idx, _meta = load_diffusion_cache(Path(cache_dir), mmap=True)
    rng = np.random.default_rng(int(seed))
    profiles: Dict[str, Any] = {}
    for genre, _idx in genre_to_idx.items():
        gdf = index_df[index_df["genre"].astype(str) == str(genre)]
        if len(gdf) == 0:
            continue
        take_n = min(int(max_per_genre), len(gdf))
        take = gdf.sample(n=take_n, random_state=int(rng.integers(0, 2**31 - 1)))
        feats: List[np.ndarray] = []
        examples: List[str] = []
        for _, row in take.iterrows():
            try:
                y = load_audio_mono(Path(str(row["path"])), seconds=float(seconds))
                feats.append(_compact_metrics(audio_metrics(y)))
                examples.append(str(row["path"]))
            except Exception:
                continue
        if feats:
            profiles[str(genre)] = {
                "mean": np.mean(np.stack(feats), axis=0).astype(float).tolist(),
                "std": np.std(np.stack(feats), axis=0).astype(float).tolist(),
                "n": int(len(feats)),
                "examples": examples[:8],
            }
    return {"cache_dir": str(cache_dir), "profiles": profiles}


def create_validation_plan(
    cache_dir: Path,
    out_path: Path,
    *,
    sources_per_genre: int = 2,
    targets_per_source: int = 3,
    seconds: float = 24.0,
    seed: int = 328,
) -> Dict[str, Any]:
    index_df, _arrays, genre_to_idx, _meta = load_diffusion_cache(Path(cache_dir), mmap=True)
    rng = np.random.default_rng(int(seed))
    genres = sorted(str(g) for g in genre_to_idx.keys())
    rows: List[Dict[str, Any]] = []
    for source_genre in genres:
        gdf = index_df[index_df["genre"].astype(str) == source_genre].drop_duplicates(subset=["track_id"])
        if len(gdf) == 0:
            continue
        take = gdf.sample(n=min(int(sources_per_genre), len(gdf)), random_state=int(rng.integers(0, 2**31 - 1)))
        target_pool = [g for g in genres if g != source_genre]
        if not target_pool:
            target_pool = genres
        for _, row in take.iterrows():
            if len(target_pool) <= int(targets_per_source):
                targets = target_pool
            else:
                targets = sorted(rng.choice(target_pool, size=int(targets_per_source), replace=False).tolist())
            for target in targets:
                rows.append(
                    {
                        "case_id": f"case_{len(rows):04d}",
                        "source_audio": str(row["path"]),
                        "source_genre": source_genre,
                        "target_genre": str(target),
                        "seconds": float(seconds),
                        "track_id": str(row.get("track_id", "")),
                    }
                )
    plan = {
        "cache_dir": str(cache_dir),
        "genres": genres,
        "sources_per_genre": int(sources_per_genre),
        "targets_per_source": int(targets_per_source),
        "seconds": float(seconds),
        "rows": rows,
    }
    _write_json(Path(out_path), plan)
    return plan


def render_validation_pack(
    *,
    checkpoint: Path,
    cache_dir: Path,
    plan_path: Path,
    out_dir: Path,
    device: str = "auto",
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.5,
    style_strength: float = 1.0,
    envelope_strength: float = 0.35,
) -> Dict[str, Any]:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    rows_by_case: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            for row in existing.get("rows", []):
                case_id = str(row.get("case_id", ""))
                if case_id and Path(str(row.get("generated_wav", ""))).exists():
                    rows_by_case[case_id] = dict(row)
        except Exception:
            rows_by_case = {}

    device_t = _device_from_arg(str(device))
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cache_dir), mmap=True)
    payload = torch.load(str(checkpoint), map_location=device_t, weights_only=False)
    ckpt_cfg = payload.get("cfg", {})
    max_frames = int(payload.get("meta", {}).get("max_frames", ckpt_cfg.get("max_frames", 320)))
    base_ch = int(ckpt_cfg.get("base_ch", 48))
    model = RetrievalFusionUNet(in_ch=16, num_genres=len(genre_to_idx), base_ch=base_ch).to(device_t)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    bank = build_track_bank(index_df, arrays)
    vocoder = load_bigvgan_robust(device=device_t)

    def write_manifest() -> Dict[str, Any]:
        ordered = [rows_by_case[str(case["case_id"])] for case in plan["rows"] if str(case["case_id"]) in rows_by_case]
        manifest = {"checkpoint": str(checkpoint), "cache_dir": str(cache_dir), "plan_path": str(plan_path), "rows": ordered}
        _write_json(manifest_path, manifest)
        pd.DataFrame(ordered).to_csv(out_dir / "manifest.csv", index=False)
        return manifest

    for case in plan["rows"]:
        case_id = str(case["case_id"])
        case_dir = out_dir / str(case["case_id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        out_wav = case_dir / f"{case['case_id']}__to__{case['target_genre']}.wav"
        out_json = out_wav.with_suffix(".json")
        if case_id in rows_by_case and out_wav.exists():
            print(json.dumps({"event": "validation_render_skip", "case_id": case_id, "done": len(rows_by_case), "total": len(plan["rows"])}), flush=True)
            continue
        if out_wav.exists() and out_json.exists():
            meta_out = json.loads(out_json.read_text(encoding="utf-8"))
            row = dict(case)
            row.update({"generated_wav": str(out_wav), "generation_meta": meta_out})
            rows_by_case[case_id] = row
            write_manifest()
            print(json.dumps({"event": "validation_render_recovered", "case_id": case_id, "done": len(rows_by_case), "total": len(plan["rows"])}), flush=True)
            continue
        target_genre = str(case["target_genre"])
        if target_genre not in genre_to_idx:
            raise ValueError(f"Unknown target genre '{target_genre}'. Available: {sorted(genre_to_idx)}")
        print(json.dumps({"event": "validation_render_start", "case_id": case_id, "target_genre": target_genre, "done": len(rows_by_case), "total": len(plan["rows"])}), flush=True)
        source = load_audio_chunk(Path(case["source_audio"]), sample_rate=DIFFUSION_SR, seconds=float(case.get("seconds", plan.get("seconds", 24.0))), start_sec=0.0)
        target_idx = int(genre_to_idx[target_genre])
        donor = choose_donor_track(source, bank, target_idx)
        generated = generate_longform(
            model,
            source_audio=source,
            target_genre_idx=target_idx,
            donor_track=donor,
            arrays=arrays,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=max_frames,
            chunk_seconds=float(chunk_seconds),
            overlap_seconds=float(overlap_seconds),
            vocoder=vocoder,
            device=device_t,
            style_strength=float(style_strength),
            envelope_strength=float(envelope_strength),
        )
        sf.write(str(out_wav), generated, DIFFUSION_SR)
        meta_out = {
            "out_wav": str(out_wav),
            "checkpoint": str(checkpoint),
            "cache_dir": str(cache_dir),
            "source_audio": str(case["source_audio"]),
            "target_genre": target_genre,
            "donor_track_id": str(donor["track_id"]),
            "seconds": float(case.get("seconds", plan.get("seconds", 24.0))),
            "chunk_seconds": float(chunk_seconds),
            "overlap_seconds": float(overlap_seconds),
            "style_strength": float(style_strength),
            "envelope_strength": float(envelope_strength),
        }
        _write_json(out_json, meta_out)
        row = dict(case)
        row.update({"generated_wav": str(out_wav), "generation_meta": meta_out})
        rows_by_case[case_id] = row
        manifest = write_manifest()
        if device_t.type == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"event": "validation_render_done", "case_id": case_id, "done": len(rows_by_case), "total": len(plan["rows"]), "out_wav": str(out_wav)}), flush=True)
    return write_manifest()


def evaluate_validation_pack(
    *,
    cache_dir: Path,
    plan_path: Path,
    pack_dir: Path,
    out_path: Path,
    reference_profiles_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pack = json.loads((Path(pack_dir) / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if reference_profiles_path and Path(reference_profiles_path).exists():
        ref = json.loads(Path(reference_profiles_path).read_text(encoding="utf-8"))
    else:
        ref = build_reference_profiles(Path(cache_dir))
        if reference_profiles_path:
            _write_json(Path(reference_profiles_path), ref)

    profile_map = {str(k): np.asarray(v["mean"], dtype=np.float32) for k, v in ref.get("profiles", {}).items()}
    profile_std_map = {str(k): np.maximum(np.asarray(v.get("std", []), dtype=np.float32), 1e-3) for k, v in ref.get("profiles", {}).items()}
    rows: List[Dict[str, Any]] = []
    for case in pack["rows"]:
        source_y = load_audio_mono(Path(case["source_audio"]), seconds=float(case.get("seconds", plan.get("seconds", 24.0))))
        gen_y = load_audio_mono(Path(case["generated_wav"]))
        source_m = audio_metrics(source_y)
        gen_m = audio_metrics(gen_y)
        gen_vec = _compact_metrics(gen_m)
        target = str(case["target_genre"])
        source_genre = str(case.get("source_genre", ""))
        target_vec = profile_map.get(target)
        source_vec = profile_map.get(source_genre)
        target_std = profile_std_map.get(target)
        source_std = profile_std_map.get(source_genre)
        target_cos = _cosine(gen_vec, target_vec) if target_vec is not None else 0.0
        source_style_cos = _cosine(gen_vec, source_vec) if source_vec is not None else 0.0
        target_style_zdist = (
            float(np.mean(np.abs((gen_vec - target_vec) / target_std)))
            if target_vec is not None and target_std is not None and len(target_std) == len(gen_vec)
            else 0.0
        )
        source_style_zdist = (
            float(np.mean(np.abs((gen_vec - source_vec) / source_std)))
            if source_vec is not None and source_std is not None and len(source_std) == len(gen_vec)
            else 0.0
        )
        style_margin = float(source_style_zdist - target_style_zdist)
        chroma_corr = _cosine(np.asarray(source_m["chroma_mean"]), np.asarray(gen_m["chroma_mean"]))
        onset_corr = _safe_corr(np.asarray(source_m["onset_env"]), np.asarray(gen_m["onset_env"]))
        rms_corr = _safe_corr(np.asarray(source_m["rms_env"]), np.asarray(gen_m["rms_env"]))
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "source_genre": source_genre,
                "target_genre": target,
                "source_audio": str(case["source_audio"]),
                "generated_wav": str(case["generated_wav"]),
                "target_style_cos": float(target_cos),
                "source_style_cos": float(source_style_cos),
                "target_style_zdist": float(target_style_zdist),
                "source_style_zdist": float(source_style_zdist),
                "style_margin": style_margin,
                "content_chroma_cos": float(chroma_corr),
                "content_onset_corr": float(onset_corr),
                "content_rms_corr": float(rms_corr),
                "warble": float(gen_m["warble"]),
                "fullness": float(gen_m["fullness"]),
                "dynamic_range_db": float(gen_m["dynamic_range_db"]),
                "hf_ratio": float(gen_m["high_ratio"]),
                "lf_ratio": float(gen_m["low_ratio"]),
            }
        )
    summary = {
        "n_cases": int(len(rows)),
        "mean_target_style_cos": float(np.mean([r["target_style_cos"] for r in rows])) if rows else 0.0,
        "mean_style_margin": float(np.mean([r["style_margin"] for r in rows])) if rows else 0.0,
        "mean_content_chroma_cos": float(np.mean([r["content_chroma_cos"] for r in rows])) if rows else 0.0,
        "mean_content_onset_corr": float(np.mean([r["content_onset_corr"] for r in rows])) if rows else 0.0,
        "mean_content_rms_corr": float(np.mean([r["content_rms_corr"] for r in rows])) if rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in rows])) if rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in rows])) if rows else 0.0,
        "rows": rows,
        "reference_profiles": ref,
    }
    _write_json(Path(out_path), summary)
    pd.DataFrame(rows).to_csv(Path(out_path).with_suffix(".csv"), index=False)
    return summary
