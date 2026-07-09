from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TARGET_GENRES = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]

from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_diffusion_data import extract_chroma, extract_onset, load_diffusion_cache, pad_or_trim


_GENRE_GRAFT_CACHE: Dict[str, Any] | None = None


def _path_pref_score(path_str: str, target_genre: str) -> float:
    s = path_str.lower()
    bad_tokens = ["vox", "vocal", "singer", "fx_", "fx\\", "_fx", "speech", "spoken", "siren", "voice"]
    score = 0.0
    if any(tok in s for tok in bad_tokens):
        score -= 2.5
    if target_genre == "hiphop_xtc":
        good = ["loop", "beat", "drum", "kit", "break", "construction", "seq", "groove", "adjuster"]
        if any(tok in s for tok in good):
            score += 1.8
        if "vox" in s or "fx" in s:
            score -= 3.0
    elif target_genre == "lofi_hh_lfbb":
        good = ["hh_lfbb", "lofi", "loop", "mid", "beat", "groove"]
        if any(tok in s for tok in good):
            score += 1.6
    elif target_genre == "baroque_classical":
        good = ["baroque", "classical", "organ", "string", "harpsi", "ensemble", "orchestra", "piano"]
        if any(tok in s for tok in good):
            score += 1.6
    elif target_genre == "cc0_other":
        good = ["instrument", "music", "organ", "piano", "drum", "loop"]
        if any(tok in s for tok in good):
            score += 0.8
    return float(score)


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported value: {type(value)!r}")


@dataclass
class HybridPushConfig:
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "Desktop Outputs" / "dggr_hybrid_vocal_push_compare")
    stem_cache_root: Path = field(default_factory=lambda: REPO_ROOT / "Desktop Outputs" / "dggr_hybrid_vocal_compare" / "hybrid_compare_20260330_150148" / "stems")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt")
    lab1_checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt")
    source_seconds: float = 60.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    ddim_steps: int = 50
    seed: int = 328


def picked_songs() -> List[Dict[str, Any]]:
    base = Path.home() / "Downloads"
    songs = [
        {"path": base / "SZA - F2F.flac", "source_genre": "cc0_other"},
        {"path": base / "beabadoobee - fairy song.flac", "source_genre": "cc0_other"},
        {"path": base / "Magdalena Bay - Imaginal Disk - 01-06 Fear, Sex.flac", "source_genre": "cc0_other"},
    ]
    for row in songs:
        if not Path(row["path"]).exists():
            raise FileNotFoundError(f"Missing compare song: {row['path']}")
    return songs


def settings_panel() -> List[Dict[str, Any]]:
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
            "label": "hybrid_push_b",
            "t_start": 300,
            "t_start_end": 214,
            "reanchor_every": 3,
            "reanchor_t_start": 178,
            "guidance_scale": 2.18,
            "style_strength": 0.84,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.32,
            "source_mel_blend": 0.01,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.06,
            "hf_start_bin": 58,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.94,
        },
        {
            "label": "hybrid_push_guard",
            "t_start": 290,
            "t_start_end": 210,
            "reanchor_every": 2,
            "reanchor_t_start": 172,
            "guidance_scale": 2.08,
            "style_strength": 0.79,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.35,
            "source_mel_blend": 0.03,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.10,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.91,
        },
        {
            "label": "style_pull_a",
            "t_start": 304,
            "t_start_end": 216,
            "reanchor_every": 3,
            "reanchor_t_start": 180,
            "guidance_scale": 2.22,
            "style_strength": 0.90,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.28,
            "source_mel_blend": 0.01,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.07,
            "hf_start_bin": 58,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.93,
        },
        {
            "label": "style_pull_b",
            "t_start": 312,
            "t_start_end": 222,
            "reanchor_every": 3,
            "reanchor_t_start": 184,
            "guidance_scale": 2.30,
            "style_strength": 0.98,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.24,
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
            "label": "style_pull_c",
            "t_start": 324,
            "t_start_end": 228,
            "reanchor_every": 4,
            "reanchor_t_start": 188,
            "guidance_scale": 2.42,
            "style_strength": 1.06,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.18,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.04,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.98,
        },
        {
            "label": "style_pull_d",
            "t_start": 336,
            "t_start_end": 236,
            "reanchor_every": 4,
            "reanchor_t_start": 192,
            "guidance_scale": 2.55,
            "style_strength": 1.14,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.12,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.03,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 1.00,
        },
        {
            "label": "style_pull_e",
            "t_start": 318,
            "t_start_end": 230,
            "reanchor_every": 5,
            "reanchor_t_start": 190,
            "guidance_scale": 2.48,
            "style_strength": 1.08,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.16,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.05,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.99,
        },
        {
            "label": "style_pull_c_sep",
            "t_start": 324,
            "t_start_end": 228,
            "reanchor_every": 4,
            "reanchor_t_start": 188,
            "guidance_scale": 2.42,
            "style_strength": 1.06,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.16,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.03,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.99,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.10,
            "backing_percussive_blend": 0.12,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.22,
            "backing_dewarble_strength": 0.22,
        },
        {
            "label": "style_pull_d_sep",
            "t_start": 336,
            "t_start_end": 236,
            "reanchor_every": 4,
            "reanchor_t_start": 192,
            "guidance_scale": 2.55,
            "style_strength": 1.14,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.10,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.02,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 1.00,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.08,
            "backing_percussive_blend": 0.10,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.30,
            "backing_dewarble_strength": 0.25,
        },
        {
            "label": "style_pull_e_sep",
            "t_start": 318,
            "t_start_end": 230,
            "reanchor_every": 5,
            "reanchor_t_start": 190,
            "guidance_scale": 2.48,
            "style_strength": 1.08,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.14,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.04,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 0.99,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.10,
            "backing_percussive_blend": 0.08,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.26,
            "backing_dewarble_strength": 0.28,
        },
        {
            "label": "style_pull_c_exfar",
            "t_start": 332,
            "t_start_end": 232,
            "reanchor_every": 4,
            "reanchor_t_start": 190,
            "guidance_scale": 2.46,
            "style_strength": 1.18,
            "style_cond_mode": "farthest_exemplar",
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.08,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.02,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 1.00,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.04,
            "backing_percussive_blend": 0.06,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.42,
            "backing_dewarble_strength": 0.34,
        },
        {
            "label": "style_pull_d_exfar",
            "t_start": 344,
            "t_start_end": 238,
            "reanchor_every": 4,
            "reanchor_t_start": 194,
            "guidance_scale": 2.60,
            "style_strength": 1.24,
            "style_cond_mode": "farthest_exemplar",
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.05,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.01,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 1.02,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.02,
            "backing_percussive_blend": 0.04,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.46,
            "backing_dewarble_strength": 0.36,
        },
        {
            "label": "style_pull_hybrid_exfar",
            "t_start": 336,
            "t_start_end": 236,
            "reanchor_every": 4,
            "reanchor_t_start": 192,
            "guidance_scale": 2.52,
            "style_strength": 1.20,
            "style_cond_mode": "hybrid_exemplar",
            "style_exemplar_weight": 0.75,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.06,
            "source_mel_blend": 0.00,
            "vocal_source_blend": 0.0,
            "vocal_start_bin": 10,
            "vocal_end_bin": 42,
            "hf_source_blend": 0.01,
            "hf_start_bin": 60,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
            "vocal_mix_gain": 0.95,
            "accomp_mix_gain": 1.01,
            "backing_timing_mode": "anchorgrid_perc_to_source",
            "backing_source_blend": 0.03,
            "backing_percussive_blend": 0.05,
            "backing_post_mode": "genre_separate",
            "backing_post_strength": 0.44,
            "backing_dewarble_strength": 0.35,
        },
    ]


def _resolve_stems(cfg: HybridPushConfig, song: Dict[str, Any]) -> Dict[str, Path]:
    stem_dir = cfg.stem_cache_root / _slug(Path(song["path"]).stem)
    source_clip = stem_dir / "source_clip.wav"
    vocals = stem_dir / "vocals.wav"
    accompaniment = stem_dir / "accompaniment.wav"
    if not (source_clip.exists() and vocals.exists() and accompaniment.exists()):
        raise FileNotFoundError(f"Missing cached stems for {song['path']} in {stem_dir}")
    return {"source_clip": source_clip, "vocals": vocals, "accompaniment": accompaniment}


def _run_longform(cfg: HybridPushConfig, setting: Dict[str, Any], render_source: Path, source_genre: str, target_genre: str, out_dir: Path, seed: int) -> None:
    if (out_dir / "longform_coherent.wav").exists():
        return
    cmd = [
        "python",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(cfg.cache_dir),
        "--checkpoint", str(cfg.checkpoint),
        "--lab1-checkpoint", str(cfg.lab1_checkpoint),
        "--source-audio", str(render_source),
        "--source-genre", str(source_genre),
        "--target-genre", str(target_genre),
        "--source-start-sec", "0.0",
        "--source-seconds", str(cfg.source_seconds),
        "--out-dir", str(out_dir),
        "--chunk-seconds", str(cfg.chunk_seconds),
        "--overlap-seconds", str(cfg.overlap_seconds),
        "--n-frames", str(cfg.n_frames),
        "--ddim-steps", str(cfg.ddim_steps),
        "--assemble-domain", "mel",
        "--device", "auto",
        "--seed", str(seed),
        "--t-start", str(setting["t_start"]),
        "--t-start-end", str(setting["t_start_end"]),
        "--reanchor-every", str(setting["reanchor_every"]),
        "--reanchor-t-start", str(setting["reanchor_t_start"]),
        "--guidance-scale", str(setting["guidance_scale"]),
        "--style-strength", str(setting["style_strength"]),
        "--prefix-blend", str(setting["prefix_blend"]),
        "--source-prefix-blend", str(setting["source_prefix_blend"]),
        "--source-mel-blend", str(setting["source_mel_blend"]),
        "--vocal-source-blend", str(setting["vocal_source_blend"]),
        "--vocal-start-bin", str(setting["vocal_start_bin"]),
        "--vocal-end-bin", str(setting["vocal_end_bin"]),
        "--hf-source-blend", str(setting["hf_source_blend"]),
        "--hf-start-bin", str(setting["hf_start_bin"]),
        "--mel-time-smooth", str(setting["mel_time_smooth"]),
        "--mel-freq-smooth", str(setting["mel_freq_smooth"]),
    ]
    if "style_cond_mode" in setting:
        cmd.extend(["--style-cond-mode", str(setting["style_cond_mode"])])
    if "style_exemplar_weight" in setting:
        cmd.extend(["--style-exemplar-weight", str(setting["style_exemplar_weight"])])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run.log").open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Longform failed: {render_source} -> {target_genre} [{setting['label']}]")


def _make_mix(setting: Dict[str, Any], stems: Dict[str, Path], rendered_dir: Path) -> Path:
    final_out = rendered_dir / "hybrid_longform_coherent.wav"
    vocals, sr_v = sf.read(str(stems["vocals"]), dtype="float32")
    src_accomp, sr_src = sf.read(str(stems["accompaniment"]), dtype="float32")
    accomp, sr_a = sf.read(str(rendered_dir / "longform_coherent.wav"), dtype="float32")
    if sr_v != sr_a or sr_src != sr_a:
        raise RuntimeError("Sample rate mismatch")
    if vocals.ndim > 1:
        vocals = vocals.mean(axis=1)
    if src_accomp.ndim > 1:
        src_accomp = src_accomp.mean(axis=1)
    if accomp.ndim > 1:
        accomp = accomp.mean(axis=1)

    vocal_debleed_strength = float(setting.get("vocal_debleed_strength", 0.0))
    vocal_debleed_floor = float(setting.get("vocal_debleed_floor", 0.18))
    if vocal_debleed_strength > 0.0:
        vocals = _debleed_vocals_with_source_accomp(
            vocals.astype(np.float32, copy=False),
            src_accomp.astype(np.float32, copy=False),
            sr_v,
            strength=vocal_debleed_strength,
            floor=vocal_debleed_floor,
        )

    backing_timing_mode = str(setting.get("backing_timing_mode", "none")).strip().lower()
    backing_source_blend = float(setting.get("backing_source_blend", 0.0))
    backing_percussive_blend = float(setting.get("backing_percussive_blend", 0.0))
    backing_post_mode = str(setting.get("backing_post_mode", "none")).strip().lower()
    backing_post_strength = float(setting.get("backing_post_strength", 0.0))
    backing_dewarble_strength = float(setting.get("backing_dewarble_strength", 0.0))
    target_genre = str(setting.get("target_genre", "")).strip().lower()
    if backing_timing_mode == "warp_to_source":
        fixed_accomp, backing_meta = _beat_warp_wave(accomp, accomp, src_accomp, sr_a)
    elif backing_timing_mode == "dtw_to_source":
        fixed_accomp, backing_meta = _dtw_warp_generated_to_source(src_accomp, accomp, sr_a)
    elif backing_timing_mode == "phrasegrid_to_source":
        fixed_accomp, backing_meta = _phrasegrid_warp_generated_to_source(src_accomp, accomp, vocals, sr_a)
    elif backing_timing_mode == "anchorgrid_to_source":
        fixed_accomp, backing_meta = _anchorgrid_warp_generated_to_source(src_accomp, accomp, sr_a)
    elif backing_timing_mode == "anchorgrid_perc_to_source":
        fixed_accomp, backing_meta = _anchorgrid_warp_generated_to_source(src_accomp, accomp, sr_a)
    else:
        fixed_accomp = accomp.astype(np.float32, copy=False)
        backing_meta = {"backing_warp_method": "none", "backing_reason": "timing_mode_none"}
    if backing_percussive_blend > 0.0:
        fixed_accomp, perc_meta = _blend_source_percussive(src_accomp, fixed_accomp, backing_percussive_blend)
        backing_meta.update(perc_meta)
    if backing_source_blend > 0.0:
        n_back = min(len(fixed_accomp), len(src_accomp))
        fixed_accomp = ((1.0 - backing_source_blend) * fixed_accomp[:n_back] + backing_source_blend * src_accomp[:n_back]).astype(np.float32)
    else:
        n_back = len(fixed_accomp)
    accomp = fixed_accomp[:n_back]
    if backing_post_mode != "none" or backing_dewarble_strength > 0.0:
        accomp, post_meta = _postprocess_backing(
            accomp,
            src_accomp[: len(accomp)],
            sr_a,
            target_genre=target_genre,
            mode=backing_post_mode,
            tone_strength=backing_post_strength,
            dewarble_strength=backing_dewarble_strength,
        )
        backing_meta.update(post_meta)
    sf.write(str(rendered_dir / "backing_fixed.wav"), accomp.astype(np.float32), sr_a)

    timing_mode = str(setting.get("vocal_timing_mode", "none")).strip().lower()
    delay_ms = float(setting.get("vocal_delay_ms", 0.0))
    if timing_mode == "beatwarp":
        proc_vocals, warp_meta = _beat_warp_wave(vocals, src_accomp, accomp, sr_a)
    else:
        proc_vocals = vocals.astype(np.float32, copy=False)
        warp_meta = {"warp_method": "none", "reason": "timing_mode_none"}
    if abs(delay_ms) > 1e-6:
        delay_samples = int(round(delay_ms * sr_a / 1000.0))
        proc_vocals = _shift_wave(proc_vocals, delay_samples)
    else:
        delay_samples = 0
    n = min(len(proc_vocals), len(accomp))
    mix = float(setting["vocal_mix_gain"]) * proc_vocals[:n] + float(setting["accomp_mix_gain"]) * accomp[:n]
    peak = float(np.max(np.abs(mix))) + 1e-8
    mix = (mix / peak * 0.95).astype(np.float32)
    sf.write(str(final_out), mix, sr_a)
    (rendered_dir / "hybrid_mix_meta.json").write_text(
        json.dumps(
            {
                **warp_meta,
                **backing_meta,
                "backing_timing_mode": backing_timing_mode,
                "backing_source_blend": backing_source_blend,
                "backing_percussive_blend": backing_percussive_blend,
                "backing_post_mode": backing_post_mode,
                "backing_post_strength": backing_post_strength,
                "backing_dewarble_strength": backing_dewarble_strength,
                "target_genre": target_genre,
                "timing_mode": timing_mode,
                "vocal_delay_ms": delay_ms,
                "vocal_delay_samples": int(delay_samples),
                "vocal_mix_gain": float(setting["vocal_mix_gain"]),
                "accomp_mix_gain": float(setting["accomp_mix_gain"]),
                "vocal_debleed_strength": vocal_debleed_strength,
                "vocal_debleed_floor": vocal_debleed_floor,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_out


def _shift_wave(y: np.ndarray, lag_samples: int) -> np.ndarray:
    if lag_samples == 0:
        return y.astype(np.float32, copy=False)
    out = np.zeros_like(y, dtype=np.float32)
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


def _estimate_alignment_lag(src_accomp: np.ndarray, gen_accomp: np.ndarray, sr: int) -> int:
    n = min(len(src_accomp), len(gen_accomp))
    if n < sr:
        return 0
    src = src_accomp[:n].astype(np.float32, copy=False)
    gen = gen_accomp[:n].astype(np.float32, copy=False)
    hop = 512
    src_env = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)
    gen_env = librosa.onset.onset_strength(y=gen, sr=sr, hop_length=hop)
    m = min(len(src_env), len(gen_env))
    if m < 8:
        return 0
    src_env = src_env[:m] - float(np.mean(src_env[:m]))
    gen_env = gen_env[:m] - float(np.mean(gen_env[:m]))
    max_lag_frames = max(1, int(round(1.0 * sr / hop)))
    best_lag = 0
    best_score = -1e18
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            a = src_env[: m - lag]
            b = gen_env[lag:m]
        else:
            shift = -lag
            a = src_env[shift:m]
            b = gen_env[: m - shift]
        if len(a) < 8 or len(b) < 8:
            continue
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        score = float(np.dot(a, b) / denom)
        if score > best_score:
            best_score = score
            best_lag = lag
    return int(best_lag * hop)


def _beat_warp_wave(y: np.ndarray, src_ref: np.ndarray, tgt_ref: np.ndarray, sr: int) -> tuple[np.ndarray, Dict[str, Any]]:
    desired_len = min(len(y), len(tgt_ref))
    if desired_len < sr:
        return y[:desired_len].astype(np.float32, copy=False), {"warp_method": "none", "reason": "too_short"}

    try:
        _, src_beats = librosa.beat.beat_track(y=src_ref[:desired_len], sr=sr, units="samples")
        _, tgt_beats = librosa.beat.beat_track(y=tgt_ref[:desired_len], sr=sr, units="samples")
    except Exception:
        return y[:desired_len].astype(np.float32, copy=False), {"warp_method": "none", "reason": "beat_track_failed"}

    src_beats = np.asarray(src_beats, dtype=np.int64)
    tgt_beats = np.asarray(tgt_beats, dtype=np.int64)
    if src_beats.size < 4 or tgt_beats.size < 4:
        return y[:desired_len].astype(np.float32, copy=False), {"warp_method": "none", "reason": "too_few_beats"}

    k = int(min(src_beats.size, tgt_beats.size))
    src_anchor = np.concatenate(([0], np.clip(src_beats[:k], 0, desired_len - 1), [desired_len - 1]))
    tgt_anchor = np.concatenate(([0], np.clip(tgt_beats[:k], 0, desired_len - 1), [desired_len - 1]))

    src_anchor = np.maximum.accumulate(src_anchor)
    tgt_anchor = np.maximum.accumulate(tgt_anchor)
    src_anchor = np.unique(src_anchor)
    tgt_anchor = np.unique(tgt_anchor)
    k2 = int(min(src_anchor.size, tgt_anchor.size))
    src_anchor = src_anchor[:k2]
    tgt_anchor = tgt_anchor[:k2]
    if k2 < 4:
        return y[:desired_len].astype(np.float32, copy=False), {"warp_method": "none", "reason": "deduped_beats"}

    target_positions = np.arange(desired_len, dtype=np.float32)
    source_positions = np.interp(target_positions, tgt_anchor.astype(np.float32), src_anchor.astype(np.float32))
    source_positions = np.clip(source_positions, 0.0, float(len(y) - 1))
    warped = np.interp(source_positions, np.arange(len(y), dtype=np.float32), y.astype(np.float32)).astype(np.float32)
    return warped, {
        "warp_method": "beat_anchor_interp",
        "src_beats_used": int(k2 - 2),
        "tgt_beats_used": int(k2 - 2),
        "src_duration_sec": float(desired_len) / float(sr),
        "tgt_duration_sec": float(desired_len) / float(sr),
    }


def _dtw_warp_generated_to_source(src: np.ndarray, gen: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, Dict[str, Any]]:
    n = min(len(src), len(gen))
    if n < sr:
        return gen[:n].astype(np.float32, copy=False), {"backing_warp_method": "none", "backing_reason": "too_short"}
    src = src[:n]
    gen = gen[:n]
    src_chroma = librosa.feature.chroma_cqt(y=src, sr=sr, hop_length=hop)
    gen_chroma = librosa.feature.chroma_cqt(y=gen, sr=sr, hop_length=hop)
    src_on = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)[None, :]
    gen_on = librosa.onset.onset_strength(y=gen, sr=sr, hop_length=hop)[None, :]
    src_feat = np.vstack([src_chroma * 0.7, src_on * 0.3]).astype(np.float32)
    gen_feat = np.vstack([gen_chroma * 0.7, gen_on * 0.3]).astype(np.float32)
    _, wp = librosa.sequence.dtw(X=src_feat, Y=gen_feat, metric="cosine")
    wp = np.asarray(wp[::-1], dtype=np.int64)
    src_frames = wp[:, 0].astype(np.float32)
    gen_frames = wp[:, 1].astype(np.float32)
    keep = np.concatenate(([True], np.diff(src_frames) > 0))
    src_frames = src_frames[keep]
    gen_frames = gen_frames[keep]
    if len(src_frames) < 2:
        return gen.astype(np.float32, copy=False), {"backing_warp_method": "none", "backing_reason": "dtw_path_too_short"}
    target_samples = np.arange(n, dtype=np.float32)
    src_samples = src_frames * float(hop)
    gen_samples = gen_frames * float(hop)
    if src_samples[0] > 0:
        src_samples = np.concatenate(([0.0], src_samples))
        gen_samples = np.concatenate(([0.0], gen_samples))
    if src_samples[-1] < n - 1:
        src_samples = np.concatenate((src_samples, [float(n - 1)]))
        gen_samples = np.concatenate((gen_samples, [float(n - 1)]))
    mapped_gen_positions = np.interp(target_samples, src_samples, gen_samples)
    mapped_gen_positions = np.clip(mapped_gen_positions, 0.0, float(len(gen) - 1))
    warped = np.interp(mapped_gen_positions, np.arange(len(gen), dtype=np.float32), gen.astype(np.float32)).astype(np.float32)
    return warped, {
        "backing_warp_method": "dtw_interp",
        "backing_hop": hop,
        "backing_path_points": int(len(src_frames)),
    }


def _vocal_phrase_intervals(vocals: np.ndarray, sr: int, hop: int = 512) -> List[tuple[int, int]]:
    if len(vocals) < max(2048, sr // 2):
        return [(0, len(vocals))]
    rms = librosa.feature.rms(y=vocals.astype(np.float32, copy=False), frame_length=2048, hop_length=hop).squeeze()
    if rms.size == 0:
        return [(0, len(vocals))]
    thr = max(0.02 * float(np.max(rms)), 2.0 * float(np.median(rms)))
    active = rms > thr
    segs: List[tuple[int, int]] = []
    i = 0
    while i < len(active):
        if not bool(active[i]):
            i += 1
            continue
        j = i + 1
        while j < len(active) and bool(active[j]):
            j += 1
        if (j - i) >= 8:
            segs.append((int(i * hop), int(min(len(vocals), j * hop))))
        i = j
    if not segs:
        return [(0, len(vocals))]
    merged: List[List[int]] = [[segs[0][0], segs[0][1]]]
    min_gap = int(round(0.25 * sr))
    pad = int(round(0.10 * sr))
    for start, end in segs[1:]:
        if start - merged[-1][1] < min_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [
        (int(max(0, start - pad)), int(min(len(vocals), end + pad)))
        for start, end in merged
    ]


def _phrasegrid_warp_generated_to_source(src: np.ndarray, gen: np.ndarray, vocals: np.ndarray, sr: int) -> tuple[np.ndarray, Dict[str, Any]]:
    n = min(len(src), len(gen), len(vocals))
    if n < sr:
        return gen[:n].astype(np.float32, copy=False), {"backing_warp_method": "none", "backing_reason": "too_short"}
    intervals = _vocal_phrase_intervals(vocals[:n], sr)
    points: List[int] = [0, n]
    for start, end in intervals:
        points.extend([int(start), int(end)])
    points = sorted(set(max(0, min(n, p)) for p in points))
    blocks = [(a, b) for a, b in zip(points[:-1], points[1:]) if (b - a) > (sr // 2)]
    if not blocks:
        blocks = [(0, n)]

    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    fade = int(round(0.05 * sr))
    dtw_points = 0
    for start, end in blocks:
        warped_seg, meta = _dtw_warp_generated_to_source(src[start:end], gen[start:end], sr)
        dtw_points += int(meta.get("backing_path_points", 0))
        m = min(len(warped_seg), end - start)
        if m <= 0:
            continue
        w = np.ones(m, dtype=np.float32)
        if m > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            w[:fade] = np.maximum(ramp, 1e-6)
            w[-fade:] = np.maximum(ramp[::-1], 1e-6)
        out[start : start + m] += warped_seg[:m] * w
        weight[start : start + m] += w
    keep = weight > 0
    out[keep] /= weight[keep]
    out[~keep] = gen[:n][~keep]
    return out.astype(np.float32, copy=False), {
        "backing_warp_method": "phrasegrid_dtw_interp",
        "backing_phrase_blocks": int(len(blocks)),
        "backing_phrase_intervals": [[int(a), int(b)] for a, b in blocks],
        "backing_path_points": int(dtw_points),
    }


def _anchor_frames(y: np.ndarray, sr: int, hop: int = 512) -> np.ndarray:
    if len(y) < sr:
        return np.asarray([0], dtype=np.int64)
    onset_env = librosa.onset.onset_strength(y=y.astype(np.float32, copy=False), sr=sr, hop_length=hop)
    delta = 0.15 * float(np.max(onset_env) + 1e-6)
    onset_peaks = librosa.util.peak_pick(onset_env, pre_max=2, post_max=2, pre_avg=6, post_avg=6, delta=delta, wait=2)
    beat_frames = librosa.beat.beat_track(y=y.astype(np.float32, copy=False), sr=sr, hop_length=hop, units="frames")[1]
    frames = np.unique(np.concatenate([np.asarray(onset_peaks, dtype=np.int64), np.asarray(beat_frames, dtype=np.int64)]))
    if frames.size == 0:
        return np.asarray([0], dtype=np.int64)
    keep = [int(frames[0])]
    for frame in frames[1:].tolist():
        if frame - keep[-1] >= 2:
            keep.append(int(frame))
    return np.asarray(keep, dtype=np.int64)


def _anchorgrid_warp_generated_to_source(src: np.ndarray, gen: np.ndarray, sr: int, hop: int = 512) -> tuple[np.ndarray, Dict[str, Any]]:
    n = min(len(src), len(gen))
    if n < sr:
        return gen[:n].astype(np.float32, copy=False), {"backing_warp_method": "none", "backing_reason": "too_short"}
    src = src[:n]
    gen = gen[:n]
    src_chroma = librosa.feature.chroma_cqt(y=src, sr=sr, hop_length=hop)
    gen_chroma = librosa.feature.chroma_cqt(y=gen, sr=sr, hop_length=hop)
    src_on = librosa.onset.onset_strength(y=src, sr=sr, hop_length=hop)[None, :]
    gen_on = librosa.onset.onset_strength(y=gen, sr=sr, hop_length=hop)[None, :]
    src_feat = np.vstack([src_chroma * 0.55, src_on * 0.45]).astype(np.float32)
    gen_feat = np.vstack([gen_chroma * 0.55, gen_on * 0.45]).astype(np.float32)
    _, wp = librosa.sequence.dtw(X=src_feat, Y=gen_feat, metric="cosine")
    wp = np.asarray(wp[::-1], dtype=np.int64)
    src_frames = wp[:, 0].astype(np.float32)
    gen_frames = wp[:, 1].astype(np.float32)
    keep = np.concatenate(([True], np.diff(src_frames) > 0))
    src_frames = src_frames[keep]
    gen_frames = gen_frames[keep]
    if len(src_frames) < 2:
        return gen.astype(np.float32, copy=False), {"backing_warp_method": "none", "backing_reason": "dtw_path_too_short"}

    src_anchor_frames = _anchor_frames(src, sr, hop=hop).astype(np.float32)
    mapped_gen_frames = np.interp(src_anchor_frames, src_frames, gen_frames)
    src_anchor_samples = np.clip(src_anchor_frames * float(hop), 0.0, float(n - 1))
    gen_anchor_samples = np.clip(mapped_gen_frames * float(hop), 0.0, float(n - 1))
    if src_anchor_samples[0] > 0.0:
        src_anchor_samples = np.concatenate(([0.0], src_anchor_samples))
        gen_anchor_samples = np.concatenate(([0.0], gen_anchor_samples))
    if src_anchor_samples[-1] < float(n - 1):
        src_anchor_samples = np.concatenate((src_anchor_samples, [float(n - 1)]))
        gen_anchor_samples = np.concatenate((gen_anchor_samples, [float(n - 1)]))
    target_samples = np.arange(n, dtype=np.float32)
    mapped_gen_positions = np.interp(target_samples, src_anchor_samples, gen_anchor_samples)
    mapped_gen_positions = np.clip(mapped_gen_positions, 0.0, float(len(gen) - 1))
    warped = np.interp(mapped_gen_positions, np.arange(len(gen), dtype=np.float32), gen.astype(np.float32)).astype(np.float32)
    return warped, {
        "backing_warp_method": "anchorgrid_interp",
        "backing_hop": hop,
        "backing_anchor_points": int(len(src_anchor_samples)),
        "backing_path_points": int(len(src_frames)),
    }


def _blend_source_percussive(src: np.ndarray, gen: np.ndarray, blend: float) -> tuple[np.ndarray, Dict[str, Any]]:
    n = min(len(src), len(gen))
    src = src[:n].astype(np.float32, copy=False)
    gen = gen[:n].astype(np.float32, copy=False)
    gen_h, gen_p = librosa.effects.hpss(gen)
    _, src_p = librosa.effects.hpss(src)
    mixed = (gen_h + (1.0 - blend) * gen_p + blend * src_p).astype(np.float32)
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = (mixed / peak).astype(np.float32)
    return mixed, {"backing_percussive_injected": True, "backing_percussive_blend": float(blend)}


def _genre_graft_cache() -> Dict[str, Any]:
    global _GENRE_GRAFT_CACHE
    if _GENRE_GRAFT_CACHE is not None:
        return _GENRE_GRAFT_CACHE
    cfg = HybridPushConfig()
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cfg.cache_dir, mmap=True)
    _GENRE_GRAFT_CACHE = {
        "index_df": index_df,
        "arrays": arrays,
        "genre_to_idx": genre_to_idx,
        "meta": meta,
        "genre_idx": np.asarray(arrays["genre_idx"], dtype=np.int64),
        "track_ids": index_df["track_id"].astype(str).to_numpy(),
    }
    return _GENRE_GRAFT_CACHE


def _cache_like_chunk_features(y: np.ndarray, sr: int, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    chroma = extract_chroma(y.astype(np.float32, copy=False), sr=sr)
    chroma = pad_or_trim(chroma, n_frames, axis=1, pad_val=0.0).astype(np.float32)
    onset = extract_onset(y.astype(np.float32, copy=False), sr=sr)
    onset = pad_or_trim(onset, n_frames, axis=0, pad_val=0.0).astype(np.float32)
    return chroma, onset


def _graft_match_score(
    seg_chroma_v: np.ndarray,
    seg_onset_v: np.ndarray,
    ref_chroma: np.ndarray,
    ref_onset: np.ndarray,
) -> float:
    ref_chroma = ref_chroma / (np.linalg.norm(ref_chroma) + 1e-8)
    ref_onset = ref_onset / (float(np.linalg.norm(ref_onset)) + 1e-8)
    chroma_score = float(np.dot(seg_chroma_v, ref_chroma))
    onset_score = float(np.dot(seg_onset_v, ref_onset))
    return 0.72 * chroma_score + 0.28 * onset_score


def _pick_genre_graft_row(
    seg: np.ndarray,
    sr: int,
    target_genre: str,
    segment_idx: int,
    *,
    fixed_track_id: str | None = None,
    prev_start_sec: float | None = None,
    expected_step_sec: float = 2.5,
) -> Dict[str, Any]:
    cache = _genre_graft_cache()
    genre_to_idx = cache["genre_to_idx"]
    if target_genre not in genre_to_idx:
        raise KeyError(f"Unknown target genre for graft: {target_genre}")
    meta = cache["meta"]
    n_frames = int(getattr(meta, "n_frames", 432))
    seg_chroma, seg_onset = _cache_like_chunk_features(seg, sr, n_frames)
    seg_chroma_v = seg_chroma.reshape(-1)
    seg_chroma_v = seg_chroma_v / (np.linalg.norm(seg_chroma_v) + 1e-8)
    seg_onset = seg_onset / (float(np.linalg.norm(seg_onset)) + 1e-8)

    genre_idx = int(genre_to_idx[target_genre])
    rows = np.flatnonzero(cache["genre_idx"] == genre_idx)
    if rows.size == 0:
        raise RuntimeError(f"No cache rows for graft genre={target_genre}")
    track_ids = cache["track_ids"]
    if fixed_track_id is not None:
        rows = rows[track_ids[rows] == str(fixed_track_id)]
        if rows.size == 0:
            rows = np.flatnonzero(cache["genre_idx"] == genre_idx)
    rng = np.random.default_rng(328 + int(segment_idx) * 13 + int(genre_idx) * 101)
    if rows.size > 192:
        rows = rng.choice(rows, size=192, replace=False)
    best_score = -1e18
    best_row = int(rows[0])
    chroma_arr = cache["arrays"]["chroma"]
    onset_arr = cache["arrays"]["onset"]
    expected_start: float | None = None
    if prev_start_sec is not None:
        expected_start = float(prev_start_sec) + float(expected_step_sec)
    if fixed_track_id is not None and rows.size > 1:
        idx_df = cache["index_df"]
        starts = np.asarray([float(idx_df.iloc[int(r)]["start_sec"]) for r in rows.tolist()], dtype=np.float32)
        order = np.argsort(starts)
        rows = rows[order]
        starts = starts[order]
        if prev_start_sec is not None:
            forward_mask = starts >= (float(prev_start_sec) + 0.35 * float(expected_step_sec))
            if np.any(forward_mask):
                rows = rows[forward_mask]
                starts = starts[forward_mask]
        if expected_start is None:
            target_start = 0.0
        else:
            target_start = float(expected_start)
        pos = int(np.argmin(np.abs(starts - target_start)))
        window_lo = max(0, pos - 1)
        window_hi = min(len(rows), pos + 3)
        rows = rows[window_lo:window_hi]
    for ridx in rows.tolist():
        ref_chroma = np.asarray(chroma_arr[int(ridx)], dtype=np.float32).reshape(-1)
        ref_onset = np.asarray(onset_arr[int(ridx)], dtype=np.float32)
        score = _graft_match_score(seg_chroma_v, seg_onset, ref_chroma, ref_onset)
        if expected_start is not None:
            row_start = float(cache["index_df"].iloc[int(ridx)].get("start_sec", 0.0))
            score -= 0.16 * abs(row_start - expected_start)
            if row_start < float(prev_start_sec) - 1e-3:
                score -= 0.35
        if score > best_score:
            best_score = score
            best_row = int(ridx)
    row = cache["index_df"].iloc[int(best_row)]
    return {
        "row_idx": int(best_row),
        "track_id": str(row.get("track_id", "")),
        "path": Path(str(row["path"])),
        "start_sec": float(row.get("start_sec", 0.0)),
        "score": float(best_score),
        "chunk_seconds": float(getattr(meta, "chunk_sec", 5.0)),
    }


def _pick_genre_graft_track(src: np.ndarray, sr: int, target_genre: str) -> str | None:
    cache = _genre_graft_cache()
    genre_to_idx = cache["genre_to_idx"]
    genre_idx = int(genre_to_idx[target_genre])
    rows = np.flatnonzero(cache["genre_idx"] == genre_idx)
    if rows.size == 0:
        return None
    track_ids = cache["track_ids"][rows]
    uniq, counts = np.unique(track_ids, return_counts=True)
    min_rows = 3 if target_genre != "baroque_classical" else 2
    keep = counts >= min_rows
    if np.any(keep):
        uniq = uniq[keep]
    else:
        uniq = uniq[counts == counts.max()]
    rng = np.random.default_rng(328 + genre_idx * 17 + len(src))
    if uniq.size > 24:
        uniq = rng.choice(uniq, size=24, replace=False)
    meta = cache["meta"]
    n_frames = int(getattr(meta, "n_frames", 432))
    probe_len = max(int(round(5.0 * sr)), sr)
    probe_hop = max(probe_len // 2, sr)
    probes: List[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, min(len(src), probe_hop * 3), probe_hop):
        end = min(len(src), start + probe_len)
        seg = src[start:end]
        if len(seg) < sr:
            continue
        chroma, onset = _cache_like_chunk_features(seg, sr, n_frames)
        chroma_v = chroma.reshape(-1)
        chroma_v = chroma_v / (np.linalg.norm(chroma_v) + 1e-8)
        onset = onset / (float(np.linalg.norm(onset)) + 1e-8)
        probes.append((chroma_v, onset))
    if not probes:
        return None
    chroma_arr = cache["arrays"]["chroma"]
    onset_arr = cache["arrays"]["onset"]
    best_track = None
    best_score = -1e18
    idx_df = cache["index_df"]
    for tid in uniq.tolist():
        tids = np.asarray(cache["track_ids"] == str(tid))
        candidate_rows_full = rows[tids[rows]]
        if candidate_rows_full.size == 0:
            continue
        candidate_rows = candidate_rows_full
        if candidate_rows.size > 32:
            candidate_rows = rng.choice(candidate_rows, size=32, replace=False)
        scores: List[float] = []
        for ridx in candidate_rows.tolist():
            ref_chroma = np.asarray(chroma_arr[int(ridx)], dtype=np.float32).reshape(-1)
            ref_onset = np.asarray(onset_arr[int(ridx)], dtype=np.float32)
            probe_scores = [_graft_match_score(p_chroma, p_onset, ref_chroma, ref_onset) for p_chroma, p_onset in probes]
            scores.append(float(max(probe_scores)))
        path_score = _path_pref_score(str(idx_df.iloc[int(candidate_rows[0])]["path"]), target_genre)
        starts = np.asarray(
            [float(idx_df.iloc[int(ridx)].get("start_sec", 0.0)) for ridx in candidate_rows_full.tolist()],
            dtype=np.float32,
        )
        span_sec = float(starts.max() - starts.min()) if starts.size > 1 else 0.0
        row_count_bonus = 0.14 * min(float(candidate_rows_full.size), 6.0) / 6.0
        span_bonus = 0.10 * min(span_sec, 26.0) / 26.0
        track_score = (
            float(np.mean(sorted(scores, reverse=True)[: min(4, len(scores))]))
            + float(path_score)
            + float(row_count_bonus)
            + float(span_bonus)
        )
        if track_score > best_score:
            best_score = track_score
            best_track = str(tid)
    return best_track


def _track_rows_sorted(target_genre: str, track_id: str) -> np.ndarray:
    cache = _genre_graft_cache()
    genre_idx = int(cache["genre_to_idx"][target_genre])
    rows = np.flatnonzero(cache["genre_idx"] == genre_idx)
    if rows.size == 0:
        return rows
    track_ids = cache["track_ids"]
    rows = rows[track_ids[rows] == str(track_id)]
    if rows.size == 0:
        return rows
    starts = np.asarray(
        [float(cache["index_df"].iloc[int(r)]["start_sec"]) for r in rows.tolist()],
        dtype=np.float32,
    )
    order = np.argsort(starts)
    return rows[order]


def _build_graft_row_schedule(target_genre: str, track_id: str, n_segments: int, hop_sec: float) -> List[int]:
    rows = _track_rows_sorted(target_genre, track_id)
    if rows.size == 0 or n_segments <= 0:
        return []
    cache = _genre_graft_cache()
    starts = np.asarray(
        [float(cache["index_df"].iloc[int(r)]["start_sec"]) for r in rows.tolist()],
        dtype=np.float32,
    )
    schedule: List[int] = []
    last_pos = 0
    repeat_count = 0
    for segment_idx in range(int(n_segments)):
        target_start = float(segment_idx) * float(hop_sec)
        local = starts[last_pos:]
        if local.size == 0:
            pos = int(len(rows) - 1)
        else:
            pos = int(last_pos + int(np.argmin(np.abs(local - target_start))))
        if schedule and pos == last_pos:
            repeat_count += 1
        else:
            repeat_count = 0
        # Allow a donor row to cover at most two overlapped segments before advancing.
        if repeat_count >= 2 and pos < (len(rows) - 1):
            pos += 1
            repeat_count = 0
        if pos > last_pos + 1:
            pos = last_pos + 1
        pos = int(np.clip(pos, last_pos, len(rows) - 1))
        schedule.append(int(rows[pos]))
        last_pos = pos
    return schedule


def _genre_graft_backing(y: np.ndarray, src: np.ndarray, sr: int, target_genre: str, strength: float) -> tuple[np.ndarray, Dict[str, Any]]:
    if strength <= 0.0 or len(y) < sr:
        return y.astype(np.float32, copy=False), {"backing_graft_applied": False}
    n = min(len(y), len(src))
    y = y[:n].astype(np.float32, copy=False)
    src = src[:n].astype(np.float32, copy=False)
    segment_seconds = 5.0
    seg_len = max(int(round(segment_seconds * sr)), sr)
    hop = max(seg_len // 2, sr)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    donor_rows: List[Dict[str, Any]] = []
    s = float(np.clip(strength, 0.0, 1.0))
    fixed_track_id = _pick_genre_graft_track(src, sr, target_genre)
    valid_segments = [(start, min(n, start + seg_len)) for start in range(0, n, hop) if min(n, start + seg_len) - start >= sr]
    hop_sec = float(hop) / float(sr)
    scheduled_rows = _build_graft_row_schedule(target_genre, fixed_track_id, len(valid_segments), hop_sec) if fixed_track_id else []
    cache = _genre_graft_cache()
    chroma_arr = cache["arrays"]["chroma"]
    onset_arr = cache["arrays"]["onset"]
    meta = cache["meta"]
    n_frames = int(getattr(meta, "n_frames", 432))

    for segment_idx, (start, end) in enumerate(valid_segments):
        seg = src[start:end]
        donor: Dict[str, Any]
        if segment_idx < len(scheduled_rows):
            ridx = int(scheduled_rows[segment_idx])
            row = cache["index_df"].iloc[ridx]
            seg_chroma, seg_onset = _cache_like_chunk_features(seg, sr, n_frames)
            seg_chroma_v = seg_chroma.reshape(-1)
            seg_chroma_v = seg_chroma_v / (np.linalg.norm(seg_chroma_v) + 1e-8)
            seg_onset = seg_onset / (float(np.linalg.norm(seg_onset)) + 1e-8)
            ref_chroma = np.asarray(chroma_arr[ridx], dtype=np.float32).reshape(-1)
            ref_onset = np.asarray(onset_arr[ridx], dtype=np.float32)
            score = _graft_match_score(seg_chroma_v, seg_onset, ref_chroma, ref_onset)
            donor = {
                "row_idx": ridx,
                "track_id": str(row.get("track_id", "")),
                "path": Path(str(row["path"])),
                "start_sec": float(row.get("start_sec", 0.0)),
                "score": float(score),
                "chunk_seconds": float(getattr(meta, "chunk_sec", 5.0)),
            }
        else:
            donor = _pick_genre_graft_row(
                seg,
                sr,
                target_genre,
                segment_idx,
                fixed_track_id=fixed_track_id,
                prev_start_sec=float(donor_rows[-1]["start_sec"]) if donor_rows else None,
                expected_step_sec=hop_sec,
            )
        donor_rows.append({k: donor[k] for k in ("row_idx", "track_id", "path", "start_sec", "score")})
        donor_audio = load_audio_chunk(
            path=donor["path"],
            sample_rate=sr,
            seconds=float(donor["chunk_seconds"]),
            start_sec=float(donor["start_sec"]),
        ).astype(np.float32)
        if donor_audio.ndim > 1:
            donor_audio = donor_audio.mean(axis=1)
        if len(donor_audio) < (end - start):
            donor_audio = np.pad(donor_audio, (0, end - start - len(donor_audio)))
        donor_audio = donor_audio[: end - start]
        donor_h, donor_p = librosa.effects.hpss(donor_audio.astype(np.float32, copy=False))
        if target_genre == "baroque_classical":
            donor_mix = 0.62 * donor_h + 0.10 * donor_p
        elif target_genre == "hiphop_xtc":
            donor_mix = 0.18 * donor_h + 0.54 * donor_p
        elif target_genre == "lofi_hh_lfbb":
            donor_mix = 0.24 * donor_h + 0.22 * donor_p
            donor_mix = _apply_spectral_tilt(donor_mix.astype(np.float32), sr, target_genre, 0.8)
        else:
            donor_mix = 0.18 * donor_h + 0.18 * donor_p
        if len(donor_mix) <= 1:
            continue
        win = np.hanning(len(donor_mix)).astype(np.float32)
        if len(win) > 8:
            win = np.maximum(win, 0.05)
        out[start:end] += donor_mix[: end - start] * win[: end - start]
        weight[start:end] += win[: end - start]

    keep = weight > 1e-6
    graft = np.zeros_like(y)
    graft[keep] = out[keep] / weight[keep]
    base = y.astype(np.float32, copy=False)
    mixed = ((1.0 - s) * base + s * graft).astype(np.float32)
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32, copy=False), {
        "backing_graft_applied": True,
        "backing_graft_strength": s,
        "backing_graft_track_id": fixed_track_id,
        "backing_graft_rows": [
            {
                "row_idx": int(item["row_idx"]),
                "track_id": str(item["track_id"]),
                "path": str(item["path"]),
                "start_sec": float(item["start_sec"]),
                "score": float(item["score"]),
            }
            for item in donor_rows[:8]
        ],
        "backing_graft_row_count": int(len(donor_rows)),
    }


def _spectral_envelope_transfer(base: np.ndarray, donor: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if len(base) < 2048 or strength <= 0.0:
        return base.astype(np.float32, copy=False)
    n_fft = 2048
    hop = 512
    base_s = librosa.stft(base.astype(np.float32, copy=False), n_fft=n_fft, hop_length=hop, win_length=n_fft)
    donor_s = librosa.stft(donor.astype(np.float32, copy=False), n_fft=n_fft, hop_length=hop, win_length=n_fft)
    base_mag = np.abs(base_s)
    donor_mag = np.abs(donor_s)
    phase = np.exp(1j * np.angle(base_s))
    base_env = np.mean(base_mag, axis=1)
    donor_env = np.mean(donor_mag, axis=1)
    ratio = (donor_env + 1e-5) / (base_env + 1e-5)
    ratio = signal.medfilt(ratio.astype(np.float32), kernel_size=9)
    ratio = np.clip(ratio, 0.55, 1.85)
    blend = float(np.clip(strength, 0.0, 1.0))
    target_mag = base_mag * (((1.0 - blend) + blend * ratio)[:, None])
    out = librosa.istft(target_mag * phase, hop_length=hop, win_length=n_fft, length=len(base))
    return out.astype(np.float32, copy=False)


def _match_energy_envelope(signal_in: np.ndarray, guide: np.ndarray, strength: float) -> np.ndarray:
    if len(signal_in) < 1024 or strength <= 0.0:
        return signal_in.astype(np.float32, copy=False)
    win = max(129, (len(signal_in) // 64) | 1)
    abs_sig = np.abs(signal_in).astype(np.float32)
    abs_guide = np.abs(guide).astype(np.float32)
    ker = np.ones(win, dtype=np.float32) / float(win)
    env_sig = np.convolve(abs_sig, ker, mode="same")
    env_guide = np.convolve(abs_guide, ker, mode="same")
    ratio = (env_guide + 1e-4) / (env_sig + 1e-4)
    ratio = np.clip(ratio, 0.20, 3.0).astype(np.float32)
    blend = float(np.clip(strength, 0.0, 1.0))
    shaped = signal_in * ((1.0 - blend) + blend * ratio)
    return shaped.astype(np.float32, copy=False)


def _genre_texture_backing(y: np.ndarray, src: np.ndarray, sr: int, target_genre: str, strength: float) -> tuple[np.ndarray, Dict[str, Any]]:
    if strength <= 0.0 or len(y) < sr:
        return y.astype(np.float32, copy=False), {"backing_texture_applied": False}
    n = min(len(y), len(src))
    base = y[:n].astype(np.float32, copy=False)
    src = src[:n].astype(np.float32, copy=False)
    segment_seconds = 5.0
    seg_len = max(int(round(segment_seconds * sr)), sr)
    hop = max(seg_len // 2, sr)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    donor_rows: List[Dict[str, Any]] = []
    s = float(np.clip(strength, 0.0, 1.0))
    fixed_track_id = _pick_genre_graft_track(src, sr, target_genre)
    valid_segments = [(start, min(n, start + seg_len)) for start in range(0, n, hop) if min(n, start + seg_len) - start >= sr]
    scheduled_rows = _build_graft_row_schedule(target_genre, fixed_track_id, len(valid_segments), float(hop) / float(sr)) if fixed_track_id else []
    cache = _genre_graft_cache()
    meta = cache["meta"]

    for segment_idx, (start, end) in enumerate(valid_segments):
        base_seg = base[start:end]
        if segment_idx < len(scheduled_rows):
            ridx = int(scheduled_rows[segment_idx])
            row = cache["index_df"].iloc[ridx]
            donor = {
                "row_idx": ridx,
                "track_id": str(row.get("track_id", "")),
                "path": Path(str(row["path"])),
                "start_sec": float(row.get("start_sec", 0.0)),
                "score": 0.0,
                "chunk_seconds": float(getattr(meta, "chunk_sec", 5.0)),
            }
        else:
            donor = _pick_genre_graft_row(base_seg, sr, target_genre, segment_idx, fixed_track_id=fixed_track_id)
        donor_rows.append({k: donor[k] for k in ("row_idx", "track_id", "path", "start_sec", "score")})
        donor_audio = load_audio_chunk(
            path=donor["path"],
            sample_rate=sr,
            seconds=float(donor["chunk_seconds"]),
            start_sec=float(donor["start_sec"]),
        ).astype(np.float32)
        if donor_audio.ndim > 1:
            donor_audio = donor_audio.mean(axis=1)
        if len(donor_audio) < len(base_seg):
            donor_audio = np.pad(donor_audio, (0, len(base_seg) - len(donor_audio)))
        donor_audio = donor_audio[: len(base_seg)]
        base_h, base_p = librosa.effects.hpss(base_seg.astype(np.float32, copy=False))
        donor_h, donor_p = librosa.effects.hpss(donor_audio.astype(np.float32, copy=False))
        harm_strength = min(1.0, 0.35 + 0.55 * s)
        perc_strength = min(1.0, 0.20 + 0.45 * s)
        tex_h = _spectral_envelope_transfer(base_h, donor_h, sr, harm_strength)
        tex_p = _spectral_envelope_transfer(base_p, donor_p, sr, perc_strength)
        if target_genre == "baroque_classical":
            texture_seg = 0.88 * tex_h + 0.18 * tex_p
            mix_alpha = 0.24 * s
        elif target_genre == "hiphop_xtc":
            texture_seg = 0.72 * tex_h + 0.72 * tex_p
            mix_alpha = 0.34 * s
        elif target_genre == "lofi_hh_lfbb":
            texture_seg = 0.78 * tex_h + 0.36 * tex_p
            texture_seg = _apply_spectral_tilt(texture_seg.astype(np.float32), sr, target_genre, 0.75)
            mix_alpha = 0.32 * s
        else:
            texture_seg = 0.74 * tex_h + 0.42 * tex_p
            mix_alpha = 0.30 * s
        seg_out = ((1.0 - mix_alpha) * base_seg + mix_alpha * texture_seg).astype(np.float32)
        win = np.hanning(len(seg_out)).astype(np.float32)
        if len(win) > 8:
            win = np.maximum(win, 0.05)
        out[start:end] += seg_out * win[: end - start]
        weight[start:end] += win[: end - start]

    keep = weight > 1e-6
    mixed = base.copy()
    mixed[keep] = out[keep] / weight[keep]
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32, copy=False), {
        "backing_texture_applied": True,
        "backing_texture_strength": s,
        "backing_graft_track_id": fixed_track_id,
        "backing_graft_rows": [
            {
                "row_idx": int(item["row_idx"]),
                "track_id": str(item["track_id"]),
                "path": str(item["path"]),
                "start_sec": float(item["start_sec"]),
                "score": float(item["score"]),
            }
            for item in donor_rows[:8]
        ],
        "backing_graft_row_count": int(len(donor_rows)),
    }


def _genre_accent_backing(y: np.ndarray, src: np.ndarray, sr: int, target_genre: str, strength: float) -> tuple[np.ndarray, Dict[str, Any]]:
    if strength <= 0.0 or len(y) < sr:
        return y.astype(np.float32, copy=False), {"backing_accent_applied": False}
    n = min(len(y), len(src))
    base = y[:n].astype(np.float32, copy=False)
    src = src[:n].astype(np.float32, copy=False)
    segment_seconds = 5.0
    seg_len = max(int(round(segment_seconds * sr)), sr)
    hop = max(seg_len // 2, sr)
    out = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    donor_rows: List[Dict[str, Any]] = []
    s = float(np.clip(strength, 0.0, 1.0))
    fixed_track_id = _pick_genre_graft_track(src, sr, target_genre)
    valid_segments = [(start, min(n, start + seg_len)) for start in range(0, n, hop) if min(n, start + seg_len) - start >= sr]
    scheduled_rows = _build_graft_row_schedule(target_genre, fixed_track_id, len(valid_segments), float(hop) / float(sr)) if fixed_track_id else []
    cache = _genre_graft_cache()
    meta = cache["meta"]

    for segment_idx, (start, end) in enumerate(valid_segments):
        base_seg = base[start:end]
        if segment_idx < len(scheduled_rows):
            ridx = int(scheduled_rows[segment_idx])
            row = cache["index_df"].iloc[ridx]
            donor = {
                "row_idx": ridx,
                "track_id": str(row.get("track_id", "")),
                "path": Path(str(row["path"])),
                "start_sec": float(row.get("start_sec", 0.0)),
                "score": 0.0,
                "chunk_seconds": float(getattr(meta, "chunk_sec", 5.0)),
            }
        else:
            donor = _pick_genre_graft_row(base_seg, sr, target_genre, segment_idx, fixed_track_id=fixed_track_id)
        donor_rows.append({k: donor[k] for k in ("row_idx", "track_id", "path", "start_sec", "score")})
        donor_audio = load_audio_chunk(
            path=donor["path"],
            sample_rate=sr,
            seconds=float(donor["chunk_seconds"]),
            start_sec=float(donor["start_sec"]),
        ).astype(np.float32)
        if donor_audio.ndim > 1:
            donor_audio = donor_audio.mean(axis=1)
        if len(donor_audio) < len(base_seg):
            donor_audio = np.pad(donor_audio, (0, len(base_seg) - len(donor_audio)))
        donor_audio = donor_audio[: len(base_seg)]
        base_h, base_p = librosa.effects.hpss(base_seg.astype(np.float32, copy=False))
        donor_h, donor_p = librosa.effects.hpss(donor_audio.astype(np.float32, copy=False))
        tex_h = _spectral_envelope_transfer(base_h, donor_h, sr, min(1.0, 0.55 + 0.35 * s))
        accent_p = _match_energy_envelope(donor_p, base_p, 0.95)
        accent_p = _spectral_envelope_transfer(accent_p, donor_p, sr, 0.35)
        if target_genre == "hiphop_xtc":
            accent = 0.82 * tex_h + 0.32 * accent_p
            alpha = 0.30 * s
        elif target_genre == "lofi_hh_lfbb":
            accent = 0.86 * tex_h + 0.18 * accent_p
            accent = _apply_spectral_tilt(accent.astype(np.float32), sr, target_genre, 0.85)
            alpha = 0.28 * s
        elif target_genre == "cc0_other":
            accent = 0.78 * tex_h + 0.22 * accent_p
            alpha = 0.27 * s
        else:
            accent = 0.88 * tex_h + 0.12 * accent_p
            alpha = 0.22 * s
        seg_out = ((1.0 - alpha) * base_seg + alpha * accent).astype(np.float32)
        win = np.hanning(len(seg_out)).astype(np.float32)
        if len(win) > 8:
            win = np.maximum(win, 0.05)
        out[start:end] += seg_out * win[: end - start]
        weight[start:end] += win[: end - start]

    keep = weight > 1e-6
    mixed = base.copy()
    mixed[keep] = out[keep] / weight[keep]
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32, copy=False), {
        "backing_accent_applied": True,
        "backing_accent_strength": s,
        "backing_graft_track_id": fixed_track_id,
        "backing_graft_rows": [
            {
                "row_idx": int(item["row_idx"]),
                "track_id": str(item["track_id"]),
                "path": str(item["path"]),
                "start_sec": float(item["start_sec"]),
                "score": float(item["score"]),
            }
            for item in donor_rows[:8]
        ],
        "backing_graft_row_count": int(len(donor_rows)),
    }


def _smooth_harmonic_warble(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    if strength <= 0.0 or len(y) < 4096:
        return y.astype(np.float32, copy=False)
    n_fft = 2048
    hop = 512
    harm, perc = librosa.effects.hpss(y.astype(np.float32, copy=False))
    S = librosa.stft(harm, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    mag = np.abs(S)
    phase = np.exp(1j * np.angle(S))
    width = max(9, int(9 + round(12 * strength)))
    if width % 2 == 0:
        width += 1
    smoothed = signal.medfilt2d(mag, kernel_size=[1, width])
    blend = np.clip(float(strength), 0.0, 0.95)
    mag_mix = ((1.0 - blend) * mag + blend * smoothed).astype(np.float32)
    harm_out = librosa.istft(mag_mix * phase, hop_length=hop, win_length=n_fft, length=len(y))
    mixed = (0.92 * harm_out + perc).astype(np.float32)
    peak = float(np.max(np.abs(mixed))) + 1e-8
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32, copy=False)


def _apply_spectral_tilt(y: np.ndarray, sr: int, target_genre: str, strength: float) -> np.ndarray:
    if strength <= 0.0 or len(y) < 4096:
        return y.astype(np.float32, copy=False)
    n_fft = 2048
    hop = 512
    S = librosa.stft(y.astype(np.float32, copy=False), n_fft=n_fft, hop_length=hop, win_length=n_fft)
    mag = np.abs(S)
    phase = np.exp(1j * np.angle(S))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    curve = np.ones_like(freqs, dtype=np.float32)
    s = float(strength)
    if target_genre == "baroque_classical":
        curve *= np.where(freqs < 110.0, 1.0 - 0.45 * s, 1.0)
        curve *= np.where((freqs >= 260.0) & (freqs <= 3200.0), 1.0 + 0.55 * s, 1.0)
        curve *= np.where(freqs >= 4200.0, 1.0 + 0.16 * s, 1.0)
    elif target_genre == "hiphop_xtc":
        curve *= np.where((freqs >= 50.0) & (freqs <= 180.0), 1.0 + 0.70 * s, 1.0)
        curve *= np.where((freqs >= 1800.0) & (freqs <= 5200.0), 1.0 + 0.18 * s, 1.0)
        curve *= np.where((freqs >= 280.0) & (freqs <= 700.0), 1.0 - 0.12 * s, 1.0)
    elif target_genre == "lofi_hh_lfbb":
        curve *= np.where(freqs >= 4500.0, 1.0 - 0.60 * s, 1.0)
        curve *= np.where((freqs >= 180.0) & (freqs <= 1500.0), 1.0 + 0.22 * s, 1.0)
        curve *= np.where(freqs < 60.0, 1.0 - 0.18 * s, 1.0)
    elif target_genre == "cc0_other":
        curve *= np.where((freqs >= 90.0) & (freqs <= 2400.0), 1.0 + 0.08 * s, 1.0)
    out = librosa.istft((mag * curve[:, None]) * phase, hop_length=hop, win_length=n_fft, length=len(y))
    peak = float(np.max(np.abs(out))) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False)


def _target_backing_finish(y: np.ndarray, sr: int, target_genre: str, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return y.astype(np.float32, copy=False)
    harm, perc = librosa.effects.hpss(y.astype(np.float32, copy=False))
    s = float(strength)
    if target_genre == "baroque_classical":
        delay_a = int(round(0.045 * sr))
        delay_b = int(round(0.093 * sr))
        tail = np.zeros_like(harm)
        if delay_a < len(harm):
            tail[delay_a:] += 0.18 * s * harm[:-delay_a]
        if delay_b < len(harm):
            tail[delay_b:] += 0.10 * s * harm[:-delay_b]
        out = harm + tail + (0.78 - 0.08 * s) * perc
    elif target_genre == "hiphop_xtc":
        out = harm + (1.08 + 0.18 * s) * perc
        out = np.tanh((1.0 + 0.12 * s) * out)
    elif target_genre == "lofi_hh_lfbb":
        out = harm + (0.82 - 0.08 * s) * perc
        out = _apply_spectral_tilt(out.astype(np.float32), sr, target_genre, min(1.0, 0.45 + 0.55 * s))
    else:
        out = harm + perc
    peak = float(np.max(np.abs(out))) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False)


def _postprocess_backing(
    y: np.ndarray,
    src: np.ndarray,
    sr: int,
    target_genre: str,
    mode: str,
    tone_strength: float,
    dewarble_strength: float,
) -> tuple[np.ndarray, Dict[str, Any]]:
    out = y.astype(np.float32, copy=False)
    meta: Dict[str, Any] = {
        "backing_post_mode": mode,
        "backing_post_strength": float(tone_strength),
        "backing_dewarble_strength": float(dewarble_strength),
    }
    if dewarble_strength > 0.0:
        out = _smooth_harmonic_warble(out, sr, dewarble_strength)
        meta["backing_dewarble_applied"] = True
    if mode == "genre_separate":
        out = _apply_spectral_tilt(out, sr, target_genre, tone_strength)
        out = _target_backing_finish(out, sr, target_genre, tone_strength)
        meta["backing_target_genre"] = target_genre
        meta["backing_post_applied"] = True
    elif mode == "genre_texture":
        out, texture_meta = _genre_texture_backing(out, src, sr, target_genre, tone_strength)
        out = _target_backing_finish(out, sr, target_genre, min(1.0, 0.25 + 0.35 * tone_strength))
        meta.update(texture_meta)
        meta["backing_target_genre"] = target_genre
        meta["backing_post_applied"] = True
    elif mode == "genre_accent":
        out, accent_meta = _genre_accent_backing(out, src, sr, target_genre, tone_strength)
        out = _target_backing_finish(out, sr, target_genre, min(1.0, 0.28 + 0.38 * tone_strength))
        meta.update(accent_meta)
        meta["backing_target_genre"] = target_genre
        meta["backing_post_applied"] = True
    elif mode == "genre_graft":
        out, graft_meta = _genre_graft_backing(out, src, sr, target_genre, tone_strength)
        out = _target_backing_finish(out, sr, target_genre, min(1.0, 0.35 + 0.45 * tone_strength))
        meta.update(graft_meta)
        meta["backing_target_genre"] = target_genre
        meta["backing_post_applied"] = True
    peak = float(np.max(np.abs(out))) + 1e-8
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32, copy=False), meta


def _debleed_vocals_with_source_accomp(
    vocals: np.ndarray,
    src_accomp: np.ndarray,
    sr: int,
    strength: float,
    floor: float = 0.18,
) -> np.ndarray:
    n = min(len(vocals), len(src_accomp))
    if n < max(sr // 2, 4096):
        return vocals[:n].astype(np.float32, copy=False)
    vocals = vocals[:n].astype(np.float32, copy=False)
    src_accomp = src_accomp[:n].astype(np.float32, copy=False)
    n_fft = 2048
    hop = 512
    V = librosa.stft(vocals, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    A = librosa.stft(src_accomp, n_fft=n_fft, hop_length=hop, win_length=n_fft)
    vmag = np.abs(V)
    amag = np.abs(A)
    mask = np.clip((vmag - strength * amag) / (vmag + 1e-8), floor, 1.0).astype(np.float32)
    cleaned = librosa.istft(V * mask, hop_length=hop, win_length=n_fft, length=n)
    peak = float(np.max(np.abs(cleaned))) + 1e-8
    if peak > 1.0:
        cleaned = cleaned / peak
    return cleaned.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stronger accompaniment-style hybrid compare.")
    parser.add_argument("--out-dir", type=str, default="", help="Existing or desired output directory.")
    parser.add_argument("--settings-filter", type=str, default="", help="Comma-separated setting labels to run.")
    parser.add_argument("--song-filter", type=str, default="", help="Case-insensitive substring filter on source filename.")
    args = parser.parse_args()

    cfg = HybridPushConfig()
    if args.out_dir.strip():
        out_root = Path(args.out_dir)
    else:
        out_root = cfg.output_root / f"hybrid_push_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    songs = picked_songs()
    settings = settings_panel()
    if args.song_filter.strip():
        pat = args.song_filter.strip().lower()
        songs = [row for row in songs if pat in Path(row["path"]).name.lower()]
        if not songs:
            raise RuntimeError(f"No songs matched --song-filter={args.song_filter!r}")
    if args.settings_filter.strip():
        wanted = {part.strip() for part in args.settings_filter.split(",") if part.strip()}
        settings = [row for row in settings if row["label"] in wanted]
        if not settings:
            raise RuntimeError(f"No settings matched --settings-filter={args.settings_filter!r}")
    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([{"path": str(row["path"]), "source_genre": row["source_genre"]} for row in songs], indent=2), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    job_idx = 0
    for setting in settings:
        for song in songs:
            stems = _resolve_stems(cfg, song)
            for target_genre in TARGET_GENRES:
                job_tag = f"{job_idx:03d}_{_slug(Path(song['path']).stem)[:52]}__to__{_slug(target_genre)}"
                out_dir = out_root / "clips" / setting["label"] / job_tag
                _run_longform(cfg, setting, stems["accompaniment"], song["source_genre"], target_genre, out_dir, cfg.seed + job_idx)
                final_mix = _make_mix(setting, stems, out_dir)
                manifest_rows.append(
                    {
                        "job_idx": job_idx,
                        "setting_label": setting["label"],
                        "source_audio": str(song["path"]),
                        "target_genre": target_genre,
                        "output_dir": str(out_dir),
                        "generated_wav": str(out_dir / "longform_coherent.wav"),
                        "final_mix_wav": str(final_mix),
                    }
                )
                with (out_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(manifest_rows)
                job_idx += 1

    summary = {
        "output_dir": str(out_root),
        "n_songs": len(songs),
        "n_settings": len(settings),
        "target_genres": TARGET_GENRES,
        "total_jobs": len(manifest_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
