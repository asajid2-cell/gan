from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import librosa.display
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
GUI_ROOT = REPO_ROOT / "gui"
ASSET_ROOT = GUI_ROOT / "assets"
OUTPUT_ROOT = Path(
    os.environ.get("GENERATION_LAB_OUTPUT_ROOT")
    or os.environ.get("DGGR_OUTPUT_ROOT")
    or str(GUI_ROOT / "outputs")
).expanduser()

for path in [REPO_ROOT, REPO_ROOT / "lab 3", REPO_ROOT / "lab 4"]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from dggr.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames
from dggr.lab3_codec_data import CodecCacheMeta
from dggr.lab3_codec_judge import CodecStyleJudge, Lab1StyleProbe, MERTStyleProbe
from dggr.lab3_codec_models import CodecLatentTranslator, MultiScaleWaveDiscriminator
from dggr.lab3_codec_train import build_style_centroid_bank, build_style_exemplar_bank
from dggr.lab3_diffusion_data import (
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from dggr.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from dggr.lab3_diffusion_train import (
    ddim_sample_v2,
    load_checkpoint,
    pitch_correlation,
    vocode_bigvgan,
)


def ensure_output_root() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT


def _utc_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(obj: Dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)
    return path


def _load_codec_cache_light(cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int], CodecCacheMeta]:
    cache_dir = Path(cache_dir)
    idx_path = cache_dir / "codec_cache_index.csv"
    npz_path = cache_dir / "codec_cache_arrays.npz"
    gmap_path = cache_dir / "codec_genre_to_idx.json"
    meta_path = cache_dir / "codec_meta.json"
    if not idx_path.exists() or not npz_path.exists() or not gmap_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Codec cache files missing in {cache_dir}")
    index_df = pd.read_csv(idx_path)
    z = np.load(npz_path)
    arrays = {
        "z_content": z["z_content"].astype(np.float32),
        "z_style": z["z_style"].astype(np.float32),
        "genre_idx": z["genre_idx"].astype(np.int64),
    }
    if "mert_feat" in z:
        arrays["mert_feat"] = z["mert_feat"].astype(np.float32)
    with gmap_path.open("r", encoding="utf-8") as f:
        genre_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    with meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = CodecCacheMeta(
        codec_model_id=str(raw["codec_model_id"]),
        codec_sample_rate=int(raw["codec_sample_rate"]),
        codec_chunk_seconds=float(raw["codec_chunk_seconds"]),
        codec_bandwidth=float(raw["codec_bandwidth"]),
        codec_frames=int(raw["codec_frames"]),
        codec_channels=int(raw["codec_channels"]),
        lab1_n_frames=int(raw["lab1_n_frames"]),
    )
    return index_df, arrays, genre_to_idx, meta


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return str(device)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb)) + 1e-8
    return float(np.dot(aa, bb) / denom)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _norm_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)


def _load_audio_file(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(
        str(path),
        sr=target_sr,
        mono=True,
        dtype=np.float32,
        res_type="soxr_hq",
    )
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = audio / max(1.0, peak)
    return audio.astype(np.float32), int(target_sr)


def _resolve_repo_relative_path(raw_path: object, fallback: Path) -> Path:
    fallback = Path(fallback)
    if raw_path:
        text = str(raw_path).strip()
        if text:
            direct = Path(text)
            if direct.exists():
                return direct
            parts = [part for part in text.replace("\\", "/").split("/") if part and part not in (".",)]
            for anchor in ("saves", "saves2", "examples", "lab 3", "lab 4", "gui", "dggr"):
                if anchor in parts:
                    idx = parts.index(anchor)
                    candidate = REPO_ROOT.joinpath(*parts[idx:])
                    if candidate.exists():
                        return candidate
    return fallback


def _save_audio(path: Path, audio: np.ndarray, sr: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sr))
    return path


def _audio_duration(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


def get_audio_info(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {
            "ok": False,
            "message": "No audio selected.",
        }
    p = Path(path)
    if not p.exists():
        return {
            "ok": False,
            "message": f"Missing file: {p}",
        }
    y, sr = librosa.load(str(p), sr=None, mono=True, dtype=np.float32)
    dur = float(len(y)) / float(sr) if sr else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return {
        "ok": True,
        "path": str(p),
        "sample_rate": int(sr),
        "seconds": round(dur, 3),
        "samples": int(len(y)),
        "peak_abs": round(peak, 4),
        "channels": 1,
    }


def _audio_to_temp_copy(path: str) -> Path:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_output_root()
    tmp_dir = OUTPUT_ROOT / "tmp_inputs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / f"{_utc_tag()}_{src.name}"
    shutil.copy2(src, dst)
    return dst


def _output_run_dir(mode: str) -> Path:
    out = ensure_output_root() / mode / _utc_tag()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _zip_bundle(bundle_dir: Path) -> Path:
    zip_path = bundle_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(bundle_dir)))
    return zip_path


def _markdown_table(rows: Sequence[Tuple[str, object]]) -> str:
    lines = ["| Item | Value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _seconds_label(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remain = seconds - minutes * 60
    return f"{minutes}m {remain:.1f}s"


def _load_example_sources() -> List[str]:
    candidates: List[Path] = []
    codec_samples = REPO_ROOT / "saves2" / "lab3_codec_transfer" / "run1055" / "samples" / "posttrain_samples"
    if codec_samples.exists():
        candidates.extend(sorted(codec_samples.glob("*_source.wav"))[:6])

    longform_root = REPO_ROOT / "saves2" / "lab4_longform_coherence" / "smoke"
    if longform_root.exists():
        candidates.append(longform_root / "source.wav")

    curated = REPO_ROOT / "examples" / "audio"
    if curated.exists():
        candidates.extend(sorted(curated.glob("*.wav")))

    seen = set()
    out: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        text = str(path)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


@dataclass
class CodecRunRecord:
    run_name: str
    run_dir: Path
    checkpoint: Path
    checkpoints: List[Path]
    cache_dir: Path
    run_state: Dict[str, object]
    genre_to_idx: Dict[str, int]
    metrics: Dict[str, object]

    @property
    def genres(self) -> List[str]:
        return sorted(self.genre_to_idx.keys())


@dataclass
class DiffusionRunRecord:
    run_name: str
    run_dir: Path
    checkpoint: Path
    checkpoints: List[Path]
    cache_dir: Path
    config: Dict[str, object]
    genre_to_idx: Dict[str, int]

    @property
    def genres(self) -> List[str]:
        return sorted(self.genre_to_idx.keys())


def _choose_checkpoint(run_dir: Path, preferred: Sequence[str]) -> Optional[Path]:
    for rel in preferred:
        candidate = run_dir / rel
        if candidate.exists():
            return candidate
    return None


def _choose_diffusion_checkpoint(run_dir: Path) -> Optional[Path]:
    if run_dir.name == "run_d002":
        preferred = [
            "checkpoints/epoch_006.pt",  # documented best subjective checkpoint
            "checkpoints/best.pt",
            "checkpoints/latest.pt",
            "checkpoints/epoch_018.pt",
        ]
    else:
        preferred = [
            "checkpoints/best.pt",
            "checkpoints/latest.pt",
            "checkpoints/epoch_006.pt",
        ]
    return _choose_checkpoint(run_dir, preferred)


def _codec_checkpoints_for_run(run_dir: Path) -> List[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    preferred = ["stage3_latest.pt", "stage2_latest.pt", "stage1_latest.pt"]
    found: List[Path] = []
    seen = set()
    for name in preferred:
        path = ckpt_dir / name
        if path.exists():
            found.append(path)
            seen.add(str(path))
    for path in sorted(ckpt_dir.glob("*.pt")):
        text = str(path)
        if text not in seen:
            found.append(path)
            seen.add(text)
    return found


def _diffusion_checkpoints_for_run(run_dir: Path) -> List[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    if run_dir.name == "run_d002":
        preferred = ["epoch_006.pt", "best.pt", "latest.pt"]
    else:
        preferred = ["best.pt", "latest.pt", "epoch_006.pt"]
    found: List[Path] = []
    seen = set()
    for name in preferred:
        path = ckpt_dir / name
        if path.exists():
            found.append(path)
            seen.add(str(path))
    for path in sorted(ckpt_dir.glob("*.pt")):
        text = str(path)
        if text not in seen:
            found.append(path)
            seen.add(text)
    return found


def discover_codec_runs() -> List[CodecRunRecord]:
    out: List[CodecRunRecord] = []
    roots = [
        REPO_ROOT / "saves2" / "lab3_codec_transfer",
        REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs",
    ]
    run_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.name == "overnight_runs":
            for tag_dir in root.iterdir():
                if tag_dir.is_dir():
                    run_dirs.extend([p for p in tag_dir.iterdir() if p.is_dir() and p.name.startswith("codec")])
        else:
            run_dirs.extend([p for p in root.iterdir() if p.is_dir()])
    for run_dir in sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        run_state = _safe_json(run_dir / "run_state.json", {})
        cache_dir = run_dir / "cache"
        checkpoints = _codec_checkpoints_for_run(run_dir)
        checkpoint = checkpoints[0] if checkpoints else None
        if not cache_dir.exists() or checkpoint is None:
            continue
        genre_to_idx = _safe_json(cache_dir / "codec_genre_to_idx.json", {})
        metrics = _safe_json(run_dir / "codec_gate_eval.json", {})
        out.append(
            CodecRunRecord(
                run_name=run_dir.name,
                run_dir=run_dir,
                checkpoint=checkpoint,
                checkpoints=checkpoints,
                cache_dir=cache_dir,
                run_state=run_state,
                genre_to_idx={str(k): int(v) for k, v in genre_to_idx.items()},
                metrics=metrics,
            )
        )
    return out


def discover_diffusion_runs() -> List[DiffusionRunRecord]:
    out: List[DiffusionRunRecord] = []
    roots = [
        REPO_ROOT / "saves2" / "lab3_diffusion",
        REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs",
    ]
    run_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.name == "overnight_runs":
            for tag_dir in root.iterdir():
                if tag_dir.is_dir():
                    run_dirs.extend([p for p in tag_dir.iterdir() if p.is_dir() and p.name.startswith("diffusion")])
        else:
            run_dirs.extend([p for p in root.iterdir() if p.is_dir()])
    for run_dir in sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True):
        config = _safe_json(run_dir / "v2_config.json", _safe_json(run_dir / "diffusion_config.json", {}))
        checkpoints = _diffusion_checkpoints_for_run(run_dir)
        checkpoint = checkpoints[0] if checkpoints else None
        if checkpoint is None:
            continue
        cache_dir = _resolve_repo_relative_path(
            config.get("cache_dir"),
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache",
        )
        if not cache_dir.exists():
            continue
        genre_to_idx = _safe_json(cache_dir / "diff_genre_to_idx.json", {})
        out.append(
            DiffusionRunRecord(
                run_name=run_dir.name,
                run_dir=run_dir,
                checkpoint=checkpoint,
                checkpoints=checkpoints,
                cache_dir=cache_dir,
                config=config,
                genre_to_idx={str(k): int(v) for k, v in genre_to_idx.items()},
            )
        )
    return out


def catalog_snapshot() -> Dict[str, object]:
    codec_runs = discover_codec_runs()
    diffusion_runs = discover_diffusion_runs()
    examples = _load_example_sources()
    codec_default = codec_runs[0].run_name if codec_runs else None
    diffusion_default = diffusion_runs[0].run_name if diffusion_runs else None
    return {
        "codec_runs": [r.run_name for r in codec_runs],
        "diffusion_runs": [r.run_name for r in diffusion_runs],
        "codec_default": codec_default,
        "diffusion_default": diffusion_default,
        "codec_default_checkpoint": next((str(r.checkpoint) for r in codec_runs if r.run_name == codec_default), None),
        "diffusion_default_checkpoint": next((str(r.checkpoint) for r in diffusion_runs if r.run_name == diffusion_default), None),
        "example_audio": examples,
    }


def codec_runs_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in discover_codec_runs():
        cfg = rec.run_state.get("config", {}) if isinstance(rec.run_state.get("config", {}), dict) else {}
        rows.append(
            {
                "run": rec.run_name,
                "checkpoint": rec.checkpoint.name,
                "n_checkpoints": len(rec.checkpoints),
                "style_cond_source": cfg.get("style_cond_source", ""),
                "translator_direct_output": bool(cfg.get("translator_direct_output", False)),
                "genres": ", ".join(rec.genres),
                "gate_keys": ", ".join(sorted(rec.metrics.keys())),
            }
        )
    return pd.DataFrame(rows)


def diffusion_runs_table() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for rec in discover_diffusion_runs():
        rows.append(
            {
                "run": rec.run_name,
                "checkpoint": rec.checkpoint.name,
                "n_checkpoints": len(rec.checkpoints),
                "cache_dir": str(rec.cache_dir),
                "guidance_default": rec.config.get("guidance_scale", 2.0),
                "genres": ", ".join(rec.genres),
            }
        )
    return pd.DataFrame(rows)


def _real_music_cache_dir() -> Path:
    return REPO_ROOT / "saves2" / "real_music_transfer" / "spotify_discovered_genres_cache"


def _real_music_run_dir() -> Path:
    return REPO_ROOT / "saves2" / "real_music_transfer" / "runs" / "real_transfer_20260512_182741"


def real_music_checkpoint_choices() -> Tuple[List[Tuple[str, str]], Optional[str]]:
    ckpt_dir = _real_music_run_dir() / "checkpoints"
    if not ckpt_dir.exists():
        return [], None
    preferred = ["best_by_val.pt", "latest.pt"]
    paths: List[Path] = []
    for name in preferred:
        p = ckpt_dir / name
        if p.exists():
            paths.append(p)
    paths.extend(sorted([p for p in ckpt_dir.glob("epoch_*.pt") if p not in paths], key=lambda p: p.name, reverse=True))
    choices = [(p.name, str(p)) for p in paths]
    return choices, (choices[0][1] if choices else None)


def real_music_genres() -> List[str]:
    gmap = _real_music_cache_dir() / "diff_genre_to_idx.json"
    if not gmap.exists():
        return []
    raw = _safe_json(gmap, {})
    return sorted(str(k) for k in raw.keys())


def real_music_runs_table() -> pd.DataFrame:
    choices, default = real_music_checkpoint_choices()
    return pd.DataFrame(
        [
            {
                "run": _real_music_run_dir().name,
                "default_checkpoint": Path(default).name if default else "",
                "n_checkpoints": len(choices),
                "cache_dir": str(_real_music_cache_dir()),
                "n_targets": len(real_music_genres()),
            }
        ]
    )


def _plot_spectrogram(ax, audio: np.ndarray, sr: int, title: str) -> None:
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("")


def _plot_wave(ax, audio: np.ndarray, sr: int, title: str) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    max_points = 5000
    step = 1
    if len(audio) > max_points:
        step = int(math.ceil(len(audio) / max_points))
        audio = audio[::step]
    t = (np.arange(len(audio), dtype=np.float32) * float(step)) / float(sr)
    ax.plot(t, audio, linewidth=0.8, color="#0f766e")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(0, max(1e-3, float(len(audio)) / float(sr)))
    ax.set_xlabel("seconds")
    ax.set_ylabel("amp")
    ax.grid(alpha=0.2)


def _clip_for_preview(audio: np.ndarray, sr: int, max_seconds: float = 15.0) -> np.ndarray:
    limit = int(max(1, round(float(sr) * float(max_seconds))))
    if len(audio) <= limit:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio[:limit], dtype=np.float32)


def _plot_audio_pair(
    source_audio: np.ndarray,
    gen_audio: np.ndarray,
    *,
    sr: int,
    title: str,
    out_path: Path,
    max_preview_seconds: float = 15.0,
) -> Path:
    source_audio = _clip_for_preview(source_audio, sr, max_seconds=max_preview_seconds)
    gen_audio = _clip_for_preview(gen_audio, sr, max_seconds=max_preview_seconds)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=16, y=0.98)
    _plot_wave(axes[0, 0], source_audio, sr, "Source waveform")
    _plot_wave(axes[0, 1], gen_audio, sr, "Generated waveform")
    _plot_spectrogram(axes[1, 0], source_audio, sr, "Source mel")
    _plot_spectrogram(axes[1, 1], gen_audio, sr, "Generated mel")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_audio_triptych(
    source_audio: np.ndarray,
    codec_audio: np.ndarray,
    diffusion_audio: np.ndarray,
    *,
    sr_codec: int,
    sr_diffusion: int,
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("DGGR compare view", fontsize=16, y=0.98)
    _plot_wave(axes[0, 0], source_audio, sr_codec, "Source waveform")
    _plot_spectrogram(axes[0, 1], source_audio, sr_codec, "Source mel")
    _plot_wave(axes[1, 0], codec_audio, sr_codec, "Codec waveform")
    _plot_spectrogram(axes[1, 1], codec_audio, sr_codec, "Codec mel")
    _plot_wave(axes[2, 0], diffusion_audio, sr_diffusion, "Diffusion waveform")
    _plot_spectrogram(axes[2, 1], diffusion_audio, sr_diffusion, "Diffusion mel")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _overlap_fade(overlap_samples: int) -> np.ndarray:
    t = np.linspace(0.0, np.pi / 2.0, max(1, int(overlap_samples)), dtype=np.float32)
    return np.cos(t).astype(np.float32) ** 2


def _split_audio_overlapping(audio: np.ndarray, *, chunk_seconds: float, overlap_seconds: float, sr: int) -> List[np.ndarray]:
    chunk_samples = int(round(float(chunk_seconds) * float(sr)))
    overlap_samples = int(round(float(overlap_seconds) * float(sr)))
    hop_samples = max(1, chunk_samples - overlap_samples)
    chunks: List[np.ndarray] = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = np.asarray(audio[pos:end], dtype=np.float32)
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk.astype(np.float32))
        if end >= len(audio):
            break
        pos += hop_samples
    if not chunks:
        chunks.append(np.zeros(max(1, chunk_samples), dtype=np.float32))
    return chunks


def _assemble_audio_crossfade(chunks: Sequence[np.ndarray], *, overlap_seconds: float, sr: int) -> np.ndarray:
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    if len(chunks) == 1:
        return np.asarray(chunks[0], dtype=np.float32)
    overlap_samples = max(1, int(round(float(overlap_seconds) * float(sr))))
    fade = _overlap_fade(overlap_samples)
    out = np.asarray(chunks[0], dtype=np.float32).copy()
    for chunk in chunks[1:]:
        cur = np.asarray(chunk, dtype=np.float32)
        real_ov = min(overlap_samples, len(out), len(cur))
        if real_ov > 0:
            f = fade[:real_ov]
            out[-real_ov:] = out[-real_ov:] * f + cur[:real_ov] * (1.0 - f)
            out = np.concatenate([out, cur[real_ov:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    return out.astype(np.float32)


def _boundary_discontinuities(audio: np.ndarray, *, chunk_seconds: float, overlap_seconds: float, sr: int, window_ms: float = 50.0) -> List[float]:
    hop_samples = int(round((float(chunk_seconds) - float(overlap_seconds)) * float(sr)))
    if hop_samples <= 0:
        return []
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=np.asarray(audio, dtype=np.float32), sr=sr, n_mels=80, hop_length=256),
        ref=np.max,
    )
    boundaries = np.arange(hop_samples, len(audio), hop_samples, dtype=np.int64)
    window_frames = max(1, int(round(float(window_ms) / 1000.0 * float(sr) / 256.0)))
    vals: List[float] = []
    for boundary in boundaries:
        frame = int(boundary // 256)
        if frame - window_frames < 0 or frame + window_frames >= mel_db.shape[1]:
            continue
        left = mel_db[:, frame - window_frames:frame].mean(axis=1)
        right = mel_db[:, frame:frame + window_frames].mean(axis=1)
        vals.append(float(np.mean(np.abs(left - right))))
    return vals


class BigVGANService:
    def __init__(self) -> None:
        self._model = None
        self._device_name = None

    def get(self, device: torch.device):
        device_name = str(device)
        if self._model is not None and self._device_name == device_name:
            return self._model
        import bigvgan as bvg

        model = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
        model.remove_weight_norm()
        model.eval().to(device)
        self._model = model
        self._device_name = device_name
        return model

    def clear(self) -> None:
        self._model = None
        self._device_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SessionCache:
    def __init__(self) -> None:
        self.codec_sessions: Dict[Tuple[str, str], "CodecInferenceSession"] = {}
        self.diffusion_sessions: Dict[Tuple[str, str], "DiffusionInferenceSession"] = {}
        self.bigvgan = BigVGANService()

    def clear(self) -> None:
        self.codec_sessions.clear()
        self.diffusion_sessions.clear()
        self.bigvgan.clear()
        self.bigvgan = BigVGANService()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_codec_sessions(self) -> None:
        self.codec_sessions.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_diffusion_sessions(self) -> None:
        self.diffusion_sessions.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def clear_bigvgan(self) -> None:
        self.bigvgan.clear()
        self.bigvgan = BigVGANService()


SESSION_CACHE = SessionCache()
_TERMINAL_LOCK = Lock()
_TERMINAL_LINES: List[str] = []


def clear_terminal_log() -> str:
    with _TERMINAL_LOCK:
        _TERMINAL_LINES.clear()
    return get_terminal_log()


def append_terminal_log(message: str) -> str:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    with _TERMINAL_LOCK:
        _TERMINAL_LINES.append(line)
        if len(_TERMINAL_LINES) > 300:
            del _TERMINAL_LINES[:-300]
    return get_terminal_log()


def get_terminal_log() -> str:
    with _TERMINAL_LOCK:
        if not _TERMINAL_LINES:
            return "[terminal ready]"
        return "\n".join(_TERMINAL_LINES)


def _apply_cpu_safe_longform_profile(
    *,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
    guidance_scale: float,
    ddim_steps: int,
    assemble_domain: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, float, float, float, int, str]:
    logger = log_callback or (lambda _msg: None)
    resolved = _resolve_device(device)
    actual_device = str(resolved)
    if resolved.type == "cuda":
        return actual_device, chunk_seconds, overlap_seconds, guidance_scale, ddim_steps, assemble_domain

    adjusted_chunk = float(chunk_seconds)
    adjusted_overlap = float(overlap_seconds)
    adjusted_guidance = float(guidance_scale)
    adjusted_steps = int(ddim_steps)
    adjusted_domain = str(assemble_domain)
    changes: List[str] = []

    if adjusted_chunk > 2.0:
        adjusted_chunk = 2.0
        changes.append("chunk_seconds=2.0")
    if adjusted_overlap > 0.35:
        adjusted_overlap = 0.35
        changes.append("overlap_seconds=0.35")
    if adjusted_guidance > 1.25:
        adjusted_guidance = 1.25
        changes.append("guidance_scale=1.25")
    if adjusted_steps > 24:
        adjusted_steps = 24
        changes.append("ddim_steps=24")
    if adjusted_domain.lower() != "wave":
        adjusted_domain = "wave"
        changes.append("assemble_domain=wave")

    if changes:
        logger(
            "CPU safety profile enabled for long-form diffusion: "
            + ", ".join(changes)
            + "."
        )
    else:
        logger("CPU safety profile confirmed: current long-form settings are already memory-safe.")

    return actual_device, adjusted_chunk, adjusted_overlap, adjusted_guidance, adjusted_steps, adjusted_domain


def _load_probe_from_file(path: Path, model_cls):
    if not path.exists():
        return None
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    model = model_cls(**config)
    state_key = "model" if "model" in payload else "state_dict"
    model.load_state_dict(payload[state_key], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _build_centroids_from_embeddings(embeddings: np.ndarray, labels: np.ndarray, n_genres: int) -> np.ndarray:
    return build_style_centroid_bank(embeddings, labels, n_genres=n_genres).cpu().numpy().astype(np.float32)


def _sample_style_vector(
    centroids: np.ndarray,
    exemplars: Optional[Dict[int, torch.Tensor]],
    target_idx: int,
    mode: str,
    alpha: float = 0.5,
) -> np.ndarray:
    centroid = np.asarray(centroids[int(target_idx)], dtype=np.float32)
    mode = str(mode).strip().lower()
    if exemplars is None or int(target_idx) not in exemplars or exemplars[int(target_idx)].shape[0] == 0:
        return _norm_vec(centroid)
    exemplar_bank = exemplars[int(target_idx)]
    sample_idx = int(torch.randint(0, int(exemplar_bank.shape[0]), (1,)).item())
    exemplar = exemplar_bank[sample_idx].detach().cpu().numpy().astype(np.float32)
    if mode == "centroid":
        mixed = centroid
    elif mode == "exemplar":
        mixed = exemplar
    else:
        mixed = float(alpha) * centroid + (1.0 - float(alpha)) * exemplar
    return _norm_vec(mixed)


def _nearest_centroid_label(vector: np.ndarray, centroids: np.ndarray, idx_to_genre: Dict[int, str]) -> Tuple[str, float]:
    sims = _norm_rows(centroids) @ _norm_vec(vector)
    best = int(np.argmax(sims))
    return idx_to_genre[best], float(sims[best])


class CodecInferenceSession:
    def __init__(self, record: CodecRunRecord, device: str = "auto") -> None:
        self.record = record
        self.device = _resolve_device(device)
        self.run_state = record.run_state
        self.config = record.run_state.get("config", {}) if isinstance(record.run_state.get("config", {}), dict) else {}
        self.index_df, self.arrays, self.genre_to_idx, self.meta = _load_codec_cache_light(record.cache_dir)
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
        self.n_genres = len(self.genre_to_idx)

        lab1_ckpt = _resolve_repo_relative_path(
            self.config.get("lab1_checkpoint"),
            REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        )
        self.lab1 = FrozenLab1Encoder(lab1_ckpt, device=str(self.device))
        from dggr.lab3_codec_bridge import FrozenEncodec

        self.codec = FrozenEncodec(
            model_id=str(self.config.get("codec_model_id", "facebook/encodec_24khz")),
            bandwidth=float(self.config.get("codec_bandwidth", 6.0)),
            chunk_seconds=float(self.config.get("codec_chunk_seconds", 5.0)),
            device=str(self.device),
        )

        self.generator = CodecLatentTranslator(
            in_channels=int(self.meta.codec_channels),
            z_content_dim=int(self.arrays["z_content"].shape[1]),
            z_style_dim=128,
            hidden_channels=int(self.config.get("translator_hidden_channels", 256)),
            n_blocks=int(self.config.get("translator_blocks", 10)),
            noise_dim=int(self.config.get("translator_noise_dim", 32)),
            residual_scale=float(self.config.get("translator_residual_scale", 0.5)),
            direct_output=bool(self.config.get("translator_direct_output", False)),
        ).to(self.device)
        ckpt = torch.load(str(record.checkpoint), map_location=self.device, weights_only=False)
        self.generator.load_state_dict(ckpt["generator"], strict=True)
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        self.style_judge = _load_probe_from_file(record.run_dir / "codec_style_judge.pt", CodecStyleJudge)
        if self.style_judge is not None:
            self.style_judge = self.style_judge.to(self.device)
        self.lab1_probe = _load_probe_from_file(record.run_dir / "lab1_style_probe.pt", Lab1StyleProbe)
        if self.lab1_probe is not None:
            self.lab1_probe = self.lab1_probe.to(self.device)
        self.mert_probe = _load_probe_from_file(record.run_dir / "mert_style_probe.pt", MERTStyleProbe)
        if self.mert_probe is not None:
            self.mert_probe = self.mert_probe.to(self.device)

        self.lab1_centroids = _build_centroids_from_embeddings(
            self.arrays["z_style"], self.arrays["genre_idx"], self.n_genres
        )
        self.exemplar_bank = build_style_exemplar_bank(
            self.arrays["z_style"], self.arrays["genre_idx"], self.n_genres
        )

        self.judge_centroids = None

    @torch.no_grad()
    def infer_clip(
        self,
        clip: np.ndarray,
        *,
        target_genre: str,
        style_mode: str = "mix",
        mix_alpha: float = 0.5,
        seed: int = 328,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        logger = log_callback or (lambda _msg: None)
        if target_genre not in self.genre_to_idx:
            raise ValueError(f"Unknown target genre: {target_genre}")

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        need = self.codec.target_num_samples()
        clip = np.asarray(clip, dtype=np.float32).reshape(-1)
        if clip.size < need:
            clip = np.pad(clip, (0, need - clip.size))
        elif clip.size > need:
            clip = clip[:need]
        clip = clip.astype(np.float32)

        q_src, _codes = self.codec.encode_chunk_embeddings(clip)
        q_src = self.codec.fix_num_frames(q_src).to(self.device).unsqueeze(0)

        clip_lab1 = self.codec.resample_audio(clip, int(self.codec.cfg.sample_rate), int(self.lab1.cfg.sample_rate))
        log_mel = extract_log_mel(clip_lab1, sr=int(self.lab1.cfg.sample_rate))
        log_mel = fix_log_mel_frames(log_mel, n_frames=int(self.meta.lab1_n_frames))
        lat_src = self.lab1.infer_log_mel(log_mel)
        z_content = torch.from_numpy(lat_src["z_content"]).unsqueeze(0).to(self.device)

        target_idx = int(self.genre_to_idx[target_genre])
        z_style_tgt_np = _sample_style_vector(self.lab1_centroids, self.exemplar_bank, target_idx, style_mode, mix_alpha)
        z_style_tgt = torch.from_numpy(z_style_tgt_np).unsqueeze(0).to(self.device)
        noise = self.generator.sample_noise(1, self.device)
        q_hat = self.generator(q_src, z_content, z_style_tgt, noise=noise)
        q_hat = self.codec.fix_num_frames(q_hat[0]).detach().cpu()
        wav_out = self.codec.decode_chunk_embeddings(q_hat)
        logger("Codec waveform decoded from translated latent embeddings.")

        out_lab1 = self.codec.resample_audio(wav_out, int(self.codec.cfg.sample_rate), int(self.lab1.cfg.sample_rate))
        out_log_mel = extract_log_mel(out_lab1, sr=int(self.lab1.cfg.sample_rate))
        out_log_mel = fix_log_mel_frames(out_log_mel, n_frames=int(self.meta.lab1_n_frames))
        lat_out = self.lab1.infer_log_mel(out_log_mel)

        metrics = {
            "content_cosine": round(_cosine(lat_src["z_content"], lat_out["z_content"]), 4),
            "lab1_target_style_cosine": round(_cosine(lat_out["z_style"], self.lab1_centroids[target_idx]), 4),
            "clip_seconds": round(float(len(clip)) / float(self.codec.cfg.sample_rate), 3),
            "seed": int(seed),
        }

        if self.style_judge is not None:
            q_eval = torch.from_numpy(q_hat.numpy()).unsqueeze(0).to(self.device)
            logits = self.style_judge(q_eval)
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            metrics["judge_predicted_genre"] = self.idx_to_genre[pred_idx]
            metrics["judge_target_confidence"] = round(float(probs[target_idx]), 4)
            metrics["judge_top_confidence"] = round(float(probs[pred_idx]), 4)
            if self.judge_centroids is not None:
                emb_out = self.style_judge.embed(q_eval)[0].detach().cpu().numpy()
                metrics["judge_target_centroid_cosine"] = round(_cosine(emb_out, self.judge_centroids[target_idx]), 4)

        return {
            "sr": int(self.codec.cfg.sample_rate),
            "source_audio": clip,
            "generated_audio": wav_out.astype(np.float32),
            "metrics": metrics,
            "target_style_vector": z_style_tgt_np,
            "record": self.record,
        }

    @torch.no_grad()
    def infer(
        self,
        audio_path: str,
        *,
        target_genre: str,
        style_mode: str = "mix",
        mix_alpha: float = 0.5,
        start_sec: float = 0.0,
        seed: int = 328,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        src_audio, src_sr = _load_audio_file(Path(audio_path), int(self.codec.cfg.sample_rate))
        need = self.codec.target_num_samples()
        start = int(round(float(start_sec) * src_sr))
        clip = src_audio[start:start + need]
        result = self.infer_clip(
            clip,
            target_genre=target_genre,
            style_mode=style_mode,
            mix_alpha=mix_alpha,
            seed=seed,
            log_callback=log_callback,
        )
        result["source_audio"] = np.asarray(clip, dtype=np.float32)
        return result


class DiffusionInferenceSession:
    def __init__(self, record: DiffusionRunRecord, device: str = "auto") -> None:
        self.record = record
        self.device = _resolve_device(device)
        self.config = record.config
        self.index_df, self.arrays, self.genre_to_idx, self.meta = load_diffusion_cache(record.cache_dir, mmap=False)
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
        self.n_genres = len(self.genre_to_idx)
        self.lab1 = FrozenLab1Encoder(
            REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
            device=str(self.device),
        )
        self.model = DiffusionUNetV2(
            in_channels=15,
            out_channels=1,
            base_ch=int(self.config.get("base_ch", 64)),
            ch_mults=tuple(int(x) for x in self.config.get("ch_mults", [1, 2, 4, 4])),
            n_res=int(self.config.get("n_res", 2)),
            attn_levels=tuple(int(x) for x in self.config.get("attn_levels", [2, 3])),
            z_content_dim=128,
            z_style_dim=128,
            dropout=float(self.config.get("dropout", 0.1)),
        ).to(self.device)
        self.schedule = NoiseSchedule(T=1000).to(self.device)
        self.ema = EMA(self.model, decay=float(self.config.get("ema_decay", 0.9999)))
        ckpt = torch.load(str(record.checkpoint), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.ema.load_state_dict(ckpt["ema"])

        self.style_centroids = _build_centroids_from_embeddings(
            self.arrays["z_style"], self.arrays["genre_idx"], self.n_genres
        )

    @torch.no_grad()
    def infer(
        self,
        audio_path: str,
        *,
        target_genre: str,
        start_sec: float = 0.0,
        clip_seconds: float = 3.0,
        guidance_scale: float = 2.0,
        ddim_steps: int = 50,
        eta: float = 0.0,
        seed: int = 328,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, object]:
        logger = log_callback or (lambda _msg: None)
        if target_genre not in self.genre_to_idx:
            raise ValueError(f"Unknown target genre: {target_genre}")

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        src_audio, sr = _load_audio_file(Path(audio_path), int(self.meta.sr))
        start = int(round(float(start_sec) * sr))
        want = int(round(float(clip_seconds) * sr))
        clip = src_audio[start:start + want]
        if clip.size < want:
            clip = np.pad(clip, (0, want - clip.size))
        clip = clip.astype(np.float32)

        n_frames = max(32, int(round((len(clip) / float(sr)) * float(self.meta.sr) / 256.0)))
        n_frames = int(math.ceil(n_frames / 16.0) * 16)
        max_frames = int(self.config.get("max_frames", 256))
        n_frames = min(n_frames, max_frames)

        mel = extract_bigvgan_mel_np(clip, sr=int(self.meta.sr))
        mel = pad_or_trim(mel, n_frames, axis=1, pad_val=float(self.meta.mel_min))
        mel_norm = ((mel - float(self.meta.mel_min)) / max(1e-6, float(self.meta.mel_max - self.meta.mel_min)))
        mel_norm = np.clip(mel_norm * 2.0 - 1.0, -1.0, 1.0).astype(np.float32)

        chroma = pad_or_trim(extract_chroma(clip, sr=int(self.meta.sr)), n_frames, axis=1, pad_val=0.0)
        onset = pad_or_trim(extract_onset(clip, sr=int(self.meta.sr)), n_frames, axis=0, pad_val=0.0)
        beat = pad_or_trim(extract_beat_grid(clip, sr=int(self.meta.sr), n_frames=n_frames), n_frames, axis=0, pad_val=0.0)
        H = mel_norm.shape[0]
        cond_feat = np.concatenate(
            [
                np.repeat(chroma[:, None, :], H, axis=1),
                np.repeat(onset[None, None, :], H, axis=1),
                np.repeat(beat[None, None, :], H, axis=1),
            ],
            axis=0,
        ).astype(np.float32)

        lab1_audio = librosa.resample(clip, orig_sr=int(self.meta.sr), target_sr=int(self.lab1.cfg.sample_rate), res_type="soxr_hq")
        log_mel = extract_log_mel(lab1_audio.astype(np.float32), sr=int(self.lab1.cfg.sample_rate))
        log_mel = fix_log_mel_frames(log_mel, n_frames=256)
        lat_src = self.lab1.infer_log_mel(log_mel)

        target_idx = int(self.genre_to_idx[target_genre])
        z_style_tgt = torch.from_numpy(self.style_centroids[target_idx]).unsqueeze(0).to(self.device)
        z_content = torch.from_numpy(lat_src["z_content"]).unsqueeze(0).to(self.device)
        cond_feat_t = torch.from_numpy(cond_feat).unsqueeze(0).to(self.device)
        logger("Loading BigVGAN vocoder.")
        vocoder = SESSION_CACHE.bigvgan.get(self.device)
        logger("Running DDIM sampler.")

        mel_gen = ddim_sample_v2(
            self.ema.shadow,
            self.schedule,
            cond_feat_t,
            z_content,
            z_style_tgt,
            n_steps=int(ddim_steps),
            guidance_scale=float(guidance_scale),
            eta=float(eta),
            device=self.device,
        )
        wav_out = vocode_bigvgan(
            mel_gen,
            float(self.meta.mel_min),
            float(self.meta.mel_max),
            vocoder,
            self.device,
        )[0]
        logger("Diffusion mel decoded to waveform with BigVGAN.")

        out_lab1 = librosa.resample(wav_out.astype(np.float32), orig_sr=int(self.meta.sr), target_sr=int(self.lab1.cfg.sample_rate), res_type="soxr_hq")
        out_log_mel = extract_log_mel(out_lab1.astype(np.float32), sr=int(self.lab1.cfg.sample_rate))
        out_log_mel = fix_log_mel_frames(out_log_mel, n_frames=256)
        lat_out = self.lab1.infer_log_mel(out_log_mel)

        metrics = {
            "content_cosine": round(_cosine(lat_src["z_content"], lat_out["z_content"]), 4),
            "lab1_target_style_cosine": round(_cosine(lat_out["z_style"], self.style_centroids[target_idx]), 4),
            "pitch_correlation": round(float(pitch_correlation(clip, wav_out, sr=int(self.meta.sr))), 4),
            "ddim_steps": int(ddim_steps),
            "guidance_scale": float(guidance_scale),
            "seed": int(seed),
            "n_frames": int(n_frames),
        }
        pred_label, pred_cos = _nearest_centroid_label(lat_out["z_style"], self.style_centroids, self.idx_to_genre)
        metrics["nearest_style_genre"] = pred_label
        metrics["nearest_style_cosine"] = round(pred_cos, 4)

        return {
            "sr": int(self.meta.sr),
            "source_audio": clip,
            "generated_audio": wav_out.astype(np.float32),
            "metrics": metrics,
            "record": self.record,
        }


def get_codec_session(run_name: str, checkpoint_path: Optional[str] = None, device: str = "auto", log_callback: Optional[Callable[[str], None]] = None) -> CodecInferenceSession:
    logger = log_callback or (lambda _msg: None)
    resolved_checkpoint = str(Path(checkpoint_path)) if checkpoint_path else None
    key = (run_name, resolved_checkpoint or "", device)
    if key not in SESSION_CACHE.codec_sessions:
        matches = [r for r in discover_codec_runs() if r.run_name == run_name]
        if not matches:
            raise ValueError(f"Codec run not found: {run_name}")
        record = matches[0]
        if resolved_checkpoint:
            checkpoint = Path(resolved_checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Codec checkpoint not found: {checkpoint}")
            record = CodecRunRecord(
                run_name=record.run_name,
                run_dir=record.run_dir,
                checkpoint=checkpoint,
                checkpoints=record.checkpoints,
                cache_dir=record.cache_dir,
                run_state=record.run_state,
                genre_to_idx=record.genre_to_idx,
                metrics=record.metrics,
            )
        logger(f"Loading codec session for {run_name} ({record.checkpoint.name}) on device={device}")
        SESSION_CACHE.codec_sessions[key] = CodecInferenceSession(record, device=device)
        logger("Codec checkpoint, cache, and probes loaded.")
    else:
        logger("Reusing cached codec session.")
    return SESSION_CACHE.codec_sessions[key]


def get_diffusion_session(run_name: str, checkpoint_path: Optional[str] = None, device: str = "auto", log_callback: Optional[Callable[[str], None]] = None) -> DiffusionInferenceSession:
    logger = log_callback or (lambda _msg: None)
    resolved_checkpoint = str(Path(checkpoint_path)) if checkpoint_path else None
    key = (run_name, resolved_checkpoint or "", device)
    if key not in SESSION_CACHE.diffusion_sessions:
        matches = [r for r in discover_diffusion_runs() if r.run_name == run_name]
        if not matches:
            raise ValueError(f"Diffusion run not found: {run_name}")
        record = matches[0]
        if resolved_checkpoint:
            checkpoint = Path(resolved_checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint}")
            record = DiffusionRunRecord(
                run_name=record.run_name,
                run_dir=record.run_dir,
                checkpoint=checkpoint,
                checkpoints=record.checkpoints,
                cache_dir=record.cache_dir,
                config=record.config,
                genre_to_idx=record.genre_to_idx,
            )
        logger(f"Loading diffusion session for {run_name} ({record.checkpoint.name}) on device={device}")
        SESSION_CACHE.diffusion_sessions[key] = DiffusionInferenceSession(record, device=device)
        logger("Diffusion checkpoint and cache loaded.")
    else:
        logger("Reusing cached diffusion session.")
    return SESSION_CACHE.diffusion_sessions[key]


def analyze_audio_for_ui(audio_path: Optional[str]) -> Tuple[str, Optional[Path]]:
    info = get_audio_info(audio_path)
    if not info.get("ok"):
        return f"### Audio status\n\n{info['message']}", None
    p = Path(str(info["path"]))
    y, sr = librosa.load(str(p), sr=22050, mono=True, dtype=np.float32, duration=15.0, res_type="soxr_hq")
    preview = _output_run_dir("analysis")
    fig_path = _plot_audio_pair(y, y, sr=int(sr), title=f"Input analysis: {p.name}", out_path=preview / "input_preview.png")
    md = "### Audio status\n\n" + _markdown_table(
        [
            ("File", p.name),
            ("Sample rate", info["sample_rate"]),
            ("Duration", _seconds_label(float(info["seconds"]))),
            ("Samples", info["samples"]),
            ("Peak abs", info["peak_abs"]),
        ]
    )
    return md, str(fig_path)


def run_codec_job(
    audio_path: str,
    codec_run: str,
    codec_checkpoint: Optional[str],
    target_genre: str,
    style_mode: str,
    mix_alpha: float,
    start_sec: float,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame]:
    logger = log_callback or (lambda _msg: None)
    ckpt_label = Path(codec_checkpoint).name if codec_checkpoint else "default"
    logger(f"Codec job requested on device={device}, run={codec_run}, checkpoint={ckpt_label}, target={target_genre}")
    bundle_dir = _output_run_dir("codec")
    logger(f"Output bundle: {bundle_dir}")
    session = get_codec_session(codec_run, checkpoint_path=codec_checkpoint, device=device, log_callback=logger)
    logger("Codec session ready. Running EnCodec + translator inference.")
    result = session.infer(
        audio_path,
        target_genre=target_genre,
        style_mode=style_mode,
        mix_alpha=mix_alpha,
        start_sec=start_sec,
        seed=seed,
        log_callback=logger,
    )
    logger("Codec inference finished. Rendering outputs and metrics.")
    src_path = _save_audio(bundle_dir / "source.wav", result["source_audio"], result["sr"])
    gen_path = _save_audio(bundle_dir / "codec_generated.wav", result["generated_audio"], result["sr"])
    fig_path = _plot_audio_pair(
        result["source_audio"],
        result["generated_audio"],
        sr=int(result["sr"]),
        title=f"Codec transfer: {Path(audio_path).name} -> {target_genre}",
        out_path=bundle_dir / "codec_compare.png",
    )
    metrics_path = _save_json(result["metrics"], bundle_dir / "metrics.json")
    _save_json(
        {
            "mode": "codec",
            "input_audio": str(audio_path),
            "codec_run": codec_run,
            "codec_checkpoint": codec_checkpoint,
            "target_genre": target_genre,
            "style_mode": style_mode,
            "mix_alpha": mix_alpha,
            "start_sec": start_sec,
            "seed": seed,
            "device": device,
            "metrics_path": str(metrics_path),
        },
        bundle_dir / "job.json",
    )
    zip_path = _zip_bundle(bundle_dir)
    logger(f"Codec bundle complete: {zip_path.name}")
    md = "\n\n".join(
        [
            "### Codec result",
            _markdown_table(
                [
                    ("Run", codec_run),
                    ("Checkpoint", Path(codec_checkpoint).name if codec_checkpoint else session.record.checkpoint.name),
                    ("Target genre", target_genre),
                    ("Style mode", style_mode),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    metrics_df = pd.DataFrame([result["metrics"]])
    return md, (int(result["sr"]), result["source_audio"]), (int(result["sr"]), result["generated_audio"]), str(fig_path), str(zip_path), metrics_df


def run_diffusion_job(
    audio_path: str,
    diffusion_run: str,
    diffusion_checkpoint: Optional[str],
    target_genre: str,
    start_sec: float,
    clip_seconds: float,
    guidance_scale: float,
    ddim_steps: int,
    eta: float,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame]:
    logger = log_callback or (lambda _msg: None)
    logger(
        f"Diffusion job requested on device={device}, run={diffusion_run}, checkpoint={Path(diffusion_checkpoint).name if diffusion_checkpoint else 'default'}, target={target_genre}, "
        f"steps={ddim_steps}, guidance={guidance_scale}"
    )
    bundle_dir = _output_run_dir("diffusion")
    logger(f"Output bundle: {bundle_dir}")
    session = get_diffusion_session(diffusion_run, checkpoint_path=diffusion_checkpoint, device=device, log_callback=logger)
    logger("Diffusion session ready. Starting mel sampling.")
    result = session.infer(
        audio_path,
        target_genre=target_genre,
        start_sec=start_sec,
        clip_seconds=clip_seconds,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        eta=eta,
        seed=seed,
        log_callback=logger,
    )
    logger("Diffusion sampling finished. Rendering outputs and metrics.")
    _save_audio(bundle_dir / "source.wav", result["source_audio"], result["sr"])
    _save_audio(bundle_dir / "diffusion_generated.wav", result["generated_audio"], result["sr"])
    fig_path = _plot_audio_pair(
        result["source_audio"],
        result["generated_audio"],
        sr=int(result["sr"]),
        title=f"Diffusion transfer: {Path(audio_path).name} -> {target_genre}",
        out_path=bundle_dir / "diffusion_compare.png",
    )
    _save_json(result["metrics"], bundle_dir / "metrics.json")
    _save_json(
        {
            "mode": "diffusion",
            "input_audio": str(audio_path),
            "diffusion_run": diffusion_run,
            "diffusion_checkpoint": diffusion_checkpoint,
            "target_genre": target_genre,
            "start_sec": start_sec,
            "clip_seconds": clip_seconds,
            "guidance_scale": guidance_scale,
            "ddim_steps": ddim_steps,
            "eta": eta,
            "seed": seed,
            "device": device,
        },
        bundle_dir / "job.json",
    )
    zip_path = _zip_bundle(bundle_dir)
    logger(f"Diffusion bundle complete: {zip_path.name}")
    md = "\n\n".join(
        [
            "### Diffusion result",
            _markdown_table(
                [
                    ("Run", diffusion_run),
                    ("Checkpoint", Path(diffusion_checkpoint).name if diffusion_checkpoint else session.record.checkpoint.name),
                    ("Target genre", target_genre),
                    ("DDIM steps", ddim_steps),
                    ("Guidance", guidance_scale),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    metrics_df = pd.DataFrame([result["metrics"]])
    return md, (int(result["sr"]), result["source_audio"]), (int(result["sr"]), result["generated_audio"]), str(fig_path), str(zip_path), metrics_df


def run_real_music_job(
    audio_path: str,
    checkpoint: Optional[str],
    target_genre: str,
    seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame]:
    from dggr.real_music_transfer import infer_real_transfer
    from dggr.real_music_validation import audio_metrics

    logger = log_callback or (lambda _msg: None)
    choices, default_ckpt = real_music_checkpoint_choices()
    ckpt = Path(checkpoint or default_ckpt or "")
    if not ckpt.exists():
        raise FileNotFoundError(f"Real-music checkpoint not found: {ckpt}")
    cache_dir = _real_music_cache_dir()
    if not cache_dir.exists():
        raise FileNotFoundError(f"Real-music cache not found: {cache_dir}")
    if target_genre not in real_music_genres():
        raise ValueError(f"Unknown real-music target family: {target_genre}")

    logger(
        f"Real-music job requested on device={device}, checkpoint={ckpt.name}, target={target_genre}, "
        f"seconds={seconds}, chunk={chunk_seconds}, overlap={overlap_seconds}"
    )
    bundle_dir = _output_run_dir("real_music")
    out_wav = bundle_dir / "real_music_generated.wav"
    logger(f"Output bundle: {bundle_dir}")
    meta = infer_real_transfer(
        checkpoint=ckpt,
        cache_dir=cache_dir,
        source_audio=Path(audio_path),
        target_genre=target_genre,
        out_wav=out_wav,
        seconds=float(seconds),
        chunk_seconds=float(chunk_seconds),
        overlap_seconds=float(overlap_seconds),
        device_arg=device,
    )
    logger("Real-music generation finished. Rendering GUI artifacts.")
    source_audio, sr = _load_audio_file(Path(audio_path), target_sr=22050)
    source_audio = source_audio[: int(round(float(seconds) * float(sr)))]
    generated_audio, _gen_sr = _load_audio_file(out_wav, target_sr=22050)
    _save_audio(bundle_dir / "source.wav", source_audio, sr)
    fig_path = _plot_audio_pair(
        source_audio,
        generated_audio,
        sr=int(sr),
        title=f"Real-music transfer: {Path(audio_path).name} -> {target_genre}",
        out_path=bundle_dir / "real_music_compare.png",
    )
    gen_metrics = audio_metrics(generated_audio, sr=22050)
    src_metrics = audio_metrics(source_audio, sr=22050)
    metrics = {
        "duration_sec": float(len(generated_audio) / 22050.0),
        "source_duration_sec": float(len(source_audio) / 22050.0),
        "target_genre": target_genre,
        "donor_track_id": str(meta.get("donor_track_id", "")),
        "content_chroma_cos": _cosine(np.asarray(src_metrics["chroma_mean"]), np.asarray(gen_metrics["chroma_mean"])),
        "warble": float(gen_metrics["warble"]),
        "fullness": float(gen_metrics["fullness"]),
        "dynamic_range_db": float(gen_metrics["dynamic_range_db"]),
        "hf_ratio": float(gen_metrics["high_ratio"]),
        "lf_ratio": float(gen_metrics["low_ratio"]),
    }
    _save_json(metrics, bundle_dir / "metrics.json")
    _save_json(
        {
            "mode": "real_music",
            "input_audio": str(audio_path),
            "checkpoint": str(ckpt),
            "cache_dir": str(cache_dir),
            "target_genre": target_genre,
            "seconds": float(seconds),
            "chunk_seconds": float(chunk_seconds),
            "overlap_seconds": float(overlap_seconds),
            "device": device,
            "generation_meta": meta,
        },
        bundle_dir / "job.json",
    )
    zip_path = _zip_bundle(bundle_dir)
    md = "\n\n".join(
        [
            "### Real-music result",
            _markdown_table(
                [
                    ("Checkpoint", ckpt.name),
                    ("Target family", target_genre),
                    ("Requested length", _seconds_label(seconds)),
                    ("Rendered length", _seconds_label(metrics["duration_sec"])),
                    ("Donor track", metrics["donor_track_id"]),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    return md, (int(sr), source_audio), (22050, generated_audio), str(fig_path), str(zip_path), pd.DataFrame([metrics])


def run_compare_job(
    audio_path: str,
    codec_run: str,
    codec_checkpoint: Optional[str],
    diffusion_run: str,
    diffusion_checkpoint: Optional[str],
    target_genre: str,
    start_sec: float,
    codec_style_mode: str,
    codec_mix_alpha: float,
    diffusion_seconds: float,
    guidance_scale: float,
    ddim_steps: int,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame]:
    logger = log_callback or (lambda _msg: None)
    logger(
        f"Compare job requested on device={device}, codec={codec_run} ({Path(codec_checkpoint).name if codec_checkpoint else 'default'}), "
        f"diffusion={diffusion_run} ({Path(diffusion_checkpoint).name if diffusion_checkpoint else 'default'}), "
        f"target={target_genre}"
    )
    bundle_dir = _output_run_dir("compare")
    logger(f"Output bundle: {bundle_dir}")
    codec_session = get_codec_session(codec_run, checkpoint_path=codec_checkpoint, device=device, log_callback=logger)
    diffusion_session = get_diffusion_session(diffusion_run, checkpoint_path=diffusion_checkpoint, device=device, log_callback=logger)
    logger("Both sessions loaded. Running codec branch first.")

    codec_result = codec_session.infer(
        audio_path,
        target_genre=target_genre,
        style_mode=codec_style_mode,
        mix_alpha=codec_mix_alpha,
        start_sec=start_sec,
        seed=seed,
        log_callback=logger,
    )
    logger("Codec branch complete. Running diffusion branch.")
    diffusion_result = diffusion_session.infer(
        audio_path,
        target_genre=target_genre,
        start_sec=start_sec,
        clip_seconds=diffusion_seconds,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        seed=seed,
        log_callback=logger,
    )
    logger("Both branches complete. Rendering compare artifacts.")

    _save_audio(bundle_dir / "source_codec_rate.wav", codec_result["source_audio"], codec_result["sr"])
    _save_audio(bundle_dir / "codec_generated.wav", codec_result["generated_audio"], codec_result["sr"])
    _save_audio(bundle_dir / "diffusion_generated.wav", diffusion_result["generated_audio"], diffusion_result["sr"])
    fig_path = _plot_audio_triptych(
        codec_result["source_audio"],
        codec_result["generated_audio"],
        diffusion_result["generated_audio"],
        sr_codec=int(codec_result["sr"]),
        sr_diffusion=int(diffusion_result["sr"]),
        out_path=bundle_dir / "compare.png",
    )
    combined_metrics = {
        "codec_content_cosine": codec_result["metrics"].get("content_cosine"),
        "codec_style_cosine": codec_result["metrics"].get("lab1_target_style_cosine"),
        "codec_judge_conf": codec_result["metrics"].get("judge_target_confidence"),
        "diffusion_content_cosine": diffusion_result["metrics"].get("content_cosine"),
        "diffusion_style_cosine": diffusion_result["metrics"].get("lab1_target_style_cosine"),
        "diffusion_pitch_correlation": diffusion_result["metrics"].get("pitch_correlation"),
    }
    _save_json(
        {
            "codec": codec_result["metrics"],
            "diffusion": diffusion_result["metrics"],
            "summary": combined_metrics,
        },
        bundle_dir / "metrics.json",
    )
    _save_json(
        {
            "mode": "compare",
            "input_audio": str(audio_path),
            "codec_run": codec_run,
            "codec_checkpoint": codec_checkpoint,
            "diffusion_run": diffusion_run,
            "diffusion_checkpoint": diffusion_checkpoint,
            "target_genre": target_genre,
            "start_sec": start_sec,
            "codec_style_mode": codec_style_mode,
            "codec_mix_alpha": codec_mix_alpha,
            "diffusion_seconds": diffusion_seconds,
            "guidance_scale": guidance_scale,
            "ddim_steps": ddim_steps,
            "seed": seed,
            "device": device,
        },
        bundle_dir / "job.json",
    )
    zip_path = _zip_bundle(bundle_dir)
    logger(f"Compare bundle complete: {zip_path.name}")
    md = "\n\n".join(
        [
            "### Codec vs diffusion compare",
            _markdown_table(
                [
                    ("Codec run", codec_run),
                    ("Codec checkpoint", Path(codec_checkpoint).name if codec_checkpoint else codec_session.record.checkpoint.name),
                    ("Diffusion run", diffusion_run),
                    ("Diffusion checkpoint", Path(diffusion_checkpoint).name if diffusion_checkpoint else diffusion_session.record.checkpoint.name),
                    ("Target genre", target_genre),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    metrics_df = pd.DataFrame([combined_metrics])
    return (
        md,
        (int(codec_result["sr"]), codec_result["source_audio"]),
        (int(codec_result["sr"]), codec_result["generated_audio"]),
        (int(diffusion_result["sr"]), diffusion_result["generated_audio"]),
        str(fig_path),
        str(zip_path),
        metrics_df,
    )


def run_codec_longform_job(
    audio_path: str,
    codec_run: str,
    codec_checkpoint: Optional[str],
    target_genre: str,
    style_mode: str,
    mix_alpha: float,
    source_start_sec: float,
    source_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame, str]:
    logger = log_callback or (lambda _msg: None)
    ckpt_label = Path(codec_checkpoint).name if codec_checkpoint else "default"
    logger(
        f"Codec long-form job requested on device={device}, run={codec_run}, checkpoint={ckpt_label}, "
        f"target={target_genre}, seconds={source_seconds}"
    )
    bundle_dir = _output_run_dir("codec_longform")
    logger(f"Output bundle: {bundle_dir}")
    session = get_codec_session(codec_run, checkpoint_path=codec_checkpoint, device=device, log_callback=logger)
    logger("Codec session ready. Preparing source excerpt.")

    sr = int(session.codec.cfg.sample_rate)
    source_audio, _ = _load_audio_file(Path(audio_path), sr)
    start = int(round(float(source_start_sec) * float(sr)))
    want = int(round(float(source_seconds) * float(sr)))
    excerpt = np.asarray(source_audio[start:start + want], dtype=np.float32)
    if excerpt.size < want:
        excerpt = np.pad(excerpt, (0, want - excerpt.size))
    excerpt = excerpt.astype(np.float32)

    effective_chunk_seconds = float(max(float(chunk_seconds), float(session.codec.cfg.chunk_seconds)))
    chunks = _split_audio_overlapping(
        excerpt,
        chunk_seconds=effective_chunk_seconds,
        overlap_seconds=float(overlap_seconds),
        sr=sr,
    )
    logger(f"Running codec long-form chunk loop with {len(chunks)} chunks.")
    gen_chunks: List[np.ndarray] = []
    chunk_metrics: List[Dict[str, object]] = []
    for idx, chunk in enumerate(chunks):
        logger(f"Codec long-form chunk {idx + 1}/{len(chunks)}")
        result = session.infer_clip(
            chunk,
            target_genre=target_genre,
            style_mode=style_mode,
            mix_alpha=mix_alpha,
            seed=int(seed) + idx,
            log_callback=logger,
        )
        gen = np.asarray(result["generated_audio"], dtype=np.float32)
        gen_chunks.append(gen)
        chunk_path = _save_audio(bundle_dir / "chunks" / f"chunk_{idx:03d}.wav", gen, sr)
        row = {"chunk_idx": idx, "chunk_path": str(chunk_path)}
        row.update(result["metrics"])
        chunk_metrics.append(row)

    assembled = _assemble_audio_crossfade(gen_chunks, overlap_seconds=float(overlap_seconds), sr=sr)
    assembled = assembled[: len(excerpt)]
    boundary_vals = _boundary_discontinuities(
        assembled,
        chunk_seconds=effective_chunk_seconds,
        overlap_seconds=float(overlap_seconds),
        sr=sr,
    )
    metrics = {
        "n_chunks": int(len(gen_chunks)),
        "source_seconds": round(float(len(excerpt)) / float(sr), 3),
        "chunk_seconds": round(float(effective_chunk_seconds), 3),
        "overlap_seconds": round(float(overlap_seconds), 3),
        "boundary_disc_db_mean": round(float(np.mean(boundary_vals)), 4) if boundary_vals else None,
        "boundary_disc_db_max": round(float(np.max(boundary_vals)), 4) if boundary_vals else None,
        "mean_content_cosine": round(float(np.mean([float(m.get("content_cosine", 0.0)) for m in chunk_metrics])), 4) if chunk_metrics else None,
        "mean_style_cosine": round(float(np.mean([float(m.get("lab1_target_style_cosine", 0.0)) for m in chunk_metrics])), 4) if chunk_metrics else None,
    }

    src_path = _save_audio(bundle_dir / "source.wav", excerpt, sr)
    gen_path = _save_audio(bundle_dir / "longform_coherent.wav", assembled, sr)
    fig_path = _plot_audio_pair(
        excerpt,
        assembled,
        sr=sr,
        title=f"Codec long-form: {Path(audio_path).name} -> {target_genre}",
        out_path=bundle_dir / "codec_longform_compare.png",
        max_preview_seconds=min(float(source_seconds), 30.0),
    )
    _save_json({"summary": metrics, "chunks": chunk_metrics}, bundle_dir / "metrics.json")
    _save_json(
        {
            "mode": "codec_longform",
            "input_audio": str(audio_path),
            "codec_run": codec_run,
            "codec_checkpoint": codec_checkpoint,
            "target_genre": target_genre,
            "style_mode": style_mode,
            "mix_alpha": mix_alpha,
            "source_start_sec": source_start_sec,
            "source_seconds": source_seconds,
            "chunk_seconds": chunk_seconds,
            "effective_chunk_seconds": effective_chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "seed": seed,
            "device": device,
        },
        bundle_dir / "job.json",
    )
    zip_path = _zip_bundle(bundle_dir)
    logger(f"Codec long-form bundle complete: {zip_path.name}")
    md = "\n\n".join(
        [
            "### Codec long-form result",
            _markdown_table(
                [
                    ("Run", codec_run),
                    ("Checkpoint", Path(codec_checkpoint).name if codec_checkpoint else session.record.checkpoint.name),
                    ("Target genre", target_genre),
                    ("Source length", _seconds_label(source_seconds)),
                    ("Chunk seconds", round(float(effective_chunk_seconds), 2)),
                    ("Overlap seconds", round(float(overlap_seconds), 2)),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    metrics_df = pd.DataFrame([metrics])
    log_text = "\n".join([f"chunk {i + 1}/{len(gen_chunks)} complete" for i in range(len(gen_chunks))])
    return md, (sr, excerpt), (sr, assembled), str(fig_path), str(zip_path), metrics_df, log_text


def run_longform_job(
    audio_path: str,
    diffusion_run: str,
    diffusion_checkpoint: Optional[str],
    source_genre: str,
    target_genre: str,
    source_start_sec: float,
    source_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float,
    t_start: int,
    reanchor_every: int,
    reanchor_t_start: int,
    guidance_scale: float,
    style_strength: float,
    prefix_blend: float,
    source_prefix_blend: float,
    source_mel_blend: float,
    hf_source_blend: float,
    mel_time_smooth: int,
    mel_freq_smooth: int,
    assemble_domain: str,
    ddim_steps: int,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], pd.DataFrame, str]:
    logger = log_callback or (lambda _msg: None)
    actual_device, chunk_seconds, overlap_seconds, guidance_scale, ddim_steps, assemble_domain = _apply_cpu_safe_longform_profile(
        device=device,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps,
        assemble_domain=assemble_domain,
        log_callback=logger,
    )
    logger(
        f"Long-form job requested on device={actual_device}, run={diffusion_run}, checkpoint={Path(diffusion_checkpoint).name if diffusion_checkpoint else 'default'}, source={source_genre}, "
        f"target={target_genre}, seconds={source_seconds}"
    )
    bundle_dir = _output_run_dir("longform")
    logger(f"Output bundle: {bundle_dir}")
    record = next((r for r in discover_diffusion_runs() if r.run_name == diffusion_run), None)
    if record is None:
        raise ValueError(f"Diffusion run not found: {diffusion_run}")

    logger("Releasing cached Generation Lab sessions before long-form subprocess.")
    SESSION_CACHE.clear()

    source_copy = _audio_to_temp_copy(audio_path)
    logger(f"Temporary source copy prepared: {source_copy.name}")
    cmd = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(record.cache_dir),
        "--checkpoint", str(Path(diffusion_checkpoint) if diffusion_checkpoint else record.checkpoint),
        "--source-audio", str(source_copy),
        "--source-genre", str(source_genre),
        "--target-genre", str(target_genre),
        "--source-start-sec", str(source_start_sec),
        "--source-seconds", str(source_seconds),
        "--out-dir", str(bundle_dir),
        "--chunk-seconds", str(chunk_seconds),
        "--overlap-seconds", str(overlap_seconds),
        "--t-start", str(t_start),
        "--reanchor-every", str(reanchor_every),
        "--reanchor-t-start", str(reanchor_t_start),
        "--ddim-steps", str(ddim_steps),
        "--guidance-scale", str(guidance_scale),
        "--style-strength", str(style_strength),
        "--prefix-blend", str(prefix_blend),
        "--source-prefix-blend", str(source_prefix_blend),
        "--source-mel-blend", str(source_mel_blend),
        "--hf-source-blend", str(hf_source_blend),
        "--mel-time-smooth", str(mel_time_smooth),
        "--mel-freq-smooth", str(mel_freq_smooth),
        "--assemble-domain", str(assemble_domain),
        "--seed", str(seed),
        "--device", str(actual_device),
    ]
    logger("Launching Lab 4 subprocess.")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        if clean:
            logger(clean)
    return_code = proc.wait()
    log_text = "\n".join(stdout_lines)
    (bundle_dir / "longform_run.log").write_text(log_text, encoding="utf-8")
    if return_code != 0:
        if return_code in (-9, 137):
            raise RuntimeError(
                "Long-form runner was killed by the OS, likely due to memory pressure. "
                "The CPU safety profile was applied, but this job still exceeded available memory.\n\n"
                f"{log_text[-4000:]}"
            )
        raise RuntimeError(f"Long-form runner failed.\n\n{log_text[-4000:]}")

    src_path = bundle_dir / "source.wav"
    gen_path = bundle_dir / "longform_coherent.wav"
    metrics = _safe_json(bundle_dir / "coherence_metrics.json", {})
    _save_json(
        {
            "mode": "longform",
            "input_audio": str(audio_path),
            "diffusion_run": diffusion_run,
            "diffusion_checkpoint": diffusion_checkpoint,
            "source_genre": source_genre,
            "target_genre": target_genre,
            "source_start_sec": source_start_sec,
            "source_seconds": source_seconds,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "t_start": t_start,
            "reanchor_every": reanchor_every,
            "reanchor_t_start": reanchor_t_start,
            "guidance_scale": guidance_scale,
            "style_strength": style_strength,
            "prefix_blend": prefix_blend,
            "source_prefix_blend": source_prefix_blend,
            "source_mel_blend": source_mel_blend,
            "hf_source_blend": hf_source_blend,
            "mel_time_smooth": mel_time_smooth,
            "mel_freq_smooth": mel_freq_smooth,
            "assemble_domain": assemble_domain,
            "ddim_steps": ddim_steps,
            "seed": seed,
            "device": actual_device,
        },
        bundle_dir / "job.json",
    )
    src_audio, src_sr = librosa.load(str(src_path), sr=None, mono=True, dtype=np.float32)
    gen_audio, gen_sr = librosa.load(str(gen_path), sr=None, mono=True, dtype=np.float32)
    fig_path = _plot_audio_pair(src_audio, gen_audio, sr=int(src_sr), title=f"Long-form: {target_genre}", out_path=bundle_dir / "longform_compare.png")
    zip_path = _zip_bundle(bundle_dir)
    logger(f"Long-form bundle complete: {zip_path.name}")
    md = "\n\n".join(
        [
            "### Long-form result",
            _markdown_table(
                [
                    ("Run", diffusion_run),
                    ("Checkpoint", Path(diffusion_checkpoint).name if diffusion_checkpoint else record.checkpoint.name),
                    ("Source genre", source_genre),
                    ("Target genre", target_genre),
                    ("Length", _seconds_label(source_seconds)),
                    ("Output bundle", bundle_dir.name),
                    ("Zip bundle", zip_path.name),
                ]
            ),
        ]
    )
    metrics_df = pd.DataFrame([metrics])
    return md, (int(src_sr), src_audio), (int(gen_sr), gen_audio), str(fig_path), str(zip_path), metrics_df, log_text[-8000:]


def run_longform_compare_job(
    audio_path: str,
    codec_run: str,
    codec_checkpoint: Optional[str],
    codec_target_genre: str,
    codec_style_mode: str,
    codec_mix_alpha: float,
    diffusion_run: str,
    diffusion_checkpoint: Optional[str],
    diffusion_source_genre: str,
    diffusion_target_genre: str,
    source_start_sec: float,
    source_seconds: float,
    codec_chunk_seconds: float,
    codec_overlap_seconds: float,
    diff_chunk_seconds: float,
    diff_overlap_seconds: float,
    t_start: int,
    reanchor_every: int,
    reanchor_t_start: int,
    guidance_scale: float,
    style_strength: float,
    prefix_blend: float,
    source_prefix_blend: float,
    source_mel_blend: float,
    hf_source_blend: float,
    mel_time_smooth: int,
    mel_freq_smooth: int,
    assemble_domain: str,
    ddim_steps: int,
    seed: int,
    device: str,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[str, Tuple[int, np.ndarray], Tuple[int, np.ndarray], Tuple[int, np.ndarray], Optional[Path], Optional[Path], Optional[Path], Optional[Path], pd.DataFrame, str]:
    logger = log_callback or (lambda _msg: None)
    logger("Starting long-form compare: codec branch first.")
    codec_md, src_a, codec_a, codec_plot, codec_zip, codec_df, codec_log = run_codec_longform_job(
        audio_path,
        codec_run,
        codec_checkpoint,
        codec_target_genre,
        codec_style_mode,
        codec_mix_alpha,
        source_start_sec,
        source_seconds,
        codec_chunk_seconds,
        codec_overlap_seconds,
        seed,
        device,
        log_callback=logger,
    )
    logger("Codec long-form branch complete. Starting diffusion long-form branch.")
    logger("Releasing cached codec/runtime state before diffusion long-form branch.")
    SESSION_CACHE.clear()
    gc.collect()
    _diff_md, src_b, diff_a, diff_plot, diff_zip, diff_df, diff_log = run_longform_job(
        audio_path,
        diffusion_run,
        diffusion_checkpoint,
        diffusion_source_genre,
        diffusion_target_genre,
        source_start_sec,
        source_seconds,
        diff_chunk_seconds,
        diff_overlap_seconds,
        t_start,
        reanchor_every,
        reanchor_t_start,
        guidance_scale,
        style_strength,
        prefix_blend,
        source_prefix_blend,
        source_mel_blend,
        hf_source_blend,
        mel_time_smooth,
        mel_freq_smooth,
        assemble_domain,
        ddim_steps,
        seed,
        device,
        log_callback=logger,
    )
    codec_metrics = codec_df.iloc[0].to_dict() if len(codec_df) else {}
    diff_metrics = diff_df.iloc[0].to_dict() if len(diff_df) else {}
    combined = {}
    for k, v in codec_metrics.items():
        combined[f"codec_{k}"] = v
    for k, v in diff_metrics.items():
        combined[f"diffusion_{k}"] = v
    md = "\n\n".join(
        [
            "### Long-form compare",
            _markdown_table(
                [
                    ("Codec run", codec_run),
                    ("Codec checkpoint", Path(codec_checkpoint).name if codec_checkpoint else "default"),
                    ("Codec target", codec_target_genre),
                    ("Diffusion run", diffusion_run),
                    ("Diffusion checkpoint", Path(diffusion_checkpoint).name if diffusion_checkpoint else "default"),
                    ("Diffusion source/target", f"{diffusion_source_genre} -> {diffusion_target_genre}"),
                    ("Source length", _seconds_label(source_seconds)),
                ]
            ),
            "Codec bundle and diffusion bundle were produced independently so you can inspect each model's long-form behavior without mixing artifacts.",
        ]
    )
    log_text = f"[codec]\n{codec_log[-4000:]}\n\n[diffusion]\n{diff_log[-4000:]}"
    return md, src_a, codec_a, diff_a, codec_plot, diff_plot, codec_zip, diff_zip, pd.DataFrame([combined]), log_text


def system_snapshot(device: str = "auto") -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    dev = _resolve_device(device)
    append_terminal_log(f"System snapshot requested on device={_device_name(dev)}")
    md = "\n\n".join(
        [
            "### System snapshot",
            _markdown_table(
                [
                    ("Repo root", REPO_ROOT),
                    ("Device", _device_name(dev)),
                    ("CUDA available", torch.cuda.is_available()),
                    ("Codec runs", len(discover_codec_runs())),
                    ("Diffusion runs", len(discover_diffusion_runs())),
                    ("Example clips", len(_load_example_sources())),
                ]
            ),
        ]
    )
    return md, codec_runs_table(), diffusion_runs_table()


def genres_for_codec_run(run_name: str) -> List[str]:
    rec = next((r for r in discover_codec_runs() if r.run_name == run_name), None)
    return rec.genres if rec else []


def genres_for_diffusion_run(run_name: str) -> List[str]:
    rec = next((r for r in discover_diffusion_runs() if r.run_name == run_name), None)
    return rec.genres if rec else []


def codec_checkpoint_choices(run_name: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    rec = next((r for r in discover_codec_runs() if r.run_name == run_name), None)
    if rec is None:
        return [], None
    choices = [(path.name, str(path)) for path in rec.checkpoints]
    value = str(rec.checkpoint) if rec.checkpoint else (choices[0][1] if choices else None)
    return choices, value


def diffusion_checkpoint_choices(run_name: str) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    rec = next((r for r in discover_diffusion_runs() if r.run_name == run_name), None)
    if rec is None:
        return [], None
    choices = [(path.name, str(path)) for path in rec.checkpoints]
    value = str(rec.checkpoint) if rec.checkpoint else (choices[0][1] if choices else None)
    return choices, value
