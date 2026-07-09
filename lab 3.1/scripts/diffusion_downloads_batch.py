from __future__ import annotations

import csv
import json
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
ALLOWED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
KNOWN_GENRES = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]


_SCRIPT_DIR = Path(__file__).resolve().parent
_LAB3_DIR = REPO_ROOT / "lab 3"
_LAB4_DIR = REPO_ROOT / "lab 4"
for _p in (_LAB3_DIR, _LAB4_DIR, REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames, load_audio_chunk
from src.lab3_diffusion_data import (
    DIFFUSION_HOP,
    DIFFUSION_SR,
    DiffusionCacheMeta,
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import (
    ddim_sample_v2_constrained,
    load_bigvgan_robust,
    load_checkpoint,
    vocode_bigvgan,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _safe_duration_seconds(path: Path) -> float | None:
    try:
        import librosa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def discover_download_audio(downloads_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not downloads_dir.exists():
        return rows
    for path in sorted(downloads_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_AUDIO_EXTS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size < 2_000_000:
            continue
        rows.append(
            {
                "path": path,
                "size_bytes": int(stat.st_size),
                "duration_seconds": _safe_duration_seconds(path),
                "extension": path.suffix.lower(),
            }
        )
    return rows


def infer_source_genre(path: Path) -> str:
    name = path.name.lower()
    hiphop_tokens = [
        "drake",
        "juice wrld",
        "j. cole",
        "cole",
        "future",
        "lil ",
        "uzi",
        "boogie",
        "6ix9ine",
        "playboi",
        "a$ap",
        "butcher",
        "cochise",
        "doechii",
        "durk",
        "wizkid",
        "xxxtentacion",
        "kendrick",
        "metro",
        "carti",
    ]
    classical_tokens = [
        "bach",
        "vivaldi",
        "mozart",
        "chopin",
        "beethoven",
        "baroque",
        "classical",
        "cello",
        "orchestra",
        "sonata",
    ]
    lofi_tokens = [
        "lofi",
        "chill",
        "study",
        "beats",
        "ambient",
        "dream",
        "seaspray",
        "mall grab",
        "boards of canada",
    ]
    if any(tok in name for tok in classical_tokens):
        return "baroque_classical"
    if any(tok in name for tok in lofi_tokens):
        return "lofi_hh_lfbb"
    if any(tok in name for tok in hiphop_tokens):
        return "hiphop_xtc"
    return "cc0_other"


def _latest_checkpoint_mtime(run_dir: Path) -> float:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return -1.0
    pts = list(ckpt_dir.glob("*.pt"))
    if not pts:
        return -1.0
    return max(p.stat().st_mtime for p in pts)


def find_latest_diffusion_run() -> Path:
    candidates: List[Path] = []
    overnight_root = REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs"
    if overnight_root.exists():
        for tag_dir in overnight_root.iterdir():
            if not tag_dir.is_dir():
                continue
            for run_dir in tag_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("diffusion_"):
                    candidates.append(run_dir)
    saves_root = REPO_ROOT / "saves2" / "lab3_diffusion"
    if saves_root.exists():
        for run_dir in saves_root.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_d"):
                candidates.append(run_dir)
    if not candidates:
        raise FileNotFoundError("No diffusion run directories found.")
    candidates = [p for p in candidates if (p / "checkpoints").exists()]
    if not candidates:
        raise FileNotFoundError("No diffusion run directories with checkpoints found.")
    candidates.sort(key=_latest_checkpoint_mtime, reverse=True)
    return candidates[0]


def resolve_checkpoint(run_dir: Path, checkpoint_path: Optional[Path] = None) -> Path:
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    preferred = [ckpt_dir / "latest.pt", ckpt_dir / "best.pt"]
    for path in preferred:
        if path.exists():
            return path
    pts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def list_epoch_checkpoints(run_dir: Path) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []
    def _epoch_num(path: Path) -> int:
        stem = path.stem
        if stem.startswith("epoch_"):
            try:
                return int(stem.split("_", 1)[1])
            except Exception:
                return -1
        return -1
    pts = [p for p in ckpt_dir.glob("epoch_*.pt") if p.is_file()]
    pts.sort(key=_epoch_num)
    return pts


def choose_checkpoint_panel(
    run_dir: Path,
    *,
    include_latest: bool = True,
    include_best: bool = True,
    include_epoch6: bool = True,
    n_random_epochs: int = 2,
    seed: int = 328,
) -> List[Dict[str, Any]]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    panel: List[Dict[str, Any]] = []
    seen: set[Path] = set()

    def _add(label: str, path: Path) -> None:
        path = Path(path)
        if path.exists() and path not in seen:
            panel.append({"label": label, "path": path})
            seen.add(path)

    if include_latest:
        _add("latest", ckpt_dir / "latest.pt")
    if include_best:
        _add("best", ckpt_dir / "best.pt")
    if include_epoch6:
        _add("epoch_006", ckpt_dir / "epoch_006.pt")

    epoch_paths = list_epoch_checkpoints(run_dir)
    random_candidates = [p for p in epoch_paths if p not in seen]
    if random_candidates and int(n_random_epochs) > 0:
        rng = random.Random(int(seed))
        k = min(int(n_random_epochs), len(random_candidates))
        picks = rng.sample(random_candidates, k=k)
        for path in sorted(picks, key=lambda p: p.stem):
            _add(path.stem, path)

    if not panel:
        fallback = resolve_checkpoint(run_dir)
        _add(fallback.stem, fallback)
    return panel


def _read_run_config(run_dir: Path) -> Dict[str, Any]:
    for name in ("v3_config.json", "v2_config.json"):
        path = run_dir / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_cache_dir(run_dir: Path, cache_dir: Optional[Path] = None) -> Path:
    if cache_dir is not None:
        path = Path(cache_dir)
        if not path.exists():
            raise FileNotFoundError(f"Cache directory not found: {path}")
        return path
    candidate = run_dir / "cache"
    if candidate.exists():
        return candidate
    cfg = _read_run_config(run_dir)
    for key in ("cache_dir", "cache-dir"):
        value = cfg.get(key)
        if value:
            path = Path(value)
            if path.exists():
                return path
    fallback = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not resolve diffusion cache directory.")


def snapshot_checkpoint(checkpoint_path: Path, snapshot_dir: Path, *, stable_wait_s: float = 1.0, attempts: int = 3) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    src = Path(checkpoint_path)
    for _ in range(max(1, attempts)):
        stat1 = src.stat()
        time.sleep(max(0.0, float(stable_wait_s)))
        stat2 = src.stat()
        if stat1.st_size == stat2.st_size and stat1.st_mtime_ns == stat2.st_mtime_ns:
            break
    dst = snapshot_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / (n + 1e-8)).astype(np.float32)


def _normalize_mel_np(mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    span = float(mel_max - mel_min)
    if span < 1e-6:
        span = 1.0
    mel_norm = (mel - float(mel_min)) / span
    mel_norm = mel_norm * 2.0 - 1.0
    return np.clip(mel_norm, -1.0, 1.0).astype(np.float32)


def build_style_centroids(arrays: Dict[str, np.ndarray], genre_to_idx: Dict[str, int]) -> Dict[str, np.ndarray]:
    genre_idx = np.asarray(arrays["genre_idx"]).astype(np.int64)
    z_style = np.asarray(arrays["z_style"], dtype=np.float32)
    out: Dict[str, np.ndarray] = {}
    for genre, idx in genre_to_idx.items():
        mask = genre_idx == int(idx)
        if not np.any(mask):
            continue
        centroid = z_style[mask].mean(axis=0).astype(np.float32)
        out[str(genre)] = _l2_normalize_np(centroid)
    return out


def extract_chunk_features(
    audio: np.ndarray,
    *,
    n_frames: int,
    mel_meta: DiffusionCacheMeta,
    lab1_encoder: FrozenLab1Encoder,
) -> Dict[str, np.ndarray]:
    mel = extract_bigvgan_mel_np(audio, sr=DIFFUSION_SR)
    mel = pad_or_trim(mel, n_frames, axis=1, pad_val=float(mel_meta.mel_min))
    mel_norm = _normalize_mel_np(mel, float(mel_meta.mel_min), float(mel_meta.mel_max))

    chroma = extract_chroma(audio, sr=DIFFUSION_SR)
    chroma = pad_or_trim(chroma, n_frames, axis=1, pad_val=0.0)
    onset = extract_onset(audio, sr=DIFFUSION_SR)
    onset = pad_or_trim(onset, n_frames, axis=0, pad_val=0.0)
    beat = extract_beat_grid(audio, sr=DIFFUSION_SR, n_frames=n_frames)
    beat = pad_or_trim(beat, n_frames, axis=0, pad_val=0.0)

    n_mels = mel_norm.shape[0]
    chroma_exp = np.repeat(chroma[:, None, :], n_mels, axis=1)
    onset_exp = np.repeat(onset[None, None, :], n_mels, axis=1)
    beat_exp = np.repeat(beat[None, None, :], n_mels, axis=1)
    cond_feat = np.concatenate([chroma_exp, onset_exp, beat_exp], axis=0).astype(np.float32)

    log_mel = extract_log_mel(audio, sr=lab1_encoder.cfg.sample_rate)
    log_mel = fix_log_mel_frames(log_mel, n_frames=n_frames)
    lat = lab1_encoder.infer_log_mel(log_mel)

    return {
        "mel_norm": mel_norm.astype(np.float32),
        "cond_feat": cond_feat.astype(np.float32),
        "z_content": lat["z_content"].astype(np.float32),
        "z_style": lat["z_style"].astype(np.float32),
    }


@dataclass
class DiffusionDownloadsBatchConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    downloads_dir: Path = field(default_factory=lambda: DEFAULT_DOWNLOADS)
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_downloads_batch")
    run_dir: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    cache_dir: Optional[Path] = None
    lab1_checkpoint: Path = field(
        default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    )
    n_clips: int = 36
    clip_seconds: float = 3.0
    n_frames: int = 256
    ddim_steps: int = 50
    guidance_scale: float = 2.0
    eta: float = 0.0
    t_start: int = 320
    style_strength: float = 0.90
    device: str = "auto"
    seed: int = 328
    snapshot_latest_checkpoint: bool = True
    stable_wait_s: float = 1.0

    def materialize(self) -> "DiffusionDownloadsBatchConfig":
        self.downloads_dir = Path(self.downloads_dir)
        self.output_root = Path(self.output_root)
        self.lab1_checkpoint = Path(self.lab1_checkpoint)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        return self


def resolve_inference_context(cfg: DiffusionDownloadsBatchConfig) -> Dict[str, Path]:
    cfg = cfg.materialize()
    run_dir = cfg.run_dir if cfg.run_dir is not None else find_latest_diffusion_run()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoint_path = resolve_checkpoint(run_dir, cfg.checkpoint_path)
    resolved_cache = resolve_cache_dir(run_dir, cfg.cache_dir)
    return {
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "cache_dir": resolved_cache,
    }


def plan_jobs(cfg: DiffusionDownloadsBatchConfig) -> List[Dict[str, Any]]:
    cfg = cfg.materialize()
    rng = random.Random(cfg.seed)
    audio_rows = discover_download_audio(cfg.downloads_dir)
    if len(audio_rows) < 1:
        raise RuntimeError(f"No suitable audio files found in {cfg.downloads_dir}")

    if cfg.n_clips <= len(audio_rows):
        selected_rows = rng.sample(audio_rows, cfg.n_clips)
    else:
        selected_rows = [rng.choice(audio_rows) for _ in range(cfg.n_clips)]

    target_counts = {genre: 0 for genre in KNOWN_GENRES}
    jobs: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected_rows):
        source_path = Path(row["path"])
        source_genre = infer_source_genre(source_path)
        allowed_targets = [g for g in KNOWN_GENRES if g != source_genre] or list(KNOWN_GENRES)
        min_count = min(target_counts[g] for g in allowed_targets)
        candidate_targets = [g for g in allowed_targets if target_counts[g] == min_count]
        target_genre = rng.choice(candidate_targets)
        target_counts[target_genre] += 1

        duration = float(row["duration_seconds"] or 0.0)
        max_start = max(0.0, duration - float(cfg.clip_seconds) - 0.1)
        start_sec = rng.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        jobs.append(
            {
                "clip_idx": idx,
                "source_audio": source_path,
                "source_genre_guess": source_genre,
                "target_genre": target_genre,
                "start_sec": round(float(start_sec), 3),
                "duration_seconds": duration if duration > 0 else None,
                "size_bytes": int(row["size_bytes"]),
            }
        )
    return jobs


def _prepare_output_dir(cfg: DiffusionDownloadsBatchConfig) -> Path:
    out_dir = cfg.output_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_device(device_value: str) -> torch.device:
    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def _build_model_for_run(run_dir: Path, device: torch.device) -> tuple[DiffusionUNetV2, EMA, NoiseSchedule]:
    cfg = _read_run_config(run_dir)
    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=int(cfg.get("base_ch", 64)),
        ch_mults=tuple(cfg.get("ch_mults", [1, 2, 4, 4])),
        n_res=int(cfg.get("n_res", 2)),
        attn_levels=tuple(cfg.get("attn_levels", [2, 3])),
        z_content_dim=128,
        z_style_dim=128,
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    ema = EMA(model, decay=0.9999)
    schedule = NoiseSchedule(T=1000).to(device)
    return model, ema, schedule


def _write_manifest(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "checkpoint_label",
        "clip_idx",
        "source_audio",
        "source_genre_guess",
        "target_genre",
        "start_sec",
        "duration_seconds",
        "size_bytes",
        "source_excerpt",
        "generated_audio",
        "run_dir",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            for key, value in list(serial.items()):
                if isinstance(value, Path):
                    serial[key] = str(value)
            writer.writerow(serial)
    return path


def _run_single_checkpoint_batch(
    *,
    cfg: DiffusionDownloadsBatchConfig,
    ctx: Dict[str, Path],
    checkpoint_path: Path,
    checkpoint_label: str,
    out_dir: Path,
    jobs: Sequence[Dict[str, Any]],
    style_centroids: Dict[str, np.ndarray],
    meta: DiffusionCacheMeta,
    device: torch.device,
    lab1: FrozenLab1Encoder,
    vocoder,
) -> List[Dict[str, Any]]:
    model, ema, schedule = _build_model_for_run(ctx["run_dir"], device)
    load_checkpoint(checkpoint_path, model, ema, optimizer=None, device=device)
    ema.shadow.eval()

    manifest_rows: List[Dict[str, Any]] = []
    style_strength = float(np.clip(float(cfg.style_strength), 0.0, 1.0))
    checkpoint_dir = out_dir / "clips" / checkpoint_label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== checkpoint: {checkpoint_label} ===")
    print(f"path: {checkpoint_path}")

    for job in jobs:
        clip_idx = int(job["clip_idx"])
        source_audio = Path(job["source_audio"])
        target_genre = str(job["target_genre"])
        start_sec = float(job["start_sec"])

        clip_tag = f"{clip_idx:03d}_{_slug(source_audio.stem)[:40]}_{_slug(target_genre)}"
        clip_dir = checkpoint_dir / clip_tag
        clip_dir.mkdir(parents=True, exist_ok=True)

        y = load_audio_chunk(
            path=source_audio,
            sample_rate=DIFFUSION_SR,
            seconds=float(cfg.clip_seconds),
            start_sec=start_sec,
        )
        features = extract_chunk_features(
            y,
            n_frames=int(cfg.n_frames),
            mel_meta=meta,
            lab1_encoder=lab1,
        )

        mel_src = torch.from_numpy(features["mel_norm"]).unsqueeze(0).unsqueeze(0).to(device)
        cond_feat = torch.from_numpy(features["cond_feat"]).unsqueeze(0).to(device)
        z_content = torch.from_numpy(features["z_content"]).unsqueeze(0).to(device)
        z_style_src = torch.from_numpy(features["z_style"]).unsqueeze(0).to(device)
        z_style_tgt = torch.from_numpy(style_centroids[target_genre]).unsqueeze(0).to(device)
        z_style_mix = F.normalize(
            (1.0 - style_strength) * z_style_src + style_strength * z_style_tgt,
            dim=-1,
        )

        with torch.no_grad():
            mel_gen = ddim_sample_v2_constrained(
                ema.shadow,
                schedule,
                cond_feat,
                z_content,
                z_style_mix,
                source_mel=mel_src,
                t_start=int(cfg.t_start),
                prefix_x0=None,
                prefix_frames=0,
                n_steps=int(cfg.ddim_steps),
                guidance_scale=float(cfg.guidance_scale),
                eta=float(cfg.eta),
                device=device,
            )
            wav_gen = vocode_bigvgan(mel_gen, float(meta.mel_min), float(meta.mel_max), vocoder, device)[0]

        source_excerpt_path = clip_dir / "source_excerpt.wav"
        gen_path = clip_dir / f"generated_{target_genre}.wav"
        sf.write(str(source_excerpt_path), y, DIFFUSION_SR)
        sf.write(str(gen_path), wav_gen, DIFFUSION_SR)

        row = {
            **job,
            "checkpoint_label": checkpoint_label,
            "source_excerpt": source_excerpt_path,
            "generated_audio": gen_path,
            "run_dir": ctx["run_dir"],
            "checkpoint_path": checkpoint_path,
        }
        manifest_rows.append(row)

        meta_path = clip_dir / "metadata.json"
        meta_path.write_text(json.dumps(row, indent=2, default=_json_default), encoding="utf-8")
        print(
            f"[{clip_idx + 1:02d}/{len(jobs)}] {source_audio.name} "
            f"{start_sec:.1f}s -> {target_genre}"
        )

        if device.type == "cuda" and (clip_idx + 1) % 4 == 0:
            torch.cuda.empty_cache()

    del model
    del ema
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return manifest_rows


def run_batch_inference(cfg: DiffusionDownloadsBatchConfig) -> Dict[str, Any]:
    cfg = cfg.materialize()
    device = _load_device(cfg.device)
    ctx = resolve_inference_context(cfg)
    out_dir = _prepare_output_dir(cfg)

    checkpoint_path = ctx["checkpoint_path"]
    if cfg.snapshot_latest_checkpoint and checkpoint_path.name == "latest.pt":
        checkpoint_path = snapshot_checkpoint(
            checkpoint_path,
            out_dir / "checkpoint_snapshot",
            stable_wait_s=cfg.stable_wait_s,
        )

    plan = plan_jobs(cfg)
    (out_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    job_plan_path = out_dir / "jobs.json"
    job_plan_path.write_text(json.dumps(plan, indent=2, default=_json_default), encoding="utf-8")

    cfg_dump = dict(asdict(cfg))
    cfg_dump["resolved_run_dir"] = str(ctx["run_dir"])
    cfg_dump["resolved_checkpoint_path"] = str(checkpoint_path)
    cfg_dump["resolved_cache_dir"] = str(ctx["cache_dir"])
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=_json_default), encoding="utf-8")

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(ctx["cache_dir"], mmap=True)
    _ = index_df
    style_centroids = build_style_centroids(arrays, genre_to_idx)

    print(f"Device: {device}")
    print(f"Run dir: {ctx['run_dir']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Cache: {ctx['cache_dir']}")
    print(f"Output dir: {out_dir}")

    lab1 = FrozenLab1Encoder(cfg.lab1_checkpoint, device=str(device))
    vocoder = load_bigvgan_robust(device=device)
    manifest_rows = _run_single_checkpoint_batch(
        cfg=cfg,
        ctx=ctx,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_path.stem,
        out_dir=out_dir,
        jobs=plan,
        style_centroids=style_centroids,
        meta=meta,
        device=device,
        lab1=lab1,
        vocoder=vocoder,
    )

    manifest_path = _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "tag": cfg.tag,
        "output_dir": out_dir,
        "run_dir": ctx["run_dir"],
        "checkpoint_path": checkpoint_path,
        "cache_dir": ctx["cache_dir"],
        "manifest_path": manifest_path,
        "n_clips": len(plan),
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    del vocoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run_multi_checkpoint_inference(
    cfg: DiffusionDownloadsBatchConfig,
    checkpoints: Sequence[Dict[str, Any]] | Sequence[Path],
) -> Dict[str, Any]:
    cfg = cfg.materialize()
    device = _load_device(cfg.device)
    ctx = resolve_inference_context(cfg)
    out_dir = _prepare_output_dir(cfg)
    plan = plan_jobs(cfg)
    (out_dir / "jobs").mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    normalized_checkpoints: List[Dict[str, Any]] = []
    for item in checkpoints:
        if isinstance(item, dict):
            label = str(item.get("label") or Path(item["path"]).stem)
            path = Path(item["path"])
        else:
            path = Path(item)
            label = path.stem
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        if cfg.snapshot_latest_checkpoint and path.name == "latest.pt":
            path = snapshot_checkpoint(
                path,
                out_dir / "checkpoint_snapshot" / _slug(label),
                stable_wait_s=cfg.stable_wait_s,
            )
        normalized_checkpoints.append({"label": label, "path": path})

    job_plan_path = out_dir / "jobs.json"
    job_plan_path.write_text(json.dumps(plan, indent=2, default=_json_default), encoding="utf-8")
    panel_path = out_dir / "checkpoint_panel.json"
    panel_path.write_text(json.dumps(normalized_checkpoints, indent=2, default=_json_default), encoding="utf-8")

    cfg_dump = dict(asdict(cfg))
    cfg_dump["resolved_run_dir"] = str(ctx["run_dir"])
    cfg_dump["resolved_cache_dir"] = str(ctx["cache_dir"])
    cfg_dump["checkpoint_labels"] = [row["label"] for row in normalized_checkpoints]
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=_json_default), encoding="utf-8")

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(ctx["cache_dir"], mmap=True)
    _ = index_df
    style_centroids = build_style_centroids(arrays, genre_to_idx)

    print(f"Device: {device}")
    print(f"Run dir: {ctx['run_dir']}")
    print(f"Cache: {ctx['cache_dir']}")
    print(f"Output dir: {out_dir}")
    print("Checkpoint panel:")
    for row in normalized_checkpoints:
        print(f"  - {row['label']}: {row['path']}")

    lab1 = FrozenLab1Encoder(cfg.lab1_checkpoint, device=str(device))
    vocoder = load_bigvgan_robust(device=device)
    manifest_rows: List[Dict[str, Any]] = []
    for row in normalized_checkpoints:
        manifest_rows.extend(
            _run_single_checkpoint_batch(
                cfg=cfg,
                ctx=ctx,
                checkpoint_path=Path(row["path"]),
                checkpoint_label=str(row["label"]),
                out_dir=out_dir,
                jobs=plan,
                style_centroids=style_centroids,
                meta=meta,
                device=device,
                lab1=lab1,
                vocoder=vocoder,
            )
        )

    manifest_path = _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "tag": cfg.tag,
        "output_dir": out_dir,
        "run_dir": ctx["run_dir"],
        "cache_dir": ctx["cache_dir"],
        "manifest_path": manifest_path,
        "checkpoint_panel_path": panel_path,
        "n_clips": len(plan),
        "n_checkpoints": len(normalized_checkpoints),
        "device": str(device),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    del vocoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary
