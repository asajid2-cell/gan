from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diffusion_downloads_batch as ddb

from dggr.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames
from dggr.lab3_codec_bridge import FrozenEncodec
from dggr.lab3_codec_models import CodecLatentTranslator
from dggr.lab3_codec_train import build_style_centroid_bank, build_style_exemplar_bank


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _load_codec_cache_light(cache_dir: Path):
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
    with gmap_path.open("r", encoding="utf-8") as f:
        genre_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return index_df, arrays, genre_to_idx, meta


def _resolve_codec_run_dir(run_dir: Optional[Path]) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    overnight_root = REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs"
    candidates: List[Path] = []
    if overnight_root.exists():
        for tag_dir in sorted(overnight_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not tag_dir.is_dir():
                continue
            candidates.extend(sorted(tag_dir.glob("codec_*"), key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        saves_root = REPO_ROOT / "saves2" / "lab3_codec_transfer"
        if saves_root.exists():
            candidates.extend(sorted([p for p in saves_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True))
    if not candidates:
        raise RuntimeError("No codec runs found.")
    return candidates[0]


def _resolve_cache_dir(run_dir: Path, run_state: Dict[str, Any]) -> Path:
    cache_dir = run_dir / "cache"
    if cache_dir.exists():
        return cache_dir
    cfg = run_state.get("config", {}) if isinstance(run_state.get("config", {}), dict) else {}
    reuse = cfg.get("reuse_cache_dir")
    if reuse:
        p = Path(str(reuse))
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not resolve codec cache for run {run_dir}")


def _codec_checkpoint_choices(run_dir: Path) -> List[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    preferred = ["stage3_latest.pt", "stage2_latest.pt", "stage1_latest.pt"]
    found: List[Path] = []
    seen = set()
    for name in preferred:
        p = ckpt_dir / name
        if p.exists():
            found.append(p)
            seen.add(str(p))
    for p in sorted(ckpt_dir.glob("*.pt")):
        if str(p) not in seen:
            found.append(p)
            seen.add(str(p))
    return found


def _pick_longform_targets(source_genre: str, max_targets: int = 2) -> List[str]:
    preferred = ["baroque_classical", "lofi_hh_lfbb", "hiphop_xtc", "cc0_other"]
    return [g for g in preferred if g != source_genre][: max(1, int(max_targets))]


def cosine_crossfade_weights(overlap_samples: int) -> np.ndarray:
    t = np.linspace(0.0, np.pi / 2.0, max(1, int(overlap_samples)), dtype=np.float32)
    return np.cos(t).astype(np.float32) ** 2


def split_audio_overlapping(audio: np.ndarray, chunk_seconds: float, overlap_seconds: float, sr: int) -> List[Dict[str, Any]]:
    chunk_samples = int(round(float(chunk_seconds) * float(sr)))
    overlap_samples = int(round(float(overlap_seconds) * float(sr)))
    hop_samples = max(1, chunk_samples - overlap_samples)
    chunks: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append({"audio": chunk.astype(np.float32), "start_sample": int(pos), "end_sample": int(end)})
        if end >= len(audio):
            break
        pos += hop_samples
    if not chunks:
        chunks.append({"audio": np.zeros(chunk_samples, dtype=np.float32), "start_sample": 0, "end_sample": 0})
    return chunks


def assemble_audio_crossfade(chunk_wavs: Sequence[np.ndarray], overlap_seconds: float, sr: int) -> np.ndarray:
    if len(chunk_wavs) == 0:
        return np.zeros(1, dtype=np.float32)
    if len(chunk_wavs) == 1:
        return np.asarray(chunk_wavs[0], dtype=np.float32)
    overlap_samples = max(1, int(round(float(overlap_seconds) * float(sr))))
    fade = cosine_crossfade_weights(overlap_samples)
    out = np.asarray(chunk_wavs[0], dtype=np.float32).copy()
    for wav in chunk_wavs[1:]:
        cur = np.asarray(wav, dtype=np.float32)
        real_ov = min(overlap_samples, len(out), len(cur))
        if real_ov > 0:
            f = fade[:real_ov]
            out[-real_ov:] = out[-real_ov:] * f + cur[:real_ov] * (1.0 - f)
            out = np.concatenate([out, cur[real_ov:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    return out.astype(np.float32)


def chunk_boundary_discontinuity(audio: np.ndarray, *, chunk_seconds: float, overlap_seconds: float, sr: int, window_ms: float = 50.0) -> List[float]:
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=256),
        ref=np.max,
    )
    hop_samples = int(round((float(chunk_seconds) - float(overlap_seconds)) * float(sr)))
    if hop_samples <= 0:
        return []
    boundaries = np.arange(hop_samples, len(audio), hop_samples, dtype=np.int64)
    window_frames = max(1, int(round(float(window_ms) / 1000.0 * float(sr) / 256.0)))
    vals: List[float] = []
    for b in boundaries:
        frame = int(b // 256)
        if frame - window_frames < 0 or frame + window_frames >= mel_db.shape[1]:
            continue
        left = mel_db[:, frame - window_frames:frame].mean(axis=1)
        right = mel_db[:, frame:frame + window_frames].mean(axis=1)
        vals.append(float(np.mean(np.abs(left - right))))
    return vals


def _style_vector_for_target(
    target_idx: int,
    *,
    style_mode: str,
    mix_alpha: float,
    centroid_bank: torch.Tensor,
    exemplar_bank: Dict[int, torch.Tensor],
    device: torch.device,
    rng: np.random.Generator,
) -> torch.Tensor:
    z_cent = centroid_bank[target_idx : target_idx + 1].to(device)
    ex_bank = exemplar_bank.get(int(target_idx))
    if ex_bank is None or len(ex_bank) == 0:
        z_ex = z_cent
    else:
        ex_i = int(rng.integers(0, int(ex_bank.shape[0])))
        z_ex = ex_bank[ex_i : ex_i + 1].to(device)
    mode = str(style_mode).strip().lower()
    if mode == "centroid":
        z = z_cent
    elif mode == "exemplar":
        z = z_ex
    else:
        a = float(mix_alpha)
        z = a * z_cent + (1.0 - a) * z_ex
    return torch.nn.functional.normalize(z, dim=-1)


@dataclass
class CodecLongformCompareConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    downloads_dir: Path = field(default_factory=lambda: ddb.DEFAULT_DOWNLOADS)
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "codec_longform_compare")
    run_dir: Optional[Path] = None
    lab1_checkpoint: Optional[Path] = None
    n_songs: int = 2
    targets_per_song: int = 2
    source_seconds: float = 45.0
    chunk_seconds: float = 5.0
    overlap_seconds: float = 0.5
    style_mode: str = "mix"
    mix_alpha: float = 0.35
    device: str = "auto"
    seed: int = 328

    def materialize(self) -> "CodecLongformCompareConfig":
        self.downloads_dir = Path(self.downloads_dir)
        self.output_root = Path(self.output_root)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
        if self.lab1_checkpoint is not None:
            self.lab1_checkpoint = Path(self.lab1_checkpoint)
        return self


def plan_longform_jobs(cfg: CodecLongformCompareConfig) -> List[Dict[str, Any]]:
    cfg = cfg.materialize()
    rng = random.Random(cfg.seed)
    audio_rows = ddb.discover_download_audio(cfg.downloads_dir)
    long_rows = [
        row
        for row in audio_rows
        if (row["duration_seconds"] or 0.0) >= float(cfg.source_seconds) + 5.0 and row["size_bytes"] >= 10_000_000
    ]
    if len(long_rows) < max(1, int(cfg.n_songs)):
        long_rows = sorted(audio_rows, key=lambda r: (r["duration_seconds"] or 0.0, r["size_bytes"]), reverse=True)
    if not long_rows:
        raise RuntimeError(f"No suitable songs found in {cfg.downloads_dir}")
    selected = rng.sample(long_rows, k=min(int(cfg.n_songs), len(long_rows)))
    jobs: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected):
        path = Path(row["path"])
        duration = float(row["duration_seconds"] or 0.0)
        max_start = max(0.0, duration - float(cfg.source_seconds) - 0.1)
        start_sec = rng.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        source_genre = ddb.infer_source_genre(path)
        for target_genre in _pick_longform_targets(source_genre, max_targets=cfg.targets_per_song):
            jobs.append(
                {
                    "job_idx": len(jobs),
                    "song_idx": idx,
                    "source_audio": path,
                    "source_genre": source_genre,
                    "target_genre": target_genre,
                    "start_sec": round(float(start_sec), 3),
                    "source_seconds": float(cfg.source_seconds),
                    "duration_seconds": duration if duration > 0 else None,
                    "size_bytes": int(row["size_bytes"]),
                }
            )
    return jobs


def resolve_checkpoint_panel(cfg: CodecLongformCompareConfig, labels: Optional[Sequence[str]] = None) -> Tuple[Path, List[Dict[str, Any]]]:
    cfg = cfg.materialize()
    run_dir = _resolve_codec_run_dir(cfg.run_dir)
    choices = _codec_checkpoint_choices(run_dir)
    if not choices:
        raise FileNotFoundError(f"No codec checkpoints found under {run_dir / 'checkpoints'}")
    wanted = {str(x).strip().lower() for x in (labels or ["stage3_latest", "stage2_latest"])}
    panel: List[Dict[str, Any]] = []
    seen = set()
    for p in choices:
        label = p.stem
        if label.lower() in wanted:
            panel.append({"label": label, "path": p})
            seen.add(label.lower())
    if not panel:
        panel = [{"label": choices[0].stem, "path": choices[0]}]
    return run_dir, panel


def _write_manifest(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "checkpoint_label",
        "job_idx",
        "song_idx",
        "source_audio",
        "source_genre",
        "target_genre",
        "start_sec",
        "source_seconds",
        "duration_seconds",
        "size_bytes",
        "output_dir",
        "source_wav",
        "generated_wav",
        "metrics_json",
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


def _run_single_job(
    *,
    cfg: CodecLongformCompareConfig,
    job: Dict[str, Any],
    checkpoint_label: str,
    checkpoint_path: Path,
    output_dir: Path,
    generator: CodecLatentTranslator,
    encodec: FrozenEncodec,
    lab1: FrozenLab1Encoder,
    genre_to_idx: Dict[str, int],
    centroid_bank: torch.Tensor,
    exemplar_bank: Dict[int, torch.Tensor],
    device: torch.device,
) -> None:
    target_idx = int(genre_to_idx[str(job["target_genre"])])
    rng = np.random.default_rng(int(cfg.seed) + int(job["job_idx"]))
    source_audio, _ = librosa.load(
        str(job["source_audio"]),
        sr=22050,
        mono=True,
        offset=float(job["start_sec"]),
        duration=float(cfg.source_seconds),
        dtype=np.float32,
        res_type="soxr_hq",
    )
    sf.write(str(output_dir / "source.wav"), source_audio.astype(np.float32), 22050)

    chunks = split_audio_overlapping(source_audio, float(cfg.chunk_seconds), float(cfg.overlap_seconds), 22050)
    expected_len = int(round(float(cfg.chunk_seconds) * 22050.0))
    chunk_wavs: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        log_mel = extract_log_mel(chunk["audio"], sr=int(lab1.cfg.sample_rate))
        log_mel = fix_log_mel_frames(log_mel, n_frames=256)
        lat = lab1.infer_log_mel(log_mel)

        chunk_24k = librosa.resample(chunk["audio"], orig_sr=22050, target_sr=int(encodec.cfg.sample_rate), res_type="soxr_hq")
        wav_t = torch.from_numpy(chunk_24k).float().view(1, 1, -1).to(device)
        q_emb = encodec.encode_embeddings(wav_t)

        z_c = torch.from_numpy(lat["z_content"]).unsqueeze(0).to(device)
        z_s = _style_vector_for_target(
            target_idx,
            style_mode=cfg.style_mode,
            mix_alpha=float(cfg.mix_alpha),
            centroid_bank=centroid_bank,
            exemplar_bank=exemplar_bank,
            device=device,
            rng=rng,
        )
        noise = generator.sample_noise(1, device=device)
        with torch.no_grad():
            q_translated = generator(q_emb, z_c, z_s, noise=noise)
            wav_out_24k = encodec.decode_embeddings(q_translated)[0, 0].detach().cpu().numpy().astype(np.float32)
        wav_out = librosa.resample(wav_out_24k, orig_sr=int(encodec.cfg.sample_rate), target_sr=22050, res_type="soxr_hq")
        if len(wav_out) > expected_len:
            wav_out = wav_out[:expected_len]
        elif len(wav_out) < expected_len:
            wav_out = np.pad(wav_out, (0, expected_len - len(wav_out)))
        chunk_wavs.append(wav_out.astype(np.float32))
        sf.write(str(output_dir / f"chunk_{i:03d}.wav"), wav_out.astype(np.float32), 22050)

    assembled = assemble_audio_crossfade(chunk_wavs, float(cfg.overlap_seconds), 22050)
    peak = float(np.max(np.abs(assembled)))
    if peak > 0:
        assembled = assembled / peak * 0.95
    sf.write(str(output_dir / "longform_coherent.wav"), assembled.astype(np.float32), 22050)

    disc_vals = chunk_boundary_discontinuity(
        assembled,
        chunk_seconds=float(cfg.chunk_seconds),
        overlap_seconds=float(cfg.overlap_seconds),
        sr=22050,
    )
    metrics = {
        "n_chunks": int(len(chunks)),
        "duration_sec": float(len(assembled) / 22050.0),
        "avg_boundary_disc_db": float(np.mean(disc_vals)) if disc_vals else 0.0,
        "p95_boundary_disc_db": float(np.percentile(disc_vals, 95)) if disc_vals else 0.0,
        "style_mode": str(cfg.style_mode),
        "mix_alpha": float(cfg.mix_alpha),
    }
    with (output_dir / "coherence_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_compare_panel(cfg: CodecLongformCompareConfig, checkpoint_labels: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    cfg = cfg.materialize()
    run_dir, panel = resolve_checkpoint_panel(cfg, labels=checkpoint_labels)
    run_state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    cache_dir = _resolve_cache_dir(run_dir, run_state)
    index_df, arrays, genre_to_idx, meta = _load_codec_cache_light(cache_dir)
    device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device if cfg.device != "auto" else "cpu")
    lab1_ckpt = cfg.lab1_checkpoint or Path(str(run_state["config"]["lab1_checkpoint"]))

    out_dir = cfg.output_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = plan_longform_jobs(cfg)
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "checkpoint_panel.json").write_text(json.dumps(panel, indent=2, default=_json_default), encoding="utf-8")
    cfg_dump = dict(asdict(cfg))
    cfg_dump["resolved_run_dir"] = str(run_dir)
    cfg_dump["resolved_cache_dir"] = str(cache_dir)
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=_json_default), encoding="utf-8")

    lab1 = FrozenLab1Encoder(Path(lab1_ckpt), device=str(device))
    encodec = FrozenEncodec(
        model_id=str(run_state["config"].get("codec_model_id", "facebook/encodec_24khz")),
        bandwidth=float(run_state["config"].get("codec_bandwidth", 6.0)),
        chunk_seconds=float(run_state["config"].get("codec_chunk_seconds", 5.0)),
        device=str(device),
    )
    centroid_bank = build_style_centroid_bank(arrays["z_style"], arrays["genre_idx"], n_genres=len(genre_to_idx)).to(device)
    exemplar_bank = build_style_exemplar_bank(arrays["z_style"], arrays["genre_idx"], n_genres=len(genre_to_idx))

    manifest_rows: List[Dict[str, Any]] = []
    for ckpt in panel:
        checkpoint_path = Path(ckpt["path"])
        label = str(ckpt["label"])
        generator = CodecLatentTranslator(
            in_channels=int(meta["codec_channels"]),
            z_content_dim=int(arrays["z_content"].shape[1]),
            z_style_dim=128,
            hidden_channels=int(run_state["config"].get("translator_hidden_channels", 256)),
            n_blocks=int(run_state["config"].get("translator_blocks", 10)),
            noise_dim=int(run_state["config"].get("translator_noise_dim", 32)),
            residual_scale=float(run_state["config"].get("translator_residual_scale", 0.5)),
            direct_output=bool(run_state["config"].get("translator_direct_output", False)),
            direct_mix=float(run_state["config"].get("translator_direct_mix", 1.0)),
        ).to(device)
        payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        generator.load_state_dict(payload["generator"], strict=True)
        generator.eval()
        for p in generator.parameters():
            p.requires_grad = False

        for job in jobs:
            source_audio = Path(job["source_audio"])
            job_tag = f"{int(job['job_idx']):02d}_{ddb._slug(source_audio.stem)[:40]}__to__{ddb._slug(str(job['target_genre']))}"
            job_out_dir = out_dir / "clips" / label / job_tag
            job_out_dir.mkdir(parents=True, exist_ok=True)
            _run_single_job(
                cfg=cfg,
                job=job,
                checkpoint_label=label,
                checkpoint_path=checkpoint_path,
                output_dir=job_out_dir,
                generator=generator,
                encodec=encodec,
                lab1=lab1,
                genre_to_idx=genre_to_idx,
                centroid_bank=centroid_bank,
                exemplar_bank=exemplar_bank,
                device=device,
            )
            manifest_rows.append(
                {
                    "checkpoint_label": label,
                    **job,
                    "output_dir": job_out_dir,
                    "source_wav": job_out_dir / "source.wav",
                    "generated_wav": job_out_dir / "longform_coherent.wav",
                    "metrics_json": job_out_dir / "coherence_metrics.json",
                    "checkpoint_path": checkpoint_path,
                }
            )

    manifest_path = _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "tag": cfg.tag,
        "output_dir": out_dir,
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "checkpoint_panel_path": out_dir / "checkpoint_panel.json",
        "jobs_path": out_dir / "jobs.json",
        "manifest_path": manifest_path,
        "n_jobs": len(jobs),
        "n_checkpoints": len(panel),
        "total_runs": len(jobs) * len(panel),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary
