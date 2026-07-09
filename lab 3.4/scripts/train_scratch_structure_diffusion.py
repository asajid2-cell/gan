from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB33_SCRIPTS = REPO_ROOT / "lab 3.3" / "scripts"
if str(LAB33_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB33_SCRIPTS))

from dggr.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames, load_audio_chunk
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import (
    DIFFUSION_HOP,
    DIFFUSION_SR,
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from dggr.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from dggr.lab3_diffusion_train import ddim_sample_v2_constrained, load_bigvgan_robust, vocode_bigvgan
from run_hybrid_vocal_push_compare import HybridPushConfig, _json_default, _make_mix, _resolve_stems, picked_songs


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _normalize_mel_np(mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    span = float(mel_max - mel_min)
    if span < 1e-6:
        span = 1.0
    x = (mel - float(mel_min)) / span
    x = x * 2.0 - 1.0
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n < 8:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float32)
    b = np.asarray(b[:n], dtype=np.float32)
    a = a - float(a.mean())
    b = b - float(b.mean())
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-8 or db < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-8 or db < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (da * db))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def split_audio_overlapping(audio: np.ndarray, chunk_seconds: float, overlap_seconds: float, sr: int) -> List[Dict[str, Any]]:
    chunk_samples = int(round(float(chunk_seconds) * float(sr)))
    overlap_samples = int(round(float(overlap_seconds) * float(sr)))
    hop_samples = max(1, chunk_samples - overlap_samples)
    rows: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(audio):
        end = min(len(audio), pos + chunk_samples)
        chunk = audio[pos:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        rows.append({"audio": chunk.astype(np.float32), "start_sample": int(pos), "end_sample": int(end)})
        if end >= len(audio):
            break
        pos += hop_samples
    if not rows:
        rows.append({"audio": np.zeros(chunk_samples, dtype=np.float32), "start_sample": 0, "end_sample": 0})
    return rows


def assemble_audio_crossfade(chunk_wavs: List[np.ndarray], overlap_seconds: float, sr: int) -> np.ndarray:
    if not chunk_wavs:
        return np.zeros(1, dtype=np.float32)
    if len(chunk_wavs) == 1:
        return chunk_wavs[0].astype(np.float32)
    overlap_samples = max(1, int(round(float(overlap_seconds) * float(sr))))
    fade = np.cos(np.linspace(0.0, np.pi / 2.0, overlap_samples, dtype=np.float32))
    out = chunk_wavs[0].astype(np.float32).copy()
    for cur in chunk_wavs[1:]:
        cur = cur.astype(np.float32)
        real_ov = min(overlap_samples, len(out), len(cur))
        if real_ov > 0:
            f = fade[:real_ov]
            out[-real_ov:] = out[-real_ov:] * f + cur[:real_ov] * (1.0 - f)
            out = np.concatenate([out, cur[real_ov:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    return out.astype(np.float32)


def smooth_mel_tensor(mel: torch.Tensor, time_kernel: int = 0, freq_kernel: int = 0) -> torch.Tensor:
    out = mel
    if int(freq_kernel) > 1:
        kf = int(freq_kernel)
        if kf % 2 == 0:
            kf += 1
        k = torch.ones((1, 1, kf, 1), device=out.device, dtype=out.dtype) / float(kf)
        out = F.conv2d(out, k, padding=(kf // 2, 0))
    if int(time_kernel) > 1:
        kt = int(time_kernel)
        if kt % 2 == 0:
            kt += 1
        k = torch.ones((1, 1, 1, kt), device=out.device, dtype=out.dtype) / float(kt)
        out = F.conv2d(out, k, padding=(0, kt // 2))
    return out


def extract_chunk_features(
    y: np.ndarray,
    *,
    n_frames: int,
    mel_min: float,
    mel_max: float,
    lab1_encoder: FrozenLab1Encoder,
) -> Dict[str, np.ndarray]:
    mel = extract_bigvgan_mel_np(y, sr=DIFFUSION_SR)
    mel = pad_or_trim(mel, n_frames, axis=1, pad_val=float(mel_min))
    mel_norm = _normalize_mel_np(mel, mel_min, mel_max)
    chroma = extract_chroma(y, sr=DIFFUSION_SR)
    chroma = pad_or_trim(chroma, n_frames, axis=1, pad_val=0.0)
    onset = extract_onset(y, sr=DIFFUSION_SR)
    onset = pad_or_trim(onset, n_frames, axis=0, pad_val=0.0)
    beat = extract_beat_grid(y, sr=DIFFUSION_SR, n_frames=n_frames)
    beat = pad_or_trim(beat, n_frames, axis=0, pad_val=0.0)
    H = mel_norm.shape[0]
    chroma_exp = np.repeat(chroma[:, None, :], H, axis=1)
    onset_exp = np.repeat(onset[None, None, :], H, axis=1)
    beat_exp = np.repeat(beat[None, None, :], H, axis=1)
    cond_feat = np.concatenate([chroma_exp, onset_exp, beat_exp], axis=0).astype(np.float32)
    log_mel = extract_log_mel(y, sr=lab1_encoder.cfg.sample_rate)
    log_mel = fix_log_mel_frames(log_mel, n_frames=n_frames)
    lat = lab1_encoder.infer_log_mel(log_mel)
    return {
        "mel_norm": mel_norm.astype(np.float32),
        "cond_feat": cond_feat.astype(np.float32),
        "z_content": lat["z_content"].astype(np.float32),
        "z_style": lat["z_style"].astype(np.float32),
    }


@dataclass
class TrainConfig:
    cache_dir: Path = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    lab1_checkpoint: Path = REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    baseline_pack: Path = Path.home() / "Desktop" / "dggr_new_model_rounds" / "round_20260331_173856" / "compare_pack"
    out_root: Path = Path.home() / "Desktop" / "dggr_scratch_structure_diffusion"
    epochs: int = 3
    batch_size: int = 2
    max_batches_per_epoch: int = 220
    val_batches: int = 8
    max_frames: int = 320
    lr: float = 2e-4
    weight_decay: float = 1e-4
    base_ch: int = 64
    dropout: float = 0.05
    seed: int = 328
    benchmark_seconds: float = 30.0
    final_seconds: float = 60.0
    ddim_steps: int = 40
    guidance_scale: float = 2.2
    prefix_blend: float = 0.88
    overlap_seconds: float = 0.5
    chunk_seconds: float = 3.0
    startup_time_smooth: int = 9
    vocal_mix_gain: float = 0.95
    accomp_mix_gain: float = 1.0
    vocal_debleed_strength: float = 0.35
    vocal_debleed_floor: float = 0.18
    device: str = "auto"
    single_genre_target: str = ""


class StructureGenreDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        index_df: pd.DataFrame,
        indices: Sequence[int],
        *,
        mel_min: float,
        mel_max: float,
        max_frames: int,
        style_centroids: Dict[int, np.ndarray],
    ) -> None:
        self.arrays = arrays
        self.index_df = index_df.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.max_frames = int(max_frames)
        self.style_centroids = {int(k): np.asarray(v, dtype=np.float32) for k, v in style_centroids.items()}

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[int(item)])
        mel = np.asarray(self.arrays["mel"][idx], dtype=np.float32)[:, : self.max_frames]
        mel_norm = _normalize_mel_np(mel, self.mel_min, self.mel_max)
        chroma = np.asarray(self.arrays["chroma"][idx], dtype=np.float32)[:, : self.max_frames]
        onset = np.asarray(self.arrays["onset"][idx], dtype=np.float32)[: self.max_frames]
        beat = np.asarray(self.arrays["beat"][idx], dtype=np.float32)[: self.max_frames]
        H = mel_norm.shape[0]
        cond_feat = np.concatenate(
            [
                np.repeat(chroma[:, None, :], H, axis=1),
                np.repeat(onset[None, None, :], H, axis=1),
                np.repeat(beat[None, None, :], H, axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        genre_idx = int(np.asarray(self.arrays["genre_idx"], dtype=np.int64)[idx])
        z_style_raw = np.asarray(self.arrays["z_style"][idx], dtype=np.float32)
        target_style = self.style_centroids[int(genre_idx)]
        return {
            "mel": torch.from_numpy(mel_norm[None, :, :]),
            "cond_feat": torch.from_numpy(cond_feat),
            "z_content": torch.from_numpy(np.asarray(self.arrays["z_content"][idx], dtype=np.float32)),
            "z_style_raw": torch.from_numpy(z_style_raw),
            "target_style": torch.from_numpy(target_style),
            "genre_idx": torch.tensor(genre_idx, dtype=torch.long),
        }

    def genre_indices(self) -> np.ndarray:
        all_genre = np.asarray(self.arrays["genre_idx"], dtype=np.int64)
        return all_genre[self.indices].astype(np.int64)


class GenreStyleFusion(nn.Module):
    def __init__(self, num_genres: int, style_dim: int = 128, emb_dim: int = 64):
        super().__init__()
        self.genre_emb = nn.Embedding(num_genres, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim + emb_dim, style_dim * 2),
            nn.SiLU(),
            nn.Linear(style_dim * 2, style_dim),
        )
        self.genre_head = nn.Linear(style_dim, num_genres)

    def forward(self, style_vec: torch.Tensor, genre_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.genre_emb(genre_idx)
        fused = self.mlp(torch.cat([style_vec, emb], dim=-1))
        fused = F.normalize(fused, dim=-1)
        logits = self.genre_head(fused)
        return fused, logits


class MelGenreJudge(nn.Module):
    def __init__(self, num_genres: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_genres)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        h = self.net(mel).flatten(1)
        return self.head(h)


def _batch_v_to_x0(schedule: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    s_a = schedule.sqrt_alphas_cumprod[t][:, None, None, None]
    s_om = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return torch.clamp(s_a * x_t - s_om * v, -1.0, 1.0)


def _build_style_centroids(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[int, np.ndarray]:
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    z_style = np.asarray(arrays["z_style"], dtype=np.float32)
    centroids: Dict[int, np.ndarray] = {}
    for g in sorted(np.unique(genre_idx[indices]).tolist()):
        rows = indices[genre_idx[indices] == int(g)]
        cent = z_style[rows].mean(axis=0).astype(np.float32)
        cent = cent / (np.linalg.norm(cent) + 1e-8)
        centroids[int(g)] = cent
    return centroids


@torch.no_grad()
def _judge_probs_from_audio(
    audio: np.ndarray,
    judge: MelGenreJudge,
    lab1: FrozenLab1Encoder,
    *,
    seconds: float = 5.0,
    n_frames: int = 320,
    device: torch.device,
) -> np.ndarray:
    sr = int(lab1.cfg.sample_rate)
    window = int(round(float(seconds) * float(sr)))
    probs: List[np.ndarray] = []
    if len(audio) <= window:
        starts = [0]
    else:
        starts = np.linspace(0, max(0, len(audio) - window), 3, dtype=np.int64).tolist()
    for st in starts:
        seg = audio[int(st) : int(st) + window].astype(np.float32)
        mel = extract_bigvgan_mel_np(seg, sr=DIFFUSION_SR)
        mel = pad_or_trim(mel, n_frames, axis=1, pad_val=-11.5)
        mel_norm = _normalize_mel_np(mel, -11.5, 2.0)
        mel_t = torch.from_numpy(mel_norm[None, None, :, :]).to(device)
        logits = judge(mel_t)
        probs.append(torch.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.float32))
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def _audio_metrics(source_accomp: np.ndarray, gen: np.ndarray, sr: int) -> Dict[str, float]:
    n = min(len(source_accomp), len(gen))
    src = np.asarray(source_accomp[:n], dtype=np.float32)
    y = np.asarray(gen[:n], dtype=np.float32)
    chroma_src = extract_chroma(src, sr=sr)
    chroma_gen = extract_chroma(y, sr=sr)
    onset_src = extract_onset(src, sr=sr)
    onset_gen = extract_onset(y, sr=sr)
    chroma_corr = _safe_corr(chroma_src.flatten(), chroma_gen.flatten())
    onset_corr = _safe_corr(onset_src, onset_gen)
    spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=80, power=2.0)
    mel_db = librosa.power_to_db(mel + 1e-6, ref=np.max).astype(np.float32)
    mel_norm = _normalize_mel_np(mel_db, -11.5, 2.0)
    hf_band = mel_norm[56:, :]
    jitter = float(np.mean(np.abs(hf_band[:, 2:] - 2.0 * hf_band[:, 1:-1] + hf_band[:, :-2]))) if hf_band.shape[1] >= 3 else 0.0
    fullness = float(0.30 * np.tanh(rms * 12.0) + 0.35 * np.tanh(spec_bw / 3000.0) + 0.35 * np.tanh(max(0.0, contrast - 5.0) / 15.0))
    warble = float(0.65 * jitter + 0.35 * spec_flat)
    structure = float(0.65 * chroma_corr + 0.35 * onset_corr)
    return {
        "chroma_corr": float(chroma_corr),
        "onset_corr": float(onset_corr),
        "structure": float(structure),
        "fullness": float(fullness),
        "warble": float(warble),
    }


def _baseline_style_paths(manifest_path: Path) -> Dict[Tuple[str, str], Path]:
    rows: Dict[Tuple[str, str], Path] = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_audio = Path(str(row["source_audio"]))
            rows[(_slug(source_audio.stem), str(row["target_genre"]))] = Path(str(row["final_mix_wav"]))
    return rows


def _make_balanced_sampler(ds: StructureGenreDataset, seed: int) -> WeightedRandomSampler:
    genre = ds.genre_indices()
    uniq, counts = np.unique(genre, return_counts=True)
    inv = {int(g): 1.0 / float(c) for g, c in zip(uniq.tolist(), counts.tolist())}
    weights = np.asarray([inv[int(g)] for g in genre.tolist()], dtype=np.float64)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True, generator=gen)


@torch.no_grad()
def generate_accompaniment_longform(
    *,
    model: nn.Module,
    fusion: GenreStyleFusion,
    schedule: NoiseSchedule,
    lab1: FrozenLab1Encoder,
    vocoder,
    source_audio: np.ndarray,
    target_genre_idx: int,
    target_style: np.ndarray,
    mel_min: float,
    mel_max: float,
    cfg: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    fusion.eval()
    chunks = split_audio_overlapping(source_audio, float(cfg.chunk_seconds), float(cfg.overlap_seconds), DIFFUSION_SR)
    overlap_frames = int(round(float(cfg.overlap_seconds) * float(DIFFUSION_SR) / float(DIFFUSION_HOP)))
    chunk_samples = int(round(float(cfg.chunk_seconds) * float(DIFFUSION_SR)))
    style_raw_t = torch.from_numpy(np.asarray(target_style, dtype=np.float32)).unsqueeze(0).to(device)
    genre_t = torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device)
    z_style_t, _ = fusion(style_raw_t, genre_t)
    prev_tail_mel: Optional[torch.Tensor] = None
    chunk_wavs: List[np.ndarray] = []

    for ch in chunks:
        feats = extract_chunk_features(
            ch["audio"],
            n_frames=int(cfg.max_frames),
            mel_min=float(mel_min),
            mel_max=float(mel_max),
            lab1_encoder=lab1,
        )
        cond_feat = torch.from_numpy(feats["cond_feat"]).unsqueeze(0).to(device)
        z_content = torch.from_numpy(feats["z_content"]).unsqueeze(0).to(device)
        mel_gen = ddim_sample_v2_constrained(
            model,
            schedule,
            cond_feat,
            z_content,
            z_style_t,
            source_mel=None,
            t_start=900,
            prefix_x0=prev_tail_mel,
            prefix_frames=int(overlap_frames),
            prefix_blend=float(cfg.prefix_blend),
            source_prefix_x0=None,
            source_prefix_blend=0.0,
            n_steps=int(cfg.ddim_steps),
            guidance_scale=float(cfg.guidance_scale),
            eta=0.0,
            device=device,
        )
        mel_gen = smooth_mel_tensor(mel_gen, time_kernel=3, freq_kernel=0)
        fresh_start = int(overlap_frames)
        fresh_end = min(int(cfg.max_frames), fresh_start + int(round(0.45 * DIFFUSION_SR / DIFFUSION_HOP)))
        if fresh_end > fresh_start:
            mel_gen[..., fresh_start:fresh_end] = smooth_mel_tensor(
                mel_gen[..., fresh_start:fresh_end],
                time_kernel=int(cfg.startup_time_smooth),
                freq_kernel=0,
            )
        mel_gen = torch.clamp(mel_gen, -1.0, 1.0)
        prev_tail_mel = mel_gen[..., -overlap_frames:].detach() if overlap_frames > 0 else None
        wav = vocode_bigvgan(mel_gen, float(mel_min), float(mel_max), vocoder, device)[0]
        if len(wav) > chunk_samples:
            wav = wav[:chunk_samples]
        elif len(wav) < chunk_samples:
            wav = np.pad(wav, (0, chunk_samples - len(wav)))
        chunk_wavs.append(wav.astype(np.float32))
    return assemble_audio_crossfade(chunk_wavs, float(cfg.overlap_seconds), DIFFUSION_SR).astype(np.float32)


@torch.no_grad()
def benchmark_checkpoint(
    *,
    ckpt_path: Path,
    cfg: TrainConfig,
    out_dir: Path,
    genre_to_idx: Dict[str, int],
    style_centroids: Dict[int, np.ndarray],
    mel_min: float,
    mel_max: float,
    device: torch.device,
    seconds: float,
    baseline_paths: Optional[Dict[Tuple[str, str], Path]] = None,
) -> Dict[str, Any]:
    lab1 = FrozenLab1Encoder(Path(cfg.lab1_checkpoint), device=str(device))
    vocoder = load_bigvgan_robust(device=device)
    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=int(cfg.base_ch),
        ch_mults=(1, 2, 4, 4),
        n_res=2,
        attn_levels=(2, 3),
        z_content_dim=128,
        z_style_dim=128,
        dropout=float(cfg.dropout),
    ).to(device)
    fusion = GenreStyleFusion(num_genres=len(genre_to_idx)).to(device)
    judge = MelGenreJudge(num_genres=len(genre_to_idx)).to(device)
    ema = EMA(model, decay=0.9995)
    schedule = NoiseSchedule(T=1000).to(device)
    payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    ema.load_state_dict(payload["ema"])
    fusion.load_state_dict(payload["fusion"])
    judge.load_state_dict(payload["judge"])
    ema.shadow.eval()
    fusion.eval()
    judge.eval()

    compare_root = out_dir / "renders"
    compare_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("song,target_genre,accompaniment_wav,hybrid_wav,baseline_wav\n")

    hybrid_cfg = HybridPushConfig()
    song_results: List[Dict[str, Any]] = []
    all_overall: List[float] = []
    all_target_cos: List[float] = []
    all_style_margin: List[float] = []
    all_warble: List[float] = []
    all_fullness: List[float] = []
    all_structure: List[float] = []
    all_separation: List[float] = []

    for song in picked_songs():
        stems = _resolve_stems(hybrid_cfg, song)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
        source_key = _slug(Path(song["path"]).stem)
        source_genre_idx = int(genre_to_idx.get(song["source_genre"], 0))
        per_target_rows: List[Dict[str, Any]] = []
        prob_vectors: Dict[str, np.ndarray] = {}
        if str(cfg.single_genre_target).strip():
            target_items = [(str(cfg.single_genre_target), int(genre_to_idx[str(cfg.single_genre_target)]))]
        else:
            target_items = list(genre_to_idx.items())
        for target_genre, tgt_idx in target_items:
            render_dir = compare_root / source_key / target_genre
            render_dir.mkdir(parents=True, exist_ok=True)
            generated = generate_accompaniment_longform(
                model=ema.shadow,
                fusion=fusion,
                schedule=schedule,
                lab1=lab1,
                vocoder=vocoder,
                source_audio=source_acc,
                target_genre_idx=int(tgt_idx),
                target_style=style_centroids[int(tgt_idx)],
                mel_min=float(mel_min),
                mel_max=float(mel_max),
                cfg=cfg,
                device=device,
            )
            sf.write(str(render_dir / "accompaniment_generated.wav"), generated, DIFFUSION_SR)
            sf.write(str(render_dir / "longform_coherent.wav"), generated, DIFFUSION_SR)
            setting = {
                "vocal_delay_ms": 0.0,
                "vocal_mix_gain": float(cfg.vocal_mix_gain),
                "accomp_mix_gain": float(cfg.accomp_mix_gain),
                "vocal_debleed_strength": float(cfg.vocal_debleed_strength),
                "vocal_debleed_floor": float(cfg.vocal_debleed_floor),
                "backing_timing_mode": "none",
                "backing_source_blend": 0.0,
                "backing_percussive_blend": 0.0,
                "backing_post_mode": "none",
                "backing_post_strength": 0.0,
                "backing_dewarble_strength": 0.0,
                "target_genre": str(target_genre),
            }
            final_mix = _make_mix(setting, stems, render_dir)
            probs = _judge_probs_from_audio(generated, judge, lab1, n_frames=int(cfg.max_frames), device=device)
            prob_vectors[str(target_genre)] = probs
            audio_metrics = _audio_metrics(source_acc, generated, sr=DIFFUSION_SR)
            target_conf = float(probs[int(tgt_idx)])
            other = np.delete(probs, int(tgt_idx)) if len(probs) > 1 else np.zeros(1, dtype=np.float32)
            target_margin = float(target_conf - float(np.max(other)))
            baseline_wav = ""
            if baseline_paths is not None:
                base = baseline_paths.get((source_key, str(target_genre)))
                baseline_wav = str(base) if base is not None else ""
            per_target_rows.append(
                {
                    "song": source_key,
                    "target_genre": str(target_genre),
                    "generated_wav": str(render_dir / "accompaniment_generated.wav"),
                    "hybrid_wav": str(final_mix),
                    "baseline_wav": baseline_wav,
                    "target_conf": float(target_conf),
                    "target_margin": float(target_margin),
                    "judge_probs": probs.tolist(),
                    **audio_metrics,
                }
            )
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                f.write(f"{source_key},{target_genre},{render_dir / 'accompaniment_generated.wav'},{final_mix},{baseline_wav}\n")

        names = list(prob_vectors.keys())
        sep_vals: List[float] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sep_vals.append(float(np.mean(np.abs(prob_vectors[names[i]] - prob_vectors[names[j]]))))
        separation = float(np.mean(sep_vals)) if sep_vals else 0.0
        for row in per_target_rows:
            row["separation"] = float(separation)
            row["overall"] = float(
                0.30 * row["target_conf"]
                + 0.20 * row["target_margin"]
                + 0.18 * row["separation"]
                + 0.14 * row["structure"]
                + 0.18 * row["fullness"]
                - 0.18 * row["warble"]
            )
            all_overall.append(float(row["overall"]))
            all_target_cos.append(float(row["target_conf"]))
            all_style_margin.append(float(row["target_margin"]))
            all_warble.append(float(row["warble"]))
            all_fullness.append(float(row["fullness"]))
            all_structure.append(float(row["structure"]))
            all_separation.append(float(row["separation"]))
        song_results.extend(per_target_rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "seconds": float(seconds),
        "n_rows": len(song_results),
        "mean_overall": float(np.mean(all_overall)) if all_overall else 0.0,
        "mean_target_cos": float(np.mean(all_target_cos)) if all_target_cos else 0.0,
        "mean_style_margin": float(np.mean(all_style_margin)) if all_style_margin else 0.0,
        "mean_warble": float(np.mean(all_warble)) if all_warble else 0.0,
        "mean_fullness": float(np.mean(all_fullness)) if all_fullness else 0.0,
        "mean_structure": float(np.mean(all_structure)) if all_structure else 0.0,
        "mean_separation": float(np.mean(all_separation)) if all_separation else 0.0,
        "rows": song_results,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def train(cfg: TrainConfig) -> Dict[str, Any]:
    device = _device_from_arg(str(cfg.device))
    out_dir = Path(cfg.out_root) / f"scratch_diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(cfg))

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=int(cfg.seed))
    if str(cfg.single_genre_target).strip():
        if str(cfg.single_genre_target) not in genre_to_idx:
            raise ValueError(f"Unknown single_genre_target={cfg.single_genre_target}")
        keep = int(genre_to_idx[str(cfg.single_genre_target)])
        train_idx = train_idx[genre_idx[train_idx] == keep]
        val_idx = val_idx[genre_idx[val_idx] == keep]
        if len(train_idx) == 0:
            raise RuntimeError(f"No train samples for target genre {cfg.single_genre_target}")
        if len(val_idx) == 0:
            val_idx = train_idx[: min(len(train_idx), 64)]
    style_centroids = _build_style_centroids(arrays, train_idx)

    train_ds = StructureGenreDataset(
        arrays,
        index_df,
        train_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        style_centroids=style_centroids,
    )
    val_ds = StructureGenreDataset(
        arrays,
        index_df,
        val_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        style_centroids=style_centroids,
    )
    train_sampler = _make_balanced_sampler(train_ds, seed=int(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=int(cfg.base_ch),
        ch_mults=(1, 2, 4, 4),
        n_res=2,
        attn_levels=(2, 3),
        z_content_dim=128,
        z_style_dim=128,
        dropout=float(cfg.dropout),
    ).to(device)
    fusion = GenreStyleFusion(num_genres=len(genre_to_idx)).to(device)
    judge = MelGenreJudge(num_genres=len(genre_to_idx)).to(device)
    ema = EMA(model, decay=0.9995)
    schedule = NoiseSchedule(T=1000).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(fusion.parameters()) + list(judge.parameters()),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    baseline_paths = _baseline_style_paths(Path(cfg.baseline_pack) / "manifest.csv") if (Path(cfg.baseline_pack) / "manifest.csv").exists() else None

    history: List[Dict[str, Any]] = []
    best_score = -1e18
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_by_judge.pt"
    global_step = 0

    judge.train()
    judge_opt = torch.optim.AdamW(judge.parameters(), lr=3e-4, weight_decay=1e-4)
    judge_steps = 0
    for batch in train_loader:
        mel = batch["mel"].to(device)
        genre = batch["genre_idx"].to(device)
        logits = judge(mel)
        loss_j = F.cross_entropy(logits, genre)
        judge_opt.zero_grad(set_to_none=True)
        loss_j.backward()
        judge_opt.step()
        judge_steps += 1
        if judge_steps >= 200:
            break
    judge.eval()
    for p in judge.parameters():
        p.requires_grad = False

    for epoch in range(int(cfg.epochs)):
        model.train()
        fusion.train()
        losses: List[float] = []
        genre_losses: List[float] = []
        start_t = time.time()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            if int(cfg.max_batches_per_epoch) > 0 and batch_idx >= int(cfg.max_batches_per_epoch):
                break
            mel = batch["mel"].to(device)
            cond_feat = batch["cond_feat"].to(device)
            z_content = batch["z_content"].to(device)
            target_style = batch["target_style"].to(device)
            genre = batch["genre_idx"].to(device)
            if random.random() < 0.10:
                z_content = torch.zeros_like(z_content)
            z_style, genre_logits = fusion(target_style, genre)
            B = mel.shape[0]
            t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(mel)
            x_t = schedule.q_sample(mel, t, noise)
            v_target = schedule.compute_v_target(mel, t, noise)
            pred = model(torch.cat([x_t, cond_feat], dim=1), t, z_content, z_style)
            loss_diff = F.mse_loss(pred, v_target)
            x0_pred = _batch_v_to_x0(schedule, x_t, t, pred)
            fake_logits = judge(x0_pred)
            loss_style_embed = F.cross_entropy(genre_logits, genre)
            loss_judge_fake = F.cross_entropy(fake_logits, genre)
            loss = loss_diff + 0.15 * loss_style_embed + 0.35 * loss_judge_fake
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            losses.append(float(loss_diff.item()))
            genre_losses.append(float(loss_judge_fake.item()))
            global_step += 1

        model.eval()
        fusion.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= int(cfg.val_batches):
                    break
                mel = batch["mel"].to(device)
                cond_feat = batch["cond_feat"].to(device)
                z_content = batch["z_content"].to(device)
                target_style = batch["target_style"].to(device)
                genre = batch["genre_idx"].to(device)
                z_style, _ = fusion(target_style, genre)
                B = mel.shape[0]
                t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
                noise = torch.randn_like(mel)
                x_t = schedule.q_sample(mel, t, noise)
                v_target = schedule.compute_v_target(mel, t, noise)
                pred = model(torch.cat([x_t, cond_feat], dim=1), t, z_content, z_style)
                val_losses.append(float(F.mse_loss(pred, v_target).item()))

        payload = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "fusion": fusion.state_dict(),
            "judge": judge.state_dict(),
            "genre_to_idx": genre_to_idx,
            "style_centroids": {str(k): v.tolist() for k, v in style_centroids.items()},
            "cfg": asdict(cfg),
            "epoch": int(epoch + 1),
            "global_step": int(global_step),
            "meta": {"mel_min": float(meta.mel_min), "mel_max": float(meta.mel_max)},
        }
        epoch_ckpt = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(payload, str(epoch_ckpt))
        torch.save(payload, str(ckpt_dir / "latest.pt"))

        bench_dir = out_dir / "benchmark" / f"epoch_{epoch + 1:03d}"
        bench = benchmark_checkpoint(
            ckpt_path=epoch_ckpt,
            cfg=cfg,
            out_dir=bench_dir,
            genre_to_idx=genre_to_idx,
            style_centroids=style_centroids,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            device=device,
            seconds=float(cfg.benchmark_seconds),
            baseline_paths=baseline_paths,
        )
        hist_row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "genre_loss": float(np.mean(genre_losses)) if genre_losses else 0.0,
            "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
            "benchmark_overall": float(bench["mean_overall"]),
            "benchmark_target_cos": float(bench["mean_target_cos"]),
            "benchmark_style_margin": float(bench["mean_style_margin"]),
            "benchmark_structure": float(bench["mean_structure"]),
            "benchmark_fullness": float(bench["mean_fullness"]),
            "benchmark_warble": float(bench["mean_warble"]),
            "benchmark_separation": float(bench["mean_separation"]),
            "epoch_seconds": float(time.time() - start_t),
        }
        history.append(hist_row)
        _write_json(out_dir / "history.json", {"rows": history})
        if float(bench["mean_overall"]) > float(best_score):
            best_score = float(bench["mean_overall"])
            torch.save(payload, str(best_ckpt))
            _write_json(out_dir / "winner_map.json", {"best_epoch": int(epoch + 1), "best_checkpoint": str(best_ckpt)})

    if not best_ckpt.exists():
        raise RuntimeError("No best checkpoint produced.")

    final_pack_dir = out_dir / "final_pack"
    final_summary = benchmark_checkpoint(
        ckpt_path=best_ckpt,
        cfg=cfg,
        out_dir=final_pack_dir,
        genre_to_idx=genre_to_idx,
        style_centroids=style_centroids,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        device=device,
        seconds=float(cfg.final_seconds),
        baseline_paths=baseline_paths,
    )
    diag_lines = [
        "# Scratch Structure Diffusion Diagnosis",
        "",
        "- Old limitation: source-timbre anchoring preserved structure but collapsed style movement and genre separation.",
        "- New family trained from scratch: unanchored structure-conditioned accompaniment diffusion with learned genre-style fusion.",
        f"- Best checkpoint: {best_ckpt}",
        f"- Final mean overall: {final_summary['mean_overall']:.4f}",
        f"- Final mean target cosine: {final_summary['mean_target_cos']:.4f}",
        f"- Final mean style margin: {final_summary['mean_style_margin']:.4f}",
        f"- Final mean separation: {final_summary['mean_separation']:.4f}",
        f"- Final mean fullness: {final_summary['mean_fullness']:.4f}",
        f"- Final mean warble: {final_summary['mean_warble']:.4f}",
        "",
        "## Compared Paths",
        "",
        f"- Baseline pack: {cfg.baseline_pack}",
        "- New scratch model benchmarked on fixed picked songs and all four targets with the same vocal-preserve workflow.",
    ]
    (out_dir / "diagnosis_report.md").write_text("\n".join(diag_lines), encoding="utf-8")
    summary = {
        "out_dir": str(out_dir),
        "best_checkpoint": str(best_ckpt),
        "final_pack_dir": str(final_pack_dir),
        "history_rows": history,
        "final_summary": final_summary,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a scratch structure-conditioned accompaniment diffusion model and benchmark it.")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_scratch_structure_diffusion")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--baseline-pack", type=Path, default=Path.home() / "Desktop" / "dggr_new_model_rounds" / "round_20260331_173856" / "compare_pack")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-batches-per-epoch", type=int, default=220)
    ap.add_argument("--max-frames", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=328)
    ap.add_argument("--single-genre-target", type=str, default="")
    args = ap.parse_args()

    cfg = TrainConfig(
        out_root=Path(args.out_root),
        cache_dir=Path(args.cache_dir),
        baseline_pack=Path(args.baseline_pack),
        epochs=int(args.epochs),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        max_frames=int(args.max_frames),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        single_genre_target=str(args.single_genre_target),
    )
    summary = train(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
