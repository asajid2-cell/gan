from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB31_SCRIPTS = REPO_ROOT / "lab 3.1" / "scripts"
if str(LAB31_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB31_SCRIPTS))

import diffusion_downloads_batch as ddb
from dggr.lab3_bridge import FrozenLab1Encoder, extract_log_mel, fix_log_mel_frames, load_audio_chunk
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import DIFFUSION_HOP, DIFFUSION_SR, denormalize_mel, load_diffusion_cache
from dggr.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from dggr.lab3_diffusion_train import ddim_sample_v2_constrained, load_bigvgan_robust, vocode_bigvgan
from dggr.lab3_mert_bridge import FrozenMERT


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_default_base_checkpoint() -> Optional[Path]:
    candidates: List[Path] = []
    overnight_root = REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs"
    if overnight_root.exists():
        for tag_dir in sorted(overnight_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not tag_dir.is_dir():
                continue
            for pat in ["diffusion_v3_*", "diffusion_v2_*"]:
                for run_dir in sorted(tag_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True):
                    for name in ["best.pt", "epoch_005.pt", "epoch_006.pt", "latest.pt"]:
                        ckpt = run_dir / "checkpoints" / name
                        if ckpt.exists():
                            candidates.append(ckpt)
    candidates.extend(
        [
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d003" / "checkpoints" / "best.pt",
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt",
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt",
            REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_005.pt",
        ]
    )
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt
    return None


@dataclass
class RetoolConfig:
    cache_dir: Path
    out_dir: Path
    bootstrap_checkpoint: Optional[Path] = None
    epochs: int = 8
    batch_size: int = 2
    grad_accum: int = 1
    max_frames: int = 256
    lr: float = 8e-5
    ema_decay: float = 0.999
    cfg_dropout_p: float = 0.05
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 250
    max_batches_per_epoch: int = 0
    seed: int = 328
    num_workers: int = 0
    base_ch: int = 64
    ch_mults: Tuple[int, ...] = (1, 2, 4, 4)
    n_res: int = 2
    attn_levels: Tuple[int, ...] = (2, 3)
    dropout: float = 0.1
    identity_weight: float = 1.0
    style_weight: float = 1.8
    anchor_weight: float = 0.55
    envelope_weight: float = 0.30
    continuity_weight: float = 0.65
    hf_penalty_weight: float = 0.20
    vocal_weight: float = 0.40
    crackle_weight: float = 0.30
    anchor_bins: int = 44
    hf_start_bin: int = 56
    vocal_start_bin: int = 10
    vocal_end_bin: int = 42
    overlap_frames: int = 40
    hf_margin: float = 0.06
    crackle_margin: float = 0.015
    style_probe_frames: int = 256
    style_every_steps: int = 1
    style_batch_splits: int = 2
    mert_model_id: str = "m-a-p/MERT-v1-95M"
    mert_layer: int = -1
    monitor_steps: int = 25
    save_every_steps: int = 100
    resume: bool = True
    lab1_checkpoint: Path = REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    downloads_dir: Path = Path.home() / "Downloads"
    epoch_train_samples: int = 3
    epoch_download_samples: int = 3
    epoch_sample_ddim_steps: int = 50
    epoch_sample_t_start: int = 260
    epoch_sample_guidance_scale: float = 1.9
    epoch_sample_style_strength: float = 0.62
    epoch_sample_eta: float = 0.0
    source_mode: str = "cache"
    downloads_source_samples_per_epoch: int = 2000
    mixed_source_samples_per_epoch: int = 2000
    downloads_mix_ratio: float = 0.30
    source_aug_prob: float = 0.65
    source_noise_std: float = 0.015
    source_cond_noise_std: float = 0.01
    source_global_offset_std: float = 0.05
    source_hf_tilt_std: float = 0.07
    source_time_mask_prob: float = 0.30
    source_time_mask_frames: int = 24
    device: str = "auto"
    dry_run: bool = False


def _normalize_mel_np(mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    span = float(mel_max - mel_min)
    if span < 1e-6:
        span = 1.0
    mel_norm = (mel - float(mel_min)) / span
    mel_norm = mel_norm * 2.0 - 1.0
    return np.clip(mel_norm, -1.0, 1.0).astype(np.float32)


def _trim_chunk_features(
    arrays: Dict[str, np.ndarray],
    real_idx: int,
    *,
    mel_min: float,
    mel_max: float,
    max_frames: int,
) -> Dict[str, np.ndarray]:
    mel = np.asarray(arrays["mel"][real_idx], dtype=np.float32)
    chroma = np.asarray(arrays["chroma"][real_idx], dtype=np.float32)
    onset = np.asarray(arrays["onset"][real_idx], dtype=np.float32)
    beat = np.asarray(arrays["beat"][real_idx], dtype=np.float32)
    if max_frames > 0 and mel.shape[-1] > int(max_frames):
        mel = mel[:, : int(max_frames)]
        chroma = chroma[:, : int(max_frames)]
        onset = onset[: int(max_frames)]
        beat = beat[: int(max_frames)]
    mel_norm = _normalize_mel_np(mel, mel_min=mel_min, mel_max=mel_max)
    H = mel_norm.shape[0]
    chroma_exp = np.repeat(chroma[:, None, :], H, axis=1)
    onset_exp = np.repeat(onset[None, None, :], H, axis=1)
    beat_exp = np.repeat(beat[None, None, :], H, axis=1)
    cond_feat = np.concatenate([chroma_exp, onset_exp, beat_exp], axis=0).astype(np.float32)
    return {
        "mel": mel_norm[None, :, :].astype(np.float32),
        "cond_feat": cond_feat,
        "z_content": np.asarray(arrays["z_content"][real_idx], dtype=np.float32).copy(),
        "z_style": np.asarray(arrays["z_style"][real_idx], dtype=np.float32).copy(),
        "genre_idx": np.int64(arrays["genre_idx"][real_idx]),
    }


def _augment_mel_cond_pair(
    mel: np.ndarray,
    cond_feat: np.ndarray,
    *,
    rng: np.random.Generator,
    cfg: RetoolConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    mel_aug = np.asarray(mel, dtype=np.float32).copy()
    cond_aug = np.asarray(cond_feat, dtype=np.float32).copy()
    if float(cfg.source_aug_prob) <= 0.0 or float(rng.random()) > float(cfg.source_aug_prob):
        return mel_aug, cond_aug

    freq_bins = int(mel_aug.shape[1])
    time_bins = int(mel_aug.shape[2])

    if float(cfg.source_global_offset_std) > 0.0:
        mel_aug += np.float32(rng.normal(0.0, float(cfg.source_global_offset_std)))

    if float(cfg.source_hf_tilt_std) > 0.0 and freq_bins > 1:
        tilt = np.linspace(-1.0, 1.0, freq_bins, dtype=np.float32)[None, :, None]
        mel_aug += tilt * np.float32(rng.normal(0.0, float(cfg.source_hf_tilt_std)))

    if float(cfg.source_noise_std) > 0.0:
        mel_aug += rng.normal(0.0, float(cfg.source_noise_std), size=mel_aug.shape).astype(np.float32)

    if float(cfg.source_cond_noise_std) > 0.0:
        cond_aug += rng.normal(0.0, float(cfg.source_cond_noise_std), size=cond_aug.shape).astype(np.float32)

    if (
        float(cfg.source_time_mask_prob) > 0.0
        and int(cfg.source_time_mask_frames) > 0
        and time_bins > 8
        and float(rng.random()) < float(cfg.source_time_mask_prob)
    ):
        width = int(min(int(cfg.source_time_mask_frames), max(4, time_bins // 4)))
        start = int(rng.integers(0, max(1, time_bins - width)))
        stop = start + width
        mel_fill = mel_aug.mean(axis=2, keepdims=True)
        mel_aug[:, :, start:stop] = mel_fill
        cond_aug[:, :, start:stop] = 0.0

    mel_aug = np.clip(mel_aug, -1.0, 1.0)
    cond_aug = np.clip(cond_aug, -3.0, 3.0)
    return mel_aug.astype(np.float32), cond_aug.astype(np.float32)


class AdjacentChunkStyleSwapDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        index_df: pd.DataFrame,
        indices: Sequence[int],
        *,
        mel_min: float,
        mel_max: float,
        max_frames: int,
        mert_sample_rate: int,
        aug_cfg: Optional[RetoolConfig] = None,
        seed: int = 328,
    ):
        self.arrays = arrays
        self.index_df = index_df.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.max_frames = int(max_frames)
        self.mert_sample_rate = int(mert_sample_rate)
        self.chunk_seconds = float(self.max_frames * DIFFUSION_HOP / DIFFUSION_SR)
        self.aug_cfg = aug_cfg
        self.base_seed = int(seed)
        self.epoch = 0

        sub = self.index_df.iloc[self.indices].copy()
        sub["real_idx"] = self.indices
        sort_cols = ["track_id"]
        if "start_sec" in sub.columns:
            sort_cols.append("start_sec")
        elif "chunk_id" in sub.columns:
            sort_cols.append("chunk_id")
        sub = sub.sort_values(sort_cols).reset_index(drop=True)

        self.pairs: List[Tuple[int, int]] = []
        for _, grp in sub.groupby("track_id"):
            rows = grp["real_idx"].tolist()
            for a, b in zip(rows[:-1], rows[1:]):
                self.pairs.append((int(a), int(b)))
        if not self.pairs:
            raise RuntimeError("No adjacent chunk pairs found for longform-aware training.")

        genre_lookup = np.asarray(arrays["genre_idx"], dtype=np.int64)
        self.by_genre: Dict[int, np.ndarray] = {}
        for g in sorted(np.unique(genre_lookup[self.indices]).tolist()):
            rows = self.indices[genre_lookup[self.indices] == int(g)]
            if len(rows) > 0:
                self.by_genre[int(g)] = rows.astype(np.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _target_audio(self, real_idx: int) -> np.ndarray:
        row = self.index_df.iloc[int(real_idx)]
        path = Path(str(row["path"]))
        start_sec = float(row.get("start_sec", 0.0))
        return load_audio_chunk(
            path=path,
            sample_rate=self.mert_sample_rate,
            seconds=self.chunk_seconds,
            start_sec=start_sec,
        ).astype(np.float32)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        src_idx, nxt_idx = self.pairs[int(item)]
        rng = np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + int(item))
        src_feat = _trim_chunk_features(
            self.arrays, src_idx, mel_min=self.mel_min, mel_max=self.mel_max, max_frames=self.max_frames
        )
        nxt_feat = _trim_chunk_features(
            self.arrays, nxt_idx, mel_min=self.mel_min, mel_max=self.mel_max, max_frames=self.max_frames
        )
        if self.aug_cfg is not None:
            src_mel_aug, src_cond_aug = _augment_mel_cond_pair(src_feat["mel"], src_feat["cond_feat"], rng=rng, cfg=self.aug_cfg)
            nxt_mel_aug, nxt_cond_aug = _augment_mel_cond_pair(nxt_feat["mel"], nxt_feat["cond_feat"], rng=rng, cfg=self.aug_cfg)
            src_feat["mel"] = src_mel_aug
            src_feat["cond_feat"] = src_cond_aug
            nxt_feat["mel"] = nxt_mel_aug
            nxt_feat["cond_feat"] = nxt_cond_aug
        src_genre = int(src_feat["genre_idx"])
        target_genres = [g for g in self.by_genre.keys() if g != src_genre]
        target_genre = int(rng.choice(target_genres if target_genres else [src_genre]))
        target_real_idx = int(rng.choice(self.by_genre[target_genre]))
        target_audio = self._target_audio(target_real_idx)
        target_z_style = np.asarray(self.arrays["z_style"][target_real_idx], dtype=np.float32).copy()
        return {
            "src_mel": torch.from_numpy(src_feat["mel"]),
            "src_cond_feat": torch.from_numpy(src_feat["cond_feat"]),
            "src_z_content": torch.from_numpy(src_feat["z_content"]),
            "src_z_style": torch.from_numpy(src_feat["z_style"]),
            "src_genre_idx": torch.tensor(src_genre, dtype=torch.long),
            "nxt_mel": torch.from_numpy(nxt_feat["mel"]),
            "nxt_cond_feat": torch.from_numpy(nxt_feat["cond_feat"]),
            "nxt_z_content": torch.from_numpy(nxt_feat["z_content"]),
            "nxt_z_style": torch.from_numpy(nxt_feat["z_style"]),
            "target_genre_idx": torch.tensor(target_genre, dtype=torch.long),
            "target_z_style": torch.from_numpy(target_z_style),
            "target_audio_mert": torch.from_numpy(target_audio),
        }


class DownloadsSourceAdjacentDataset(Dataset):
    def __init__(
        self,
        *,
        downloads_dir: Path,
        cache_arrays: Dict[str, np.ndarray],
        cache_index_df: pd.DataFrame,
        cache_indices: Sequence[int],
        genre_to_idx: Dict[str, int],
        mel_min: float,
        mel_max: float,
        max_frames: int,
        mert_sample_rate: int,
        lab1_encoder: FrozenLab1Encoder,
        samples_per_epoch: int,
        aug_cfg: Optional[RetoolConfig] = None,
        seed: int = 328,
    ):
        self.cache_arrays = cache_arrays
        self.cache_index_df = cache_index_df.reset_index(drop=True)
        self.cache_indices = np.asarray(cache_indices, dtype=np.int64)
        self.genre_to_idx = {str(k): int(v) for k, v in genre_to_idx.items()}
        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.max_frames = int(max_frames)
        self.mert_sample_rate = int(mert_sample_rate)
        self.lab1_encoder = lab1_encoder
        self.samples_per_epoch = max(1, int(samples_per_epoch))
        self.chunk_seconds = float(self.max_frames * DIFFUSION_HOP / DIFFUSION_SR)
        self.aug_cfg = aug_cfg
        self.base_seed = int(seed)
        self.epoch = 0

        self.by_genre: Dict[int, np.ndarray] = {}
        cache_genre_lookup = np.asarray(self.cache_arrays["genre_idx"], dtype=np.int64)
        for g in sorted(np.unique(cache_genre_lookup[self.cache_indices]).tolist()):
            rows = self.cache_indices[cache_genre_lookup[self.cache_indices] == int(g)]
            if len(rows) > 0:
                self.by_genre[int(g)] = rows.astype(np.int64)

        min_seconds = self.chunk_seconds * 2.05
        audio_rows = ddb.discover_download_audio(Path(downloads_dir))
        self.download_rows: List[Dict[str, object]] = []
        for row in audio_rows:
            duration = float(row.get("duration_seconds") or 0.0)
            if duration < min_seconds:
                continue
            path = Path(str(row["path"]))
            self.download_rows.append(
                {
                    "path": path,
                    "duration_seconds": duration,
                    "source_genre": str(ddb.infer_source_genre(path)),
                }
            )
        if not self.download_rows:
            raise RuntimeError(f"No usable downloads audio found in {downloads_dir} for chunk_seconds={self.chunk_seconds:.2f}.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _target_audio(self, real_idx: int) -> np.ndarray:
        row = self.cache_index_df.iloc[int(real_idx)]
        path = Path(str(row["path"]))
        start_sec = float(row.get("start_sec", 0.0))
        return load_audio_chunk(
            path=path,
            sample_rate=self.mert_sample_rate,
            seconds=self.chunk_seconds,
            start_sec=start_sec,
        ).astype(np.float32)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + int(item))
        row = self.download_rows[int(rng.integers(0, len(self.download_rows)))]
        path = Path(str(row["path"]))
        duration = float(row["duration_seconds"])
        max_start = max(0.0, duration - self.chunk_seconds * 2.0 - 0.05)
        start_sec = float(rng.uniform(0.0, max_start)) if max_start > 0.0 else 0.0
        next_start = start_sec + self.chunk_seconds

        src_audio = load_audio_chunk(path=path, sample_rate=DIFFUSION_SR, seconds=self.chunk_seconds, start_sec=start_sec).astype(np.float32)
        nxt_audio = load_audio_chunk(path=path, sample_rate=DIFFUSION_SR, seconds=self.chunk_seconds, start_sec=next_start).astype(np.float32)
        src_feat = _extract_download_features(
            src_audio,
            n_frames=self.max_frames,
            mel_min=self.mel_min,
            mel_max=self.mel_max,
            lab1_encoder=self.lab1_encoder,
        )
        nxt_feat = _extract_download_features(
            nxt_audio,
            n_frames=self.max_frames,
            mel_min=self.mel_min,
            mel_max=self.mel_max,
            lab1_encoder=self.lab1_encoder,
        )
        if self.aug_cfg is not None:
            src_mel_aug, src_cond_aug = _augment_mel_cond_pair(src_feat["mel"], src_feat["cond_feat"], rng=rng, cfg=self.aug_cfg)
            nxt_mel_aug, nxt_cond_aug = _augment_mel_cond_pair(nxt_feat["mel"], nxt_feat["cond_feat"], rng=rng, cfg=self.aug_cfg)
            src_feat["mel"] = src_mel_aug
            src_feat["cond_feat"] = src_cond_aug
            nxt_feat["mel"] = nxt_mel_aug
            nxt_feat["cond_feat"] = nxt_cond_aug

        source_genre_name = str(row["source_genre"])
        source_genre_idx = int(self.genre_to_idx.get(source_genre_name, self.genre_to_idx.get("cc0_other", 0)))
        target_genres = [g for g in self.by_genre.keys() if g != source_genre_idx]
        target_genre = int(rng.choice(target_genres if target_genres else list(self.by_genre.keys())))
        target_real_idx = int(rng.choice(self.by_genre[target_genre]))
        target_audio = self._target_audio(target_real_idx)
        target_z_style = np.asarray(self.cache_arrays["z_style"][target_real_idx], dtype=np.float32).copy()
        return {
            "src_mel": torch.from_numpy(src_feat["mel"]),
            "src_cond_feat": torch.from_numpy(src_feat["cond_feat"]),
            "src_z_content": torch.from_numpy(src_feat["z_content"]),
            "src_z_style": torch.from_numpy(src_feat["z_style"]),
            "src_genre_idx": torch.tensor(source_genre_idx, dtype=torch.long),
            "nxt_mel": torch.from_numpy(nxt_feat["mel"]),
            "nxt_cond_feat": torch.from_numpy(nxt_feat["cond_feat"]),
            "nxt_z_content": torch.from_numpy(nxt_feat["z_content"]),
            "nxt_z_style": torch.from_numpy(nxt_feat["z_style"]),
            "target_genre_idx": torch.tensor(target_genre, dtype=torch.long),
            "target_z_style": torch.from_numpy(target_z_style),
            "target_audio_mert": torch.from_numpy(target_audio),
        }


class MixedSourceAdjacentDataset(Dataset):
    def __init__(
        self,
        cache_ds: AdjacentChunkStyleSwapDataset,
        downloads_ds: DownloadsSourceAdjacentDataset,
        *,
        samples_per_epoch: int,
        downloads_mix_ratio: float,
        seed: int = 328,
    ):
        self.cache_ds = cache_ds
        self.downloads_ds = downloads_ds
        self.samples_per_epoch = max(1, int(samples_per_epoch))
        self.downloads_mix_ratio = float(np.clip(downloads_mix_ratio, 0.0, 1.0))
        self.base_seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.cache_ds.set_epoch(epoch)
        self.downloads_ds.set_epoch(epoch)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.base_seed + self.epoch * 1_000_003 + int(item))
        use_downloads = bool(rng.random() < self.downloads_mix_ratio)
        if use_downloads:
            sub_idx = int(rng.integers(0, len(self.downloads_ds)))
            return self.downloads_ds[sub_idx]
        sub_idx = int(rng.integers(0, len(self.cache_ds)))
        return self.cache_ds[sub_idx]


def _batch_v_to_x0(schedule: NoiseSchedule, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    s_a = schedule.sqrt_alphas_cumprod[t][:, None, None, None]
    s_om = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
    return torch.clamp(s_a * x_t - s_om * v, -1.0, 1.0)


def _mel_norm_to_audio_tensor(
    mel_norm: torch.Tensor,
    *,
    mel_min: float,
    mel_max: float,
    vocoder,
) -> torch.Tensor:
    log_mel = denormalize_mel(mel_norm.squeeze(1), mel_min, mel_max)
    audio = vocoder(log_mel)
    if audio.ndim == 3:
        audio = audio.squeeze(1)
    return audio


def _build_style_centroids(arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[int, np.ndarray]:
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    z_style = np.asarray(arrays["z_style"], dtype=np.float32)
    centroids: Dict[int, np.ndarray] = {}
    for g in sorted(np.unique(genre_idx[indices]).tolist()):
        rows = indices[genre_idx[indices] == int(g)]
        if len(rows) == 0:
            continue
        cent = z_style[rows].mean(axis=0).astype(np.float32)
        cent = cent / (np.linalg.norm(cent) + 1e-8)
        centroids[int(g)] = cent
    return centroids


def _genre_name_maps(genre_to_idx: Dict[str, int]) -> Tuple[Dict[str, int], Dict[int, str]]:
    idx_to_genre = {int(v): str(k) for k, v in genre_to_idx.items()}
    genre_to_idx_norm = {str(k): int(v) for k, v in genre_to_idx.items()}
    return genre_to_idx_norm, idx_to_genre


def _extract_download_features(
    audio: np.ndarray,
    *,
    n_frames: int,
    mel_min: float,
    mel_max: float,
    lab1_encoder: FrozenLab1Encoder,
) -> Dict[str, np.ndarray]:
    import librosa

    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=audio,
            sr=DIFFUSION_SR,
            n_fft=1024,
            hop_length=DIFFUSION_HOP,
            n_mels=80,
            fmin=0.0,
            fmax=8000.0,
            power=2.0,
        ),
        ref=1.0,
    ).astype(np.float32)
    mel = mel[:, : int(n_frames)]
    if mel.shape[1] < int(n_frames):
        pad = np.full((mel.shape[0], int(n_frames) - mel.shape[1]), float(mel_min), dtype=np.float32)
        mel = np.concatenate([mel, pad], axis=1)

    chroma = librosa.feature.chroma_cqt(y=audio, sr=DIFFUSION_SR).astype(np.float32)
    chroma = chroma[:, : int(n_frames)]
    if chroma.shape[1] < int(n_frames):
        chroma = np.pad(chroma, ((0, 0), (0, int(n_frames) - chroma.shape[1])))

    onset = librosa.onset.onset_strength(y=audio, sr=DIFFUSION_SR, hop_length=DIFFUSION_HOP).astype(np.float32)
    onset = onset[: int(n_frames)]
    if onset.shape[0] < int(n_frames):
        onset = np.pad(onset, (0, int(n_frames) - onset.shape[0]))

    tempo, beats = librosa.beat.beat_track(y=audio, sr=DIFFUSION_SR, hop_length=DIFFUSION_HOP)
    beat = np.zeros((int(n_frames),), dtype=np.float32)
    if len(np.atleast_1d(beats)) > 0:
        beats = np.asarray(beats, dtype=np.int64)
        beats = beats[(beats >= 0) & (beats < int(n_frames))]
        beat[beats] = 1.0
    _ = tempo

    mel_norm = _normalize_mel_np(mel, mel_min=float(mel_min), mel_max=float(mel_max))
    H = mel_norm.shape[0]
    chroma_exp = np.repeat(chroma[:, None, :], H, axis=1)
    onset_exp = np.repeat(onset[None, None, :], H, axis=1)
    beat_exp = np.repeat(beat[None, None, :], H, axis=1)
    cond_feat = np.concatenate([chroma_exp, onset_exp, beat_exp], axis=0).astype(np.float32)

    log_mel = extract_log_mel(audio, sr=lab1_encoder.cfg.sample_rate)
    log_mel = fix_log_mel_frames(log_mel, n_frames=int(n_frames))
    lat = lab1_encoder.infer_log_mel(log_mel)
    return {
        "mel": mel_norm[None, :, :].astype(np.float32),
        "cond_feat": cond_feat,
        "z_content": lat["z_content"].astype(np.float32),
        "z_style": lat["z_style"].astype(np.float32),
    }


def _prepare_epoch_monitor_plan(
    *,
    cfg: RetoolConfig,
    train_ds: Optional[AdjacentChunkStyleSwapDataset],
    style_centroids: Dict[int, np.ndarray],
    idx_to_genre: Dict[int, str],
    out_dir: Path,
) -> Dict[str, Any]:
    plan_path = out_dir / "epoch_sample_plan.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text(encoding="utf-8"))

    train_monitors: List[Dict[str, Any]] = []
    rng = random.Random(int(cfg.seed) + 701)
    if train_ds is not None:
        train_indices = list(range(len(train_ds)))
        rng.shuffle(train_indices)
        for sample_idx in train_indices[: max(0, int(cfg.epoch_train_samples))]:
            src_idx, _ = train_ds.pairs[int(sample_idx)]
            src_row = train_ds.index_df.iloc[int(src_idx)]
            src_genre = int(np.asarray(train_ds.arrays["genre_idx"], dtype=np.int64)[int(src_idx)])
            target_choices = [g for g in style_centroids.keys() if int(g) != int(src_genre)]
            target_genre = int(rng.choice(target_choices if target_choices else [src_genre]))
            train_monitors.append(
                {
                    "sample_idx": int(sample_idx),
                    "track_path": str(src_row["path"]),
                    "start_sec": float(src_row.get("start_sec", 0.0)),
                    "source_genre": idx_to_genre.get(int(src_genre), f"genre_{src_genre}"),
                    "target_genre": idx_to_genre.get(int(target_genre), f"genre_{target_genre}"),
                }
            )

    download_rows = ddb.discover_download_audio(Path(cfg.downloads_dir))
    min_seconds = float(cfg.max_frames * DIFFUSION_HOP / DIFFUSION_SR) + 1.0
    download_rows = [
        row
        for row in download_rows
        if float(row.get("duration_seconds") or 0.0) >= min_seconds and all(ord(ch) < 128 for ch in str(row["path"]))
    ]
    rng.shuffle(download_rows)
    download_monitors: List[Dict[str, Any]] = []
    for row in download_rows[: max(0, int(cfg.epoch_download_samples))]:
        source_audio = Path(str(row["path"]))
        duration = float(row.get("duration_seconds") or 0.0)
        max_start = max(0.0, duration - min_seconds - 0.1)
        start_sec = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
        source_genre = ddb.infer_source_genre(source_audio)
        target_candidates = [g for g in idx_to_genre.values() if g != source_genre]
        target_genre = rng.choice(target_candidates if target_candidates else list(idx_to_genre.values()))
        download_monitors.append(
            {
                "source_audio": str(source_audio),
                "start_sec": round(float(start_sec), 3),
                "source_genre": source_genre,
                "target_genre": target_genre,
            }
        )

    plan = {"train_monitors": train_monitors, "download_monitors": download_monitors}
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


@torch.no_grad()
def _generate_epoch_samples(
    *,
    epoch: int,
    cfg: RetoolConfig,
    out_dir: Path,
    ema: EMA,
    schedule: NoiseSchedule,
    train_ds: Optional[AdjacentChunkStyleSwapDataset],
    style_centroids: Dict[int, np.ndarray],
    genre_to_idx: Dict[str, int],
    idx_to_genre: Dict[int, str],
    mel_min: float,
    mel_max: float,
    device: torch.device,
    vocoder,
) -> None:
    sample_dir = out_dir / "epoch_samples" / f"epoch_{epoch + 1:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    plan = _prepare_epoch_monitor_plan(
        cfg=cfg,
        train_ds=train_ds,
        style_centroids=style_centroids,
        idx_to_genre=idx_to_genre,
        out_dir=out_dir,
    )
    plan_out = sample_dir / "plan.json"
    plan_out.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    style_strength = float(np.clip(float(cfg.epoch_sample_style_strength), 0.0, 1.0))
    ddim_steps = int(cfg.epoch_sample_ddim_steps)
    t_start = int(cfg.epoch_sample_t_start)
    guidance_scale = float(cfg.epoch_sample_guidance_scale)
    eta = float(cfg.epoch_sample_eta)

    train_dir = sample_dir / "train_cache"
    train_dir.mkdir(parents=True, exist_ok=True)
    if train_ds is not None:
        for i, mon in enumerate(plan.get("train_monitors", [])):
            batch = train_ds[int(mon["sample_idx"])]
            src_mel = batch["src_mel"].unsqueeze(0).to(device)
            cond_feat = batch["src_cond_feat"].unsqueeze(0).to(device)
            z_content = batch["src_z_content"].unsqueeze(0).to(device)
            z_style_src = batch["src_z_style"].unsqueeze(0).to(device)
            tgt_name = str(mon["target_genre"])
            tgt_idx = int(genre_to_idx.get(tgt_name, int(batch["target_genre_idx"].item())))
            z_style_tgt = torch.from_numpy(style_centroids[tgt_idx]).unsqueeze(0).to(device)
            z_style_mix = F.normalize((1.0 - style_strength) * z_style_src + style_strength * z_style_tgt, dim=-1)
            mel_gen = ddim_sample_v2_constrained(
                ema.shadow,
                schedule,
                cond_feat,
                z_content,
                z_style_mix,
                source_mel=src_mel,
                t_start=t_start,
                prefix_x0=None,
                prefix_frames=0,
                n_steps=ddim_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                device=device,
            )
            wav_src = vocode_bigvgan(src_mel, float(mel_min), float(mel_max), vocoder, device)[0]
            wav_gen = vocode_bigvgan(mel_gen, float(mel_min), float(mel_max), vocoder, device)[0]
            prefix = f"{i:02d}_{Path(str(mon['track_path'])).stem[:40]}"
            sf.write(str(train_dir / f"{prefix}_source.wav"), wav_src, DIFFUSION_SR)
            sf.write(str(train_dir / f"{prefix}_to_{tgt_name}.wav"), wav_gen, DIFFUSION_SR)

    download_dir = sample_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    lab1 = FrozenLab1Encoder(Path(cfg.lab1_checkpoint), device=str(device))
    n_frames = int(cfg.max_frames)
    for i, mon in enumerate(plan.get("download_monitors", [])):
        source_audio = Path(str(mon["source_audio"]))
        y = load_audio_chunk(
            path=source_audio,
            sample_rate=DIFFUSION_SR,
            seconds=float(cfg.max_frames * DIFFUSION_HOP / DIFFUSION_SR),
            start_sec=float(mon["start_sec"]),
        )
        feats = _extract_download_features(
            y,
            n_frames=n_frames,
            mel_min=float(mel_min),
            mel_max=float(mel_max),
            lab1_encoder=lab1,
        )
        mel_src = torch.from_numpy(feats["mel"]).unsqueeze(0).to(device)
        cond_feat = torch.from_numpy(feats["cond_feat"]).unsqueeze(0).to(device)
        z_content = torch.from_numpy(feats["z_content"]).unsqueeze(0).to(device)
        z_style_src = torch.from_numpy(feats["z_style"]).unsqueeze(0).to(device)
        tgt_name = str(mon["target_genre"])
        tgt_idx = int(genre_to_idx[tgt_name])
        z_style_tgt = torch.from_numpy(style_centroids[tgt_idx]).unsqueeze(0).to(device)
        z_style_mix = F.normalize((1.0 - style_strength) * z_style_src + style_strength * z_style_tgt, dim=-1)
        mel_gen = ddim_sample_v2_constrained(
            ema.shadow,
            schedule,
            cond_feat,
            z_content,
            z_style_mix,
            source_mel=mel_src,
            t_start=t_start,
            prefix_x0=None,
            prefix_frames=0,
            n_steps=ddim_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            device=device,
        )
        wav_gen = vocode_bigvgan(mel_gen, float(mel_min), float(mel_max), vocoder, device)[0]
        prefix = f"{i:02d}_{source_audio.stem[:40]}"
        sf.write(str(download_dir / f"{prefix}_source.wav"), y, DIFFUSION_SR)
        sf.write(str(download_dir / f"{prefix}_to_{tgt_name}.wav"), wav_gen, DIFFUSION_SR)

    del lab1
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _time_delta(x: torch.Tensor) -> torch.Tensor:
    return x[..., 1:] - x[..., :-1]


def _time_delta2(x: torch.Tensor) -> torch.Tensor:
    return x[..., 2:] - 2.0 * x[..., 1:-1] + x[..., :-2]


def _compute_losses(
    *,
    model,
    schedule: NoiseSchedule,
    batch: Dict[str, torch.Tensor],
    target_z_style: torch.Tensor,
    optimizer_step: int,
    cfg: RetoolConfig,
    device: torch.device,
    mert: FrozenMERT,
    vocoder,
    mel_min: float,
    mel_max: float,
) -> Dict[str, torch.Tensor]:
    src_mel = batch["src_mel"].to(device)
    src_cond = batch["src_cond_feat"].to(device)
    src_zc = batch["src_z_content"].to(device)
    src_zs = batch["src_z_style"].to(device)
    nxt_mel = batch["nxt_mel"].to(device)
    nxt_cond = batch["nxt_cond_feat"].to(device)
    nxt_zc = batch["nxt_z_content"].to(device)
    nxt_zs = batch["nxt_z_style"].to(device)
    target_audio = batch["target_audio_mert"]

    B = src_mel.shape[0]
    t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)

    noise_src = torch.randn_like(src_mel)
    noise_nxt = torch.randn_like(nxt_mel)
    x_t_src = schedule.q_sample(src_mel, t, noise_src)
    x_t_nxt = schedule.q_sample(nxt_mel, t, noise_nxt)
    v_target_src = schedule.compute_v_target(src_mel, t, noise_src)
    v_target_nxt = schedule.compute_v_target(nxt_mel, t, noise_nxt)

    if cfg.cfg_dropout_p > 0:
        mask_content = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
        src_zc_id = src_zc * mask_content
        nxt_zc_id = nxt_zc * mask_content
    else:
        src_zc_id = src_zc
        nxt_zc_id = nxt_zc

    id_pred_src = model(torch.cat([x_t_src, src_cond], dim=1), t, src_zc_id, src_zs)
    id_pred_nxt = model(torch.cat([x_t_nxt, nxt_cond], dim=1), t, nxt_zc_id, nxt_zs)
    loss_identity = 0.5 * (F.mse_loss(id_pred_src, v_target_src) + F.mse_loss(id_pred_nxt, v_target_nxt))

    swap_pred_src = model(torch.cat([x_t_src, src_cond], dim=1), t, src_zc, target_z_style)
    swap_pred_nxt = model(torch.cat([x_t_nxt, nxt_cond], dim=1), t, nxt_zc, target_z_style)
    x0_swap_src = _batch_v_to_x0(schedule, x_t_src, t, swap_pred_src)
    x0_swap_nxt = _batch_v_to_x0(schedule, x_t_nxt, t, swap_pred_nxt)

    anchor_bins = max(8, min(int(cfg.anchor_bins), src_mel.shape[2]))
    loss_anchor = F.l1_loss(x0_swap_src[:, :, :anchor_bins, :], src_mel[:, :, :anchor_bins, :])
    loss_envelope = F.l1_loss(x0_swap_src.mean(dim=2), src_mel.mean(dim=2))

    vocal_lo = max(0, min(int(cfg.vocal_start_bin), x0_swap_src.shape[2] - 1))
    vocal_hi = max(vocal_lo + 1, min(int(cfg.vocal_end_bin), x0_swap_src.shape[2]))
    src_vocal = src_mel[:, :, vocal_lo:vocal_hi, :]
    gen_vocal = x0_swap_src[:, :, vocal_lo:vocal_hi, :]
    loss_vocal_env = F.l1_loss(gen_vocal.mean(dim=2), src_vocal.mean(dim=2))
    loss_vocal_delta = F.l1_loss(_time_delta(gen_vocal.mean(dim=2)), _time_delta(src_vocal.mean(dim=2)))
    loss_vocal = loss_vocal_env + 0.5 * loss_vocal_delta

    ov = max(4, min(int(cfg.overlap_frames), x0_swap_src.shape[-1] // 2))
    tail = x0_swap_src[:, :, :, -ov:]
    head = x0_swap_nxt[:, :, :, :ov]
    loss_cont_env = F.l1_loss(tail.mean(dim=2), head.mean(dim=2))
    loss_cont_std = F.l1_loss(tail.std(dim=2), head.std(dim=2))
    hf_bin = max(0, min(int(cfg.hf_start_bin), x0_swap_src.shape[2] - 1))
    loss_cont_hf = F.l1_loss(tail[:, :, hf_bin:, :].mean(dim=2), head[:, :, hf_bin:, :].mean(dim=2))
    loss_continuity = loss_cont_env + 0.5 * loss_cont_std + 0.5 * loss_cont_hf

    src_hf = src_mel[:, :, hf_bin:, :].mean(dim=(2, 3))
    gen_hf = x0_swap_src[:, :, hf_bin:, :].mean(dim=(2, 3))
    loss_hf = F.relu(gen_hf - src_hf - float(cfg.hf_margin)).mean()

    gen_hf_band = x0_swap_src[:, :, hf_bin:, :]
    src_hf_band = src_mel[:, :, hf_bin:, :]
    gen_jitter = _time_delta2(gen_hf_band).abs().mean(dim=(2, 3))
    src_jitter = _time_delta2(src_hf_band).abs().mean(dim=(2, 3))
    loss_crackle = F.relu(gen_jitter - src_jitter - float(cfg.crackle_margin)).mean()

    if int(cfg.style_every_steps) <= 1 or (optimizer_step % int(cfg.style_every_steps) == 0):
        style_probe_frames = max(32, min(int(cfg.style_probe_frames), int(x0_swap_src.shape[-1])))
        if style_probe_frames < int(x0_swap_src.shape[-1]):
            start_frame = max(0, (int(x0_swap_src.shape[-1]) - style_probe_frames) // 2)
            end_frame = start_frame + style_probe_frames
            mel_style = x0_swap_src[..., start_frame:end_frame]
        else:
            mel_style = x0_swap_src
        gen_audio = _mel_norm_to_audio_tensor(
            mel_style,
            mel_min=mel_min,
            mel_max=mel_max,
            vocoder=vocoder,
        )
        target_audio_style = target_audio
        target_samples = int(gen_audio.shape[-1])
        if target_audio_style.shape[-1] > target_samples:
            target_audio_style = target_audio_style[..., :target_samples]
        elif target_audio_style.shape[-1] < target_samples:
            target_audio_style = F.pad(target_audio_style, (0, target_samples - target_audio_style.shape[-1]))
        split_count = max(1, min(int(cfg.style_batch_splits), int(gen_audio.shape[0])))
        chunk_size = max(1, int(np.ceil(float(gen_audio.shape[0]) / float(split_count))))
        loss_style = torch.zeros((), device=device)
        chunk_terms = 0
        for start in range(0, int(gen_audio.shape[0]), chunk_size):
            end = min(int(gen_audio.shape[0]), start + chunk_size)
            with torch.no_grad():
                tgt_feat_chunk = F.normalize(
                    mert.extract_features_batch_tensor(
                        target_audio_style[start:end].to(device), sample_rate=int(mert.cfg.sample_rate)
                    ),
                    dim=-1,
                )
            gen_feat_chunk = F.normalize(
                mert.extract_features_batch_tensor(gen_audio[start:end], sample_rate=DIFFUSION_SR),
                dim=-1,
            )
            loss_style = loss_style + (1.0 - (gen_feat_chunk * tgt_feat_chunk).sum(dim=-1)).mean()
            chunk_terms += 1
            del tgt_feat_chunk, gen_feat_chunk
        loss_style = loss_style / max(1, chunk_terms)
        del gen_audio
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        loss_style = torch.zeros((), device=device)

    total = (
        float(cfg.identity_weight) * loss_identity
        + float(cfg.style_weight) * loss_style
        + float(cfg.anchor_weight) * loss_anchor
        + float(cfg.envelope_weight) * loss_envelope
        + float(cfg.continuity_weight) * loss_continuity
        + float(cfg.hf_penalty_weight) * loss_hf
        + float(cfg.vocal_weight) * loss_vocal
        + float(cfg.crackle_weight) * loss_crackle
    )
    return {
        "loss_total": total,
        "loss_identity": loss_identity.detach(),
        "loss_style": loss_style.detach(),
        "loss_anchor": loss_anchor.detach(),
        "loss_envelope": loss_envelope.detach(),
        "loss_continuity": loss_continuity.detach(),
        "loss_hf": loss_hf.detach(),
        "loss_vocal": loss_vocal.detach(),
        "loss_crackle": loss_crackle.detach(),
    }


@torch.no_grad()
def evaluate_identity_loss(
    *,
    model,
    schedule: NoiseSchedule,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 8,
) -> float:
    model.eval()
    losses: List[float] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= int(max_batches):
            break
        mel = batch["src_mel"].to(device)
        cond = batch["src_cond_feat"].to(device)
        zc = batch["src_z_content"].to(device)
        zs = batch["src_z_style"].to(device)
        B = mel.shape[0]
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(mel)
        x_t = schedule.q_sample(mel, t, noise)
        v_target = schedule.compute_v_target(mel, t, noise)
        pred = model(torch.cat([x_t, cond], dim=1), t, zc, zs)
        losses.append(float(F.mse_loss(pred, v_target).item()))
    return float(np.mean(losses)) if losses else float("nan")


def _save_checkpoint(
    path: Path,
    *,
    model,
    ema: EMA,
    optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    batch_idx_in_epoch: int = 0,
    scheduler: Optional[object] = None,
    history: Optional[List[Dict[str, float]]] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_val_loss),
        "batch_idx_in_epoch": int(batch_idx_in_epoch),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    if history is not None:
        ckpt["history"] = history
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, str(path))


def train_retooled_diffusion(cfg: RetoolConfig) -> Dict[str, str]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = _device_from_arg(cfg.device)
    print(f"device={device}")

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    track_ids = index_df["track_id"].to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, track_ids, val_ratio=0.1, seed=int(cfg.seed))
    print(f"cache={cfg.cache_dir} train={len(train_idx)} val={len(val_idx)} frames={cfg.max_frames}")
    genre_to_idx_norm, idx_to_genre = _genre_name_maps(genre_to_idx)

    mert = FrozenMERT(
        model_id=str(cfg.mert_model_id),
        chunk_seconds=float(cfg.max_frames * DIFFUSION_HOP / DIFFUSION_SR),
        device=str(device),
        layer=int(cfg.mert_layer),
    )
    vocoder = load_bigvgan_robust(device=device)
    needs_lab1_runtime = str(cfg.source_mode).lower() in {"downloads", "mixed"} or int(cfg.epoch_download_samples) > 0
    lab1_runtime = FrozenLab1Encoder(Path(cfg.lab1_checkpoint), device=str(device)) if needs_lab1_runtime else None

    if str(cfg.source_mode).lower() == "downloads":
        if lab1_runtime is None:
            raise RuntimeError("Downloads source mode requires Lab1 runtime encoder.")
        train_ds = DownloadsSourceAdjacentDataset(
            downloads_dir=Path(cfg.downloads_dir),
            cache_arrays=arrays,
            cache_index_df=index_df,
            cache_indices=train_idx,
            genre_to_idx=genre_to_idx_norm,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            mert_sample_rate=int(mert.cfg.sample_rate),
            lab1_encoder=lab1_runtime,
            samples_per_epoch=int(cfg.downloads_source_samples_per_epoch),
            aug_cfg=cfg,
            seed=int(cfg.seed),
        )
    elif str(cfg.source_mode).lower() == "mixed":
        if lab1_runtime is None:
            raise RuntimeError("Mixed source mode requires Lab1 runtime encoder.")
        cache_train_ds = AdjacentChunkStyleSwapDataset(
            arrays,
            index_df,
            train_idx,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            mert_sample_rate=int(mert.cfg.sample_rate),
            aug_cfg=cfg,
            seed=int(cfg.seed),
        )
        downloads_train_ds = DownloadsSourceAdjacentDataset(
            downloads_dir=Path(cfg.downloads_dir),
            cache_arrays=arrays,
            cache_index_df=index_df,
            cache_indices=train_idx,
            genre_to_idx=genre_to_idx_norm,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            mert_sample_rate=int(mert.cfg.sample_rate),
            lab1_encoder=lab1_runtime,
            samples_per_epoch=int(cfg.downloads_source_samples_per_epoch),
            aug_cfg=cfg,
            seed=int(cfg.seed) + 17,
        )
        train_ds = MixedSourceAdjacentDataset(
            cache_train_ds,
            downloads_train_ds,
            samples_per_epoch=int(cfg.mixed_source_samples_per_epoch),
            downloads_mix_ratio=float(cfg.downloads_mix_ratio),
            seed=int(cfg.seed),
        )
    else:
        train_ds = AdjacentChunkStyleSwapDataset(
            arrays,
            index_df,
            train_idx,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            mert_sample_rate=int(mert.cfg.sample_rate),
            aug_cfg=cfg,
            seed=int(cfg.seed),
        )
    val_ds = AdjacentChunkStyleSwapDataset(
        arrays,
        index_df,
        val_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        mert_sample_rate=int(mert.cfg.sample_rate),
        aug_cfg=None,
        seed=int(cfg.seed) + 1,
    )
    def _make_train_loader(epoch_seed: int) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(int(epoch_seed))
        return DataLoader(
            train_ds,
            batch_size=int(cfg.batch_size),
            shuffle=True,
            generator=gen,
            num_workers=int(cfg.num_workers),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=int(cfg.base_ch),
        ch_mults=tuple(int(x) for x in cfg.ch_mults),
        n_res=int(cfg.n_res),
        attn_levels=tuple(int(x) for x in cfg.attn_levels),
        z_content_dim=128,
        z_style_dim=128,
        dropout=float(cfg.dropout),
    ).to(device)
    ema = EMA(model, decay=float(cfg.ema_decay))
    schedule = NoiseSchedule(T=1000).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        betas=(0.9, 0.999),
        weight_decay=float(cfg.weight_decay),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    start_epoch = 0
    start_batch_idx = 0
    global_step = 0
    best_val = float("inf")
    latest_ckpt = ckpt_dir / "latest.pt"
    history: List[Dict[str, float]] = []

    steps_per_epoch_nominal = len(_make_train_loader(int(cfg.seed)))
    if int(cfg.max_batches_per_epoch) > 0:
        steps_per_epoch_nominal = min(steps_per_epoch_nominal, int(cfg.max_batches_per_epoch))
    total_steps = max(1, (steps_per_epoch_nominal // max(1, int(cfg.grad_accum))) * int(cfg.epochs))

    def _lr_lambda(step: int) -> float:
        if step < int(cfg.warmup_steps):
            return float(step + 1) / float(max(1, int(cfg.warmup_steps)))
        progress = float(step - int(cfg.warmup_steps)) / float(max(1, total_steps - int(cfg.warmup_steps)))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if bool(cfg.resume) and latest_ckpt.exists():
        payload = torch.load(str(latest_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        if payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload.get("epoch", 0))
        start_batch_idx = int(payload.get("batch_idx_in_epoch", 0))
        global_step = int(payload.get("global_step", 0))
        best_val = float(payload.get("best_loss", payload.get("best_val_loss", float("inf"))))
        history = list(payload.get("history", []))
        if payload.get("python_rng_state") is not None:
            random.setstate(payload["python_rng_state"])
        if payload.get("numpy_rng_state") is not None:
            np.random.set_state(payload["numpy_rng_state"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        print(f"resumed_from={latest_ckpt} epoch={start_epoch+1} batch={start_batch_idx} step={global_step}")
    elif cfg.bootstrap_checkpoint is not None:
        payload = torch.load(str(cfg.bootstrap_checkpoint), map_location=device, weights_only=False)
        if "model" not in payload or "ema" not in payload:
            raise ValueError(f"Bootstrap checkpoint missing model/ema: {cfg.bootstrap_checkpoint}")
        model.load_state_dict(payload["model"])
        ema.load_state_dict(payload["ema"])
        print(f"bootstrapped_from={cfg.bootstrap_checkpoint}")

    compat_cfg = {
        "cache_dir": str(cfg.cache_dir),
        "out_dir": str(cfg.out_dir),
        "base_ch": int(cfg.base_ch),
        "ch_mults": [int(x) for x in cfg.ch_mults],
        "n_res": int(cfg.n_res),
        "attn_levels": [int(x) for x in cfg.attn_levels],
        "dropout": float(cfg.dropout),
        "ema_decay": float(cfg.ema_decay),
        "max_frames": int(cfg.max_frames),
        "seed": int(cfg.seed),
    }
    (cfg.out_dir / "v2_config.json").write_text(json.dumps(compat_cfg, indent=2), encoding="utf-8")
    (cfg.out_dir / "retool_config.json").write_text(
        json.dumps(cfg.__dict__, indent=2, default=_json_default), encoding="utf-8"
    )
    accum = max(1, int(cfg.grad_accum))
    style_centroids = _build_style_centroids(arrays, train_idx)
    genre_to_idx_norm, idx_to_genre = _genre_name_maps(genre_to_idx)
    print(f"adjacent_pairs train={len(train_ds)} val={len(val_ds)} target_genres={sorted(style_centroids.keys())}")

    if cfg.dry_run:
        train_ds.set_epoch(0)
        sample_batch = next(iter(_make_train_loader(int(cfg.seed))))
        tgt = sample_batch["target_genre_idx"].to(device)
        tgt_z = torch.stack([torch.from_numpy(style_centroids[int(g.item())]) for g in tgt.cpu()], dim=0).to(device)
        losses = _compute_losses(
            model=model,
            schedule=schedule,
            batch=sample_batch,
            target_z_style=tgt_z,
            optimizer_step=0,
            cfg=cfg,
            device=device,
            mert=mert,
            vocoder=vocoder,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
        )
        print({k: float(v.item()) for k, v in losses.items()})
        return {"status": "dry_run_ok", "out_dir": str(cfg.out_dir)}

    for epoch in range(start_epoch, int(cfg.epochs)):
        train_ds.set_epoch(epoch)
        train_loader = _make_train_loader(int(cfg.seed) + epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = {
            "loss_total": 0.0,
            "loss_identity": 0.0,
            "loss_style": 0.0,
            "loss_anchor": 0.0,
            "loss_envelope": 0.0,
            "loss_continuity": 0.0,
            "loss_hf": 0.0,
            "loss_vocal": 0.0,
            "loss_crackle": 0.0,
        }
        n_batches = 0
        t0 = time.time()
        resume_batch_idx = int(start_batch_idx) if epoch == int(start_epoch) else 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx < resume_batch_idx:
                continue
            if int(cfg.max_batches_per_epoch) > 0 and batch_idx >= int(cfg.max_batches_per_epoch):
                break
            target_genre_idx = batch["target_genre_idx"]
            target_z_style = torch.stack(
                [torch.from_numpy(style_centroids[int(g.item())]) for g in target_genre_idx],
                dim=0,
            ).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                losses = _compute_losses(
                    model=model,
                    schedule=schedule,
                    batch=batch,
                    target_z_style=target_z_style,
                    optimizer_step=global_step,
                    cfg=cfg,
                    device=device,
                    mert=mert,
                    vocoder=vocoder,
                    mel_min=float(meta.mel_min),
                    mel_max=float(meta.mel_max),
                )
                loss = losses["loss_total"] / float(accum)

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            for key in epoch_totals:
                epoch_totals[key] += float(losses[key].item())
            n_batches += 1
            global_step += 1

            if (batch_idx + 1) % accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)
                scheduler.step()

            if int(cfg.monitor_steps) > 0 and global_step % int(cfg.monitor_steps) == 0:
                avg_total = epoch_totals["loss_total"] / max(1, n_batches)
                avg_style = epoch_totals["loss_style"] / max(1, n_batches)
                avg_cont = epoch_totals["loss_continuity"] / max(1, n_batches)
                avg_vocal = epoch_totals["loss_vocal"] / max(1, n_batches)
                avg_crackle = epoch_totals["loss_crackle"] / max(1, n_batches)
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch+1}] batch {batch_idx+1} "
                    f"loss={avg_total:.4f} style={avg_style:.4f} cont={avg_cont:.4f} "
                    f"vocal={avg_vocal:.4f} crackle={avg_crackle:.4f} lr={lr:.2e}"
                )

            if int(cfg.save_every_steps) > 0 and global_step % int(cfg.save_every_steps) == 0:
                _save_checkpoint(
                    latest_ckpt,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val,
                    batch_idx_in_epoch=batch_idx + 1,
                    scheduler=scheduler,
                    history=history,
                    scaler=scaler,
                )

        if n_batches == 0:
            raise RuntimeError("No batches were processed. Check max_batches_per_epoch / dataset setup.")

        if n_batches % accum != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            scheduler.step()

        val_id = evaluate_identity_loss(
            model=ema.shadow,
            schedule=schedule,
            loader=val_loader,
            device=device,
            max_batches=8,
        )
        avg = {k: v / max(1, n_batches) for k, v in epoch_totals.items()}
        avg["epoch"] = float(epoch + 1)
        avg["val_identity"] = float(val_id)
        avg["epoch_seconds"] = float(time.time() - t0)
        history.append(avg)
        (cfg.out_dir / "v2_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        _save_checkpoint(
            latest_ckpt,
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val,
            batch_idx_in_epoch=0,
            scheduler=scheduler,
            history=history,
            scaler=scaler,
        )
        _save_checkpoint(
            ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
            model=model,
            ema=ema,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val,
            batch_idx_in_epoch=0,
            scheduler=scheduler,
            history=history,
            scaler=scaler,
        )

        if np.isfinite(val_id) and float(val_id) < float(best_val):
            best_val = float(val_id)
            _save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_loss=best_val,
                batch_idx_in_epoch=0,
                scheduler=scheduler,
                history=history,
                scaler=scaler,
            )

        if isinstance(train_ds, AdjacentChunkStyleSwapDataset) or int(cfg.epoch_download_samples) > 0:
            try:
                _generate_epoch_samples(
                    epoch=epoch,
                    cfg=cfg,
                    out_dir=cfg.out_dir,
                    ema=ema,
                    schedule=schedule,
                    train_ds=train_ds if isinstance(train_ds, AdjacentChunkStyleSwapDataset) else None,
                    style_centroids=style_centroids,
                    genre_to_idx=genre_to_idx_norm,
                    idx_to_genre=idx_to_genre,
                    mel_min=float(meta.mel_min),
                    mel_max=float(meta.mel_max),
                    device=device,
                    vocoder=vocoder,
                )
            except Exception as e:
                print(f"[epoch sample gen failed: {e}]")

        print(
            f"epoch={epoch+1}/{cfg.epochs} total={avg['loss_total']:.4f} "
            f"style={avg['loss_style']:.4f} cont={avg['loss_continuity']:.4f} "
            f"vocal={avg['loss_vocal']:.4f} crackle={avg['loss_crackle']:.4f} "
            f"val_id={val_id:.5f} best={best_val:.5f}"
        )
        start_batch_idx = 0

    summary = {
        "status": "ok",
        "out_dir": str(cfg.out_dir),
        "latest_checkpoint": str(latest_ckpt),
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "history_path": str(cfg.out_dir / "v2_history.json"),
        "epoch_samples_dir": str(cfg.out_dir / "epoch_samples"),
    }
    (cfg.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    default_base = resolve_default_base_checkpoint()
    p = argparse.ArgumentParser(description="Longform-aware diffusion retool training.")
    p.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--bootstrap-checkpoint", type=Path, default=default_base)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--cfg-dropout-p", type=float, default=0.05)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=250)
    p.add_argument("--max-batches-per-epoch", type=int, default=0)
    p.add_argument("--seed", type=int, default=328)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--ch-mults", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--attn-levels", nargs="+", type=int, default=[2, 3])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--identity-weight", type=float, default=1.0)
    p.add_argument("--style-weight", type=float, default=1.8)
    p.add_argument("--anchor-weight", type=float, default=0.55)
    p.add_argument("--envelope-weight", type=float, default=0.30)
    p.add_argument("--continuity-weight", type=float, default=0.65)
    p.add_argument("--hf-penalty-weight", type=float, default=0.20)
    p.add_argument("--vocal-weight", type=float, default=0.40)
    p.add_argument("--crackle-weight", type=float, default=0.30)
    p.add_argument("--anchor-bins", type=int, default=44)
    p.add_argument("--hf-start-bin", type=int, default=56)
    p.add_argument("--vocal-start-bin", type=int, default=10)
    p.add_argument("--vocal-end-bin", type=int, default=42)
    p.add_argument("--overlap-frames", type=int, default=40)
    p.add_argument("--hf-margin", type=float, default=0.06)
    p.add_argument("--crackle-margin", type=float, default=0.015)
    p.add_argument("--style-probe-frames", type=int, default=256)
    p.add_argument("--style-every-steps", type=int, default=1)
    p.add_argument("--style-batch-splits", type=int, default=2)
    p.add_argument("--mert-model-id", type=str, default="m-a-p/MERT-v1-95M")
    p.add_argument("--mert-layer", type=int, default=-1)
    p.add_argument("--monitor-steps", type=int, default=25)
    p.add_argument("--save-every-steps", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)
    p.add_argument("--lab1-checkpoint", type=Path, default=REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt")
    p.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--epoch-train-samples", type=int, default=3)
    p.add_argument("--epoch-download-samples", type=int, default=3)
    p.add_argument("--epoch-sample-ddim-steps", type=int, default=50)
    p.add_argument("--epoch-sample-t-start", type=int, default=260)
    p.add_argument("--epoch-sample-guidance-scale", type=float, default=1.9)
    p.add_argument("--epoch-sample-style-strength", type=float, default=0.62)
    p.add_argument("--epoch-sample-eta", type=float, default=0.0)
    p.add_argument("--source-mode", type=str, default="cache", choices=["cache", "downloads", "mixed"])
    p.add_argument("--downloads-source-samples-per-epoch", type=int, default=2000)
    p.add_argument("--mixed-source-samples-per-epoch", type=int, default=2000)
    p.add_argument("--downloads-mix-ratio", type=float, default=0.30)
    p.add_argument("--source-aug-prob", type=float, default=0.65)
    p.add_argument("--source-noise-std", type=float, default=0.015)
    p.add_argument("--source-cond-noise-std", type=float, default=0.01)
    p.add_argument("--source-global-offset-std", type=float, default=0.05)
    p.add_argument("--source-hf-tilt-std", type=float, default=0.07)
    p.add_argument("--source-time-mask-prob", type=float, default=0.30)
    p.add_argument("--source-time-mask-frames", type=int, default=24)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = RetoolConfig(
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        bootstrap_checkpoint=Path(args.bootstrap_checkpoint) if args.bootstrap_checkpoint else None,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        max_frames=int(args.max_frames),
        lr=float(args.lr),
        ema_decay=float(args.ema_decay),
        cfg_dropout_p=float(args.cfg_dropout_p),
        grad_clip_norm=float(args.grad_clip_norm),
        weight_decay=float(args.weight_decay),
        warmup_steps=int(args.warmup_steps),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        base_ch=int(args.base_ch),
        ch_mults=tuple(int(x) for x in args.ch_mults),
        n_res=int(args.n_res),
        attn_levels=tuple(int(x) for x in args.attn_levels),
        dropout=float(args.dropout),
        identity_weight=float(args.identity_weight),
        style_weight=float(args.style_weight),
        anchor_weight=float(args.anchor_weight),
        envelope_weight=float(args.envelope_weight),
        continuity_weight=float(args.continuity_weight),
        hf_penalty_weight=float(args.hf_penalty_weight),
        vocal_weight=float(args.vocal_weight),
        crackle_weight=float(args.crackle_weight),
        anchor_bins=int(args.anchor_bins),
        hf_start_bin=int(args.hf_start_bin),
        vocal_start_bin=int(args.vocal_start_bin),
        vocal_end_bin=int(args.vocal_end_bin),
        overlap_frames=int(args.overlap_frames),
        hf_margin=float(args.hf_margin),
        crackle_margin=float(args.crackle_margin),
        style_probe_frames=int(args.style_probe_frames),
        style_every_steps=int(args.style_every_steps),
        style_batch_splits=int(args.style_batch_splits),
        mert_model_id=str(args.mert_model_id),
        mert_layer=int(args.mert_layer),
        monitor_steps=int(args.monitor_steps),
        save_every_steps=int(args.save_every_steps),
        resume=bool(args.resume),
        lab1_checkpoint=Path(args.lab1_checkpoint),
        downloads_dir=Path(args.downloads_dir),
        epoch_train_samples=int(args.epoch_train_samples),
        epoch_download_samples=int(args.epoch_download_samples),
        epoch_sample_ddim_steps=int(args.epoch_sample_ddim_steps),
        epoch_sample_t_start=int(args.epoch_sample_t_start),
        epoch_sample_guidance_scale=float(args.epoch_sample_guidance_scale),
        epoch_sample_style_strength=float(args.epoch_sample_style_strength),
        epoch_sample_eta=float(args.epoch_sample_eta),
        source_mode=str(args.source_mode),
        downloads_source_samples_per_epoch=int(args.downloads_source_samples_per_epoch),
        mixed_source_samples_per_epoch=int(args.mixed_source_samples_per_epoch),
        downloads_mix_ratio=float(args.downloads_mix_ratio),
        source_aug_prob=float(args.source_aug_prob),
        source_noise_std=float(args.source_noise_std),
        source_cond_noise_std=float(args.source_cond_noise_std),
        source_global_offset_std=float(args.source_global_offset_std),
        source_hf_tilt_std=float(args.source_hf_tilt_std),
        source_time_mask_prob=float(args.source_time_mask_prob),
        source_time_mask_frames=int(args.source_time_mask_frames),
        device=str(args.device),
        dry_run=bool(args.dry_run),
    )
    summary = train_retooled_diffusion(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
