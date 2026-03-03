from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class ChunkEncoder(nn.Module):
    """Lab 1-compatible encoder architecture for checkpoint loading."""

    def __init__(self, n_sources: int, z_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.shared = nn.Linear(128, 256)
        self.content_head = nn.Linear(256, z_dim)
        self.style_head = nn.Linear(256, z_dim)
        self.style_cls = nn.Linear(z_dim, n_sources)
        self.content_style_adv = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, n_sources),
        )
        self.music_head = nn.Linear(256, 1)

    def forward(self, log_mel: torch.Tensor, grl_lambda: float = 1.0):
        x = log_mel.unsqueeze(1)
        h = self.backbone(x).flatten(1)
        h = F.relu(self.shared(h))
        z_content = F.normalize(self.content_head(h), dim=-1)
        z_style = F.normalize(self.style_head(h), dim=-1)
        z_content_rev = grad_reverse(z_content, lambda_=grl_lambda)
        return {
            "z_content": z_content,
            "z_style": z_style,
            "style_logits": self.style_cls(z_style),
            "content_style_logits": self.content_style_adv(z_content_rev),
            "music_logit": self.music_head(h).squeeze(-1),
        }


def load_audio_chunk(path: Path, sample_rate: int, seconds: float, start_sec: float = 0.0) -> np.ndarray:
    y, _ = librosa.load(
        str(path),
        sr=int(sample_rate),
        mono=True,
        offset=max(0.0, float(start_sec)),
        duration=float(seconds),
        dtype=np.float32,
        res_type="soxr_hq",
    )
    target_len = int(round(sample_rate * seconds))
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    if len(y) == 0:
        raise ValueError(f"Empty audio after load: {path}")
    y = librosa.util.normalize(y)
    return y.astype(np.float32)


def extract_log_mel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=96,
        fmin=20,
        fmax=sr // 2,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def descriptor32_from_logmel(log_mel: np.ndarray) -> np.ndarray:
    """32-D handcrafted descriptor from 16 mel bands (mean + std)."""
    if log_mel.ndim != 2 or log_mel.shape[0] < 16:
        raise ValueError(f"Unexpected log_mel shape: {log_mel.shape}")
    band_idx = np.linspace(0, log_mel.shape[0] - 1, 16, dtype=np.int64)
    band = log_mel[band_idx, :]
    mu = band.mean(axis=1)
    sigma = band.std(axis=1)
    d = np.concatenate([mu, sigma], axis=0).astype(np.float32)
    # Per-vector normalization for stable scale across codecs/datasets.
    d = (d - d.mean()) / (d.std() + 1e-8)
    return d


@dataclass
class EncoderConfig:
    sample_rate: int
    chunk_seconds: float
    z_dim: int


class FrozenLab1Encoder:
    def __init__(self, checkpoint_path: Path, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not hasattr(torch, "_utils"):
            torch._utils = importlib.import_module("torch._utils")

        payload = torch.load(str(self.checkpoint_path), map_location="cpu")
        cfg = payload.get("cfg", {})
        source_to_idx = payload.get("source_to_idx", {})
        if not source_to_idx:
            raise ValueError("Checkpoint missing source_to_idx")

        self.cfg = EncoderConfig(
            sample_rate=int(cfg.get("sample_rate", 22050)),
            chunk_seconds=float(cfg.get("chunk_seconds", 5.0)),
            z_dim=int(cfg.get("z_dim", 128)),
        )
        self.source_to_idx = source_to_idx

        self.model = ChunkEncoder(n_sources=len(source_to_idx), z_dim=self.cfg.z_dim).to(self.device)
        self.model.load_state_dict(payload["model"], strict=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def infer_file(self, path: Path, start_sec: float = 0.0) -> Optional[Dict[str, np.ndarray]]:
        path = Path(path)
        if not path.exists():
            return None

        y = load_audio_chunk(
            path=path,
            sample_rate=self.cfg.sample_rate,
            seconds=self.cfg.chunk_seconds,
            start_sec=start_sec,
        )
        log_mel = extract_log_mel(y, sr=self.cfg.sample_rate)
        x = torch.from_numpy(log_mel).unsqueeze(0).to(self.device, non_blocking=True)

        out = self.model(x, grl_lambda=0.0)
        z_content = out["z_content"][0].detach().cpu().numpy().astype(np.float32)
        z_style = out["z_style"][0].detach().cpu().numpy().astype(np.float32)
        music_prob = float(torch.sigmoid(out["music_logit"])[0].item())
        d32 = descriptor32_from_logmel(log_mel)
        target_160 = np.concatenate([z_style, d32], axis=0).astype(np.float32)

        return {
            "z_content": z_content,
            "z_style": z_style,
            "descriptor32": d32,
            "target160": target_160,
            "music_prob": np.asarray(music_prob, dtype=np.float32),
        }
