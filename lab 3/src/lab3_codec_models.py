from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, n_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, n_channels * 2),
            nn.SiLU(),
            nn.Linear(n_channels * 2, n_channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        ab = self.proj(cond)
        a, b = torch.chunk(ab, chunks=2, dim=1)
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        return x * (1.0 + a) + b


class ResidualFiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.film1 = FiLM(cond_dim=cond_dim, n_channels=channels)
        self.film2 = FiLM(cond_dim=cond_dim, n_channels=channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.film1(h, cond)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.film2(h, cond)
        h = F.silu(h)
        return x + h


class CodecLatentTranslator(nn.Module):
    """
    Maps source quantized codec latents to style-shifted latents while preserving content.
    """

    def __init__(
        self,
        in_channels: int = 128,
        z_content_dim: int = 128,
        z_style_dim: int = 128,
        hidden_channels: int = 256,
        n_blocks: int = 10,
        noise_dim: int = 32,
        residual_scale: float = 0.5,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.z_content_dim = int(z_content_dim)
        self.z_style_dim = int(z_style_dim)
        self.noise_dim = int(noise_dim)
        self.residual_scale = float(residual_scale)
        cond_dim = self.z_content_dim + self.z_style_dim + self.noise_dim

        self.in_proj = nn.Conv1d(self.in_channels, hidden_channels, kernel_size=3, padding=1)
        dilations = [1, 2, 4, 8, 1, 2, 4, 8, 1, 2]
        if int(n_blocks) != len(dilations):
            dilations = [2 ** (i % 4) for i in range(int(n_blocks))]
        self.blocks = nn.ModuleList(
            [
                ResidualFiLMBlock(channels=hidden_channels, cond_dim=cond_dim, dilation=int(d))
                for d in dilations
            ]
        )
        self.out_norm = nn.GroupNorm(num_groups=8, num_channels=hidden_channels)
        self.out_proj = nn.Conv1d(hidden_channels, self.in_channels, kernel_size=3, padding=1)

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.noise_dim, device=device)

    def forward(
        self,
        q_src: torch.Tensor,
        z_content: torch.Tensor,
        z_style_tgt: torch.Tensor,
        noise: torch.Tensor | None = None,
        style_dropout_p: float = 0.0,
        style_jitter_std: float = 0.0,
    ) -> torch.Tensor:
        b = int(q_src.shape[0])
        if noise is None:
            noise = torch.zeros((b, self.noise_dim), device=q_src.device, dtype=q_src.dtype)
        zs = z_style_tgt
        if style_dropout_p > 0.0:
            keep = (torch.rand((b, 1), device=q_src.device) >= float(style_dropout_p)).to(zs.dtype)
            zs = zs * keep
        if style_jitter_std > 0.0:
            zs = zs + torch.randn_like(zs) * float(style_jitter_std)
        cond = torch.cat([z_content, zs, noise], dim=1)

        h = self.in_proj(q_src)
        for blk in self.blocks:
            h = blk(h, cond)
        h = self.out_norm(h)
        h = F.silu(h)
        delta = torch.tanh(self.out_proj(h))
        return q_src + self.residual_scale * delta


class WaveDiscBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int, p: int):
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class SingleScaleWaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                WaveDiscBlock(1, 32, 15, 1, 7),
                WaveDiscBlock(32, 64, 41, 4, 20),
                WaveDiscBlock(64, 128, 41, 4, 20),
                WaveDiscBlock(128, 256, 41, 4, 20),
                WaveDiscBlock(256, 512, 41, 4, 20),
            ]
        )
        self.out = nn.utils.weight_norm(nn.Conv1d(512, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats: List[torch.Tensor] = []
        h = wav
        for blk in self.blocks:
            h = blk(h)
            feats.append(h)
        logits = self.out(h)
        return logits, feats


class MultiScaleWaveDiscriminator(nn.Module):
    def __init__(self, n_scales: int = 3):
        super().__init__()
        self.discs = nn.ModuleList([SingleScaleWaveDiscriminator() for _ in range(int(n_scales))])
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

    def forward(self, wav: torch.Tensor) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        outs: List[Tuple[torch.Tensor, List[torch.Tensor]]] = []
        x = wav
        for i, d in enumerate(self.discs):
            if i > 0:
                x = self.pool(x)
            outs.append(d(x))
        return outs


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


def feature_matching_loss(
    real_feats: Sequence[torch.Tensor],
    fake_feats: Sequence[torch.Tensor],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fake_feats[0].device)
    for a, b in zip(real_feats, fake_feats):
        loss = loss + F.l1_loss(a.detach(), b)
    return loss / max(1, len(fake_feats))


def multiscale_hinge_d_loss(
    real_outs: Sequence[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outs: Sequence[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=real_outs[0][0].device)
    for (real_logits, _), (fake_logits, _) in zip(real_outs, fake_outs):
        loss = loss + hinge_d_loss(real_logits, fake_logits)
    return loss / max(1, len(real_outs))


def multiscale_hinge_g_loss(
    fake_outs: Sequence[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fake_outs[0][0].device)
    for fake_logits, _ in fake_outs:
        loss = loss + hinge_g_loss(fake_logits)
    return loss / max(1, len(fake_outs))


def multiscale_feature_matching_loss(
    real_outs: Sequence[Tuple[torch.Tensor, List[torch.Tensor]]],
    fake_outs: Sequence[Tuple[torch.Tensor, List[torch.Tensor]]],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=fake_outs[0][0].device)
    for (_, rf), (_, ff) in zip(real_outs, fake_outs):
        loss = loss + feature_matching_loss(rf, ff)
    return loss / max(1, len(real_outs))


@dataclass
class CodecTrainWeights:
    adv: float = 0.5
    latent_l1: float = 4.0
    latent_continuity: float = 1.0
    content: float = 2.0
    style: float = 3.0
    mrstft: float = 2.0
    feature_match: float = 1.0
    mode_seeking: float = 0.0

