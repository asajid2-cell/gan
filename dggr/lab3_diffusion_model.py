"""Diffusion UNet for mel spectrogram generation with FiLM conditioning.

Architecture:
  - Input:  [B, 15, 80, 432]  (1 noisy mel + 12 chroma + 1 onset + 1 beat)
  - Output: [B, 1, 80, 432]   (predicted noise epsilon)
  - Global conditioning: z_content(128) + z_style(128) = 256D via FiLM
  - Time embedding: sinusoidal → MLP → FiLM
  - Channel progression: [64, 128, 256, 256]
  - Self-attention at lower resolutions (levels 2, 3)
  - EMA wrapper for stable generation

Skip connection convention (diffusers-style):
  - Down path stores: conv_in output + n_res outputs per level + 1 downsample per level
  - Up path: ALL levels consume n_res+1 skips (matching down path exactly)
  - Total skips = 1 + sum(n_res + has_downsample) per level
"""

from __future__ import annotations

import copy
import math
from typing import List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for beta_t.  Returns [T] float32."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class NoiseSchedule(nn.Module):
    """Pre-computed noise schedule for DDPM / DDIM."""

    def __init__(self, T: int = 1000):
        super().__init__()
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var)
        self.register_buffer("posterior_log_var_clipped", torch.log(torch.clamp(posterior_var, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.T = T

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise."""
        s_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s_one_m = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s_alpha * x0 + s_one_m * noise

    @torch.no_grad()
    def ddim_sample_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step.  t, t_prev are scalar timestep indices."""
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_prev - sigma ** 2, min=0.0)) * model_output
        noise = torch.randn_like(x_t) if eta > 0 and t_prev > 0 else torch.zeros_like(x_t)
        return torch.sqrt(alpha_prev) * pred_x0 + dir_xt + sigma * noise

    def compute_v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """v-prediction target: v = sqrt(alpha_bar)*noise - sqrt(1-alpha_bar)*x0."""
        s_a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        s_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return s_a * noise - s_om * x0

    def v_to_eps(self, x_t: torch.Tensor, t_scalar: int, v: torch.Tensor) -> torch.Tensor:
        """Convert v-prediction to epsilon for reuse with DDIM step."""
        s_a = self.sqrt_alphas_cumprod[t_scalar]
        s_om = self.sqrt_one_minus_alphas_cumprod[t_scalar]
        return s_om * x_t + s_a * v

    def v_to_x0(self, x_t: torch.Tensor, t_scalar: int, v: torch.Tensor) -> torch.Tensor:
        """Recover x0 from v-prediction."""
        s_a = self.sqrt_alphas_cumprod[t_scalar]
        s_om = self.sqrt_one_minus_alphas_cumprod[t_scalar]
        return s_a * x_t - s_om * v


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FiLMResBlock(nn.Module):
    """Residual block with FiLM conditioning.

    GroupNorm → SiLU → Conv → + FiLM(scale,shift) → GroupNorm → SiLU → Dropout → Conv → + skip
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).view(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(attn)


class Downsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class DiffusionUNet(nn.Module):
    """Conditional UNet for mel spectrogram diffusion.

    Skip convention:
      Down path stores one skip per ResBlock output, plus one per downsample output,
      plus the initial conv_in output.  All up levels have (n_res+1) ResBlocks to
      consume the matching number of skips.
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 1,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4, 4),
        n_res: int = 2,
        attn_levels: Tuple[int, ...] = (2, 3),
        z_content_dim: int = 128,
        z_style_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_levels = len(ch_mults)
        channels = [base_ch * m for m in ch_mults]
        cond_dim = base_ch * 4
        attn_set: Set[int] = set(attn_levels)

        # ---- Time + conditioning embeddings ----
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(z_content_dim + z_style_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # ---- Input ----
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Down path (flat lists) ----
        # For each level: n_res ResBlocks (with optional Attn each) + optional Downsample
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_chs: List[int] = [channels[0]]  # initial conv_in skip
        ch = channels[0]

        for level in range(n_levels):
            tgt = channels[level]
            for _ in range(n_res):
                self.down_res.append(FiLMResBlock(ch, tgt, cond_dim, dropout))
                ch = tgt
                self.down_attn.append(
                    SelfAttention2d(ch) if level in attn_set else nn.Identity()
                )
                skip_chs.append(ch)
            if level < n_levels - 1:
                self.downsamples.append(Downsample2d(ch))
                skip_chs.append(ch)
            else:
                self.downsamples.append(None)  # type: ignore[arg-type]

        # ---- Bottleneck ----
        self.mid1 = FiLMResBlock(ch, ch, cond_dim, dropout)
        self.mid_attn = SelfAttention2d(ch)
        self.mid2 = FiLMResBlock(ch, ch, cond_dim, dropout)

        # ---- Up path (flat lists) ----
        # ALL levels have (n_res + 1) ResBlocks to match the number of skips
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(n_levels)):
            tgt = channels[level]
            n_up = n_res + 1  # +1 for downsample skip (or initial skip for last up level)
            for _ in range(n_up):
                skip_ch = skip_chs.pop()
                self.up_res.append(FiLMResBlock(ch + skip_ch, tgt, cond_dim, dropout))
                ch = tgt
                self.up_attn.append(
                    SelfAttention2d(ch) if level in attn_set else nn.Identity()
                )
            if level > 0:
                self.upsamples.append(Upsample2d(ch))
            else:
                self.upsamples.append(None)  # type: ignore[arg-type]

        assert len(skip_chs) == 0, f"Skip channel mismatch: {len(skip_chs)} leftover"

        # ---- Output ----
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

        # Store structure info for forward pass
        self._n_levels = n_levels
        self._n_res = n_res

    def forward(
        self,
        x: torch.Tensor,         # [B, 15, 80, 432]
        t: torch.Tensor,          # [B] integer timesteps
        z_content: torch.Tensor,  # [B, z_content_dim]
        z_style: torch.Tensor,    # [B, z_style_dim]
    ) -> torch.Tensor:
        cond = self.time_emb(t) + self.z_proj(torch.cat([z_content, z_style], dim=-1))

        # ---- Down ----
        h = self.conv_in(x)
        skips = [h]

        res_i = 0
        for level in range(self._n_levels):
            for _ in range(self._n_res):
                h = self.down_res[res_i](h, cond)
                h = self.down_attn[res_i](h)
                res_i += 1
                skips.append(h)
            ds = self.downsamples[level]
            if ds is not None:
                h = ds(h)
                skips.append(h)

        # ---- Mid ----
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)

        # ---- Up ----
        res_i = 0
        up_i = 0
        for level in reversed(range(self._n_levels)):
            n_up = self._n_res + 1
            for _ in range(n_up):
                s = skips.pop()
                if h.shape[2:] != s.shape[2:]:
                    h = F.interpolate(h, size=s.shape[2:], mode="nearest")
                h = self.up_res[res_i](torch.cat([h, s], dim=1), cond)
                h = self.up_attn[res_i](h)
                res_i += 1
            us = self.upsamples[up_i]
            up_i += 1
            if us is not None:
                h = us(h)

        return self.conv_out(h)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# ---------------------------------------------------------------------------
# V2: StyleAdaIN + separate style conditioning + v-prediction support
# ---------------------------------------------------------------------------

class StyleAdaIN(nn.Module):
    """Adaptive Instance Normalization for dedicated style conditioning.

    Unlike FiLM (which mixes time+content+style), this gives style its own
    conditioning pathway via instance normalization + learned affine.
    """

    def __init__(self, channels: int, style_dim: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.proj = nn.Linear(style_dim, channels * 2)
        # Init to identity transform
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.proj(style).chunk(2, dim=-1)
        return h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class DiffusionUNetV2(nn.Module):
    """V2 UNet: separate style AdaIN + time/content FiLM.

    Key differences from V1:
      - time+content → FiLM (controls structure/melody)
      - style → dedicated AdaIN per block (controls timbre/texture)
      - Predicts v (velocity) instead of epsilon
    """

    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 1,
        base_ch: int = 64,
        ch_mults: Tuple[int, ...] = (1, 2, 4, 4),
        n_res: int = 2,
        attn_levels: Tuple[int, ...] = (2, 3),
        z_content_dim: int = 128,
        z_style_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        n_levels = len(ch_mults)
        channels = [base_ch * m for m in ch_mults]
        cond_dim = base_ch * 4
        style_dim = z_style_dim
        attn_set: Set[int] = set(attn_levels)

        # ---- Time + content (SEPARATE from style) ----
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.content_proj = nn.Sequential(
            nn.Linear(z_content_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # ---- Style projection ----
        self.style_proj = nn.Sequential(
            nn.Linear(z_style_dim, style_dim * 2),
            nn.SiLU(),
            nn.Linear(style_dim * 2, style_dim),
        )

        # ---- Input ----
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Down path ----
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_style = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        skip_chs: List[int] = [channels[0]]
        ch = channels[0]

        for level in range(n_levels):
            tgt = channels[level]
            for _ in range(n_res):
                self.down_res.append(FiLMResBlock(ch, tgt, cond_dim, dropout))
                ch = tgt
                self.down_style.append(StyleAdaIN(ch, style_dim))
                self.down_attn.append(
                    SelfAttention2d(ch) if level in attn_set else nn.Identity()
                )
                skip_chs.append(ch)
            if level < n_levels - 1:
                self.downsamples.append(Downsample2d(ch))
                skip_chs.append(ch)
            else:
                self.downsamples.append(None)  # type: ignore[arg-type]

        # ---- Bottleneck ----
        self.mid1 = FiLMResBlock(ch, ch, cond_dim, dropout)
        self.mid_style1 = StyleAdaIN(ch, style_dim)
        self.mid_attn = SelfAttention2d(ch)
        self.mid2 = FiLMResBlock(ch, ch, cond_dim, dropout)
        self.mid_style2 = StyleAdaIN(ch, style_dim)

        # ---- Up path ----
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.up_style = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(n_levels)):
            tgt = channels[level]
            n_up = n_res + 1
            for _ in range(n_up):
                skip_ch = skip_chs.pop()
                self.up_res.append(FiLMResBlock(ch + skip_ch, tgt, cond_dim, dropout))
                ch = tgt
                self.up_style.append(StyleAdaIN(ch, style_dim))
                self.up_attn.append(
                    SelfAttention2d(ch) if level in attn_set else nn.Identity()
                )
            if level > 0:
                self.upsamples.append(Upsample2d(ch))
            else:
                self.upsamples.append(None)  # type: ignore[arg-type]

        assert len(skip_chs) == 0, f"Skip mismatch: {len(skip_chs)} leftover"

        # ---- Output ----
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

        self._n_levels = n_levels
        self._n_res = n_res

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        z_content: torch.Tensor,
        z_style: torch.Tensor,
    ) -> torch.Tensor:
        cond_tc = self.time_emb(t) + self.content_proj(z_content)
        style = self.style_proj(z_style)

        # ---- Down ----
        h = self.conv_in(x)
        skips = [h]

        res_i = 0
        for level in range(self._n_levels):
            for _ in range(self._n_res):
                h = self.down_res[res_i](h, cond_tc)
                h = self.down_style[res_i](h, style)
                h = self.down_attn[res_i](h)
                res_i += 1
                skips.append(h)
            ds = self.downsamples[level]
            if ds is not None:
                h = ds(h)
                skips.append(h)

        # ---- Mid ----
        h = self.mid1(h, cond_tc)
        h = self.mid_style1(h, style)
        h = self.mid_attn(h)
        h = self.mid2(h, cond_tc)
        h = self.mid_style2(h, style)

        # ---- Up ----
        res_i = 0
        up_i = 0
        for level in reversed(range(self._n_levels)):
            n_up = self._n_res + 1
            for _ in range(n_up):
                s = skips.pop()
                if h.shape[2:] != s.shape[2:]:
                    h = F.interpolate(h, size=s.shape[2:], mode="nearest")
                h = self.up_res[res_i](torch.cat([h, s], dim=1), cond_tc)
                h = self.up_style[res_i](h, style)
                h = self.up_attn[res_i](h)
                res_i += 1
            us = self.upsamples[up_i]
            up_i += 1
            if us is not None:
                h = us(h)

        return self.conv_out(h)
