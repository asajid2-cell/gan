from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFResidualBlock(nn.Module):
    """
    Multi-Receptive-Field residual fusion block.
    """

    def __init__(self, channels: int, kernels: Sequence[int] = (3, 7, 11), dilation: int = 1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = (int(k) // 2) * int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=int(k), stride=1, padding=pad, dilation=int(dilation)),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                )
            )
        self.out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0.0
        for b in self.branches:
            y = y + b(x)
        y = y / float(len(self.branches))
        return x + self.out(y)


class ReconstructionDecoder(nn.Module):
    """
    Conditional mel decoder.
    Input:
      - z_content: [B, zc_dim]
      - v_target:  [B, cond_dim]
    Output:
      - mel_norm:  [B, n_mels, n_frames] in [-1, 1]
    """

    def __init__(
        self,
        zc_dim: int = 128,
        cond_dim: int = 160,
        n_mels: int = 96,
        n_frames: int = 256,
        base_channels: int = 256,
        norm: str = "instance",
        upsample: str = "transpose",
        spectral_norm: bool = False,
        mrf: bool = False,
        mrf_kernels: Sequence[int] = (3, 7, 11),
    ):
        super().__init__()
        if n_mels % 8 != 0 or n_frames % 8 != 0:
            raise ValueError("n_mels and n_frames must be divisible by 8.")
        self.n_mels = int(n_mels)
        self.n_frames = int(n_frames)
        self.h0 = self.n_mels // 8
        self.w0 = self.n_frames // 8
        self.base = int(base_channels)
        upsample_key = str(upsample).strip().lower()
        if upsample_key not in {"transpose", "pixelshuffle", "nearest"}:
            raise ValueError("upsample must be one of: {'transpose', 'pixelshuffle', 'nearest'}")
        norm_key = str(norm).strip().lower()
        if norm_key not in {"instance", "batch"}:
            raise ValueError("norm must be one of: {'instance', 'batch'}")
        use_sn = bool(spectral_norm)
        use_mrf = bool(mrf)

        def maybe_sn(layer: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(layer) if use_sn else layer

        def norm2d(c: int) -> nn.Module:
            if norm_key == "instance":
                return nn.InstanceNorm2d(c, affine=True)
            return nn.BatchNorm2d(c)

        def up_block(in_c: int, out_c: int) -> nn.Module:
            if upsample_key == "transpose":
                return nn.Sequential(
                    maybe_sn(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1)),
                    norm2d(out_c),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            if upsample_key == "nearest":
                return nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    maybe_sn(nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)),
                    norm2d(out_c),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            # PixelShuffle upsampling avoids checkerboard artifacts from transposed convs.
            return nn.Sequential(
                maybe_sn(nn.Conv2d(in_c, out_c * 4, kernel_size=3, stride=1, padding=1)),
                nn.PixelShuffle(2),
                norm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.fc = maybe_sn(nn.Linear(int(zc_dim + cond_dim), self.base * self.h0 * self.w0))
        self.up1 = up_block(self.base, self.base // 2)
        self.up2 = up_block(self.base // 2, self.base // 4)
        self.up3 = up_block(self.base // 4, self.base // 8)
        self.mrf1 = MRFResidualBlock(self.base // 2, kernels=mrf_kernels) if use_mrf else nn.Identity()
        self.mrf2 = MRFResidualBlock(self.base // 4, kernels=mrf_kernels) if use_mrf else nn.Identity()
        self.mrf3 = MRFResidualBlock(self.base // 8, kernels=mrf_kernels) if use_mrf else nn.Identity()
        self.out = maybe_sn(nn.Conv2d(self.base // 8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, z_content: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_content, v_target], dim=1)
        h = self.fc(x)
        h = h.view(x.shape[0], self.base, self.h0, self.w0)
        h = self.up1(h)
        h = self.mrf1(h)
        h = self.up2(h)
        h = self.mrf2(h)
        h = self.up3(h)
        h = self.mrf3(h)
        mel = torch.tanh(self.out(h)).squeeze(1)
        return mel


class MelDiscriminator(nn.Module):
    """
    Conditional projection mel discriminator.
    Input:
      - mel_norm: [B, n_mels, n_frames]
      - cond: [B, cond_dim] (optional when cond_dim == 0)
    Output:
      - logits: [B]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        cond_dim: int = 0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        c = int(base_channels)
        self.cond_dim = int(cond_dim)
        use_sn = bool(spectral_norm)

        def maybe_sn(layer: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(layer) if use_sn else layer

        self.net = nn.Sequential(
            maybe_sn(nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            maybe_sn(nn.Conv2d(c * 4, c * 8, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(c * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        feat_dim = c * 8
        self.head = maybe_sn(nn.Linear(feat_dim, 1))
        if self.cond_dim > 0:
            self.cond_proj = maybe_sn(nn.Linear(self.cond_dim, feat_dim))
        else:
            self.cond_proj = None

    def forward(
        self,
        mel_norm: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        x = mel_norm.unsqueeze(1)
        h = self.net(x).flatten(1)
        logit = self.head(h).squeeze(1)
        if self.cond_proj is not None and cond is not None:
            c = F.normalize(self.cond_proj(cond), dim=-1)
            h_n = F.normalize(h, dim=-1)
            logit = logit + torch.sum(h_n * c, dim=-1)
        if return_features:
            return logit, h
        return logit


class MultiScaleMelDiscriminator(nn.Module):
    """
    Multi-resolution discriminator:
    - scale 0: full resolution
    - scale 1: 1/2 resolution
    - scale 2: 1/4 resolution
    Returns average logit; features are concatenated for feature matching.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        cond_dim: int = 0,
        num_scales: int = 3,
        spectral_norm: bool = False,
    ):
        super().__init__()
        if int(num_scales) < 1:
            raise ValueError("num_scales must be >= 1")
        self.num_scales = int(num_scales)
        self.discriminators = nn.ModuleList(
            [
                MelDiscriminator(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    cond_dim=cond_dim,
                    spectral_norm=spectral_norm,
                )
                for _ in range(self.num_scales)
            ]
        )

    def forward(
        self,
        mel_norm: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        x = mel_norm
        logits = []
        features = []
        for i, disc in enumerate(self.discriminators):
            if return_features:
                li, fi = disc(x, cond, return_features=True)
                features.append(fi)
            else:
                li = disc(x, cond, return_features=False)
            logits.append(li)
            if i < (self.num_scales - 1):
                # Downsample along mel/time dimensions for the next discriminator.
                x = F.avg_pool2d(x.unsqueeze(1), kernel_size=(2, 2), stride=(2, 2)).squeeze(1)

        logit = torch.stack(logits, dim=0).mean(dim=0)
        if return_features:
            feat = torch.cat(features, dim=-1)
            return logit, feat
        return logit


class SubBandMelDiscriminator(nn.Module):
    """
    Frequency sub-band discriminator:
    - low, mid, high mel slices are judged independently
    - logits are averaged; features are concatenated
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        cond_dim: int = 0,
        low_bins: int = 32,
        mid_bins: int = 32,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.low_bins = int(low_bins)
        self.mid_bins = int(mid_bins)
        self.d_low = MelDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            cond_dim=cond_dim,
            spectral_norm=spectral_norm,
        )
        self.d_mid = MelDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            cond_dim=cond_dim,
            spectral_norm=spectral_norm,
        )
        self.d_high = MelDiscriminator(
            in_channels=in_channels,
            base_channels=base_channels,
            cond_dim=cond_dim,
            spectral_norm=spectral_norm,
        )

    def _split(self, mel_norm: torch.Tensor):
        n_mels = int(mel_norm.shape[1])
        low = max(8, min(self.low_bins, n_mels - 16))
        mid = max(8, min(self.mid_bins, n_mels - low - 8))
        hi_start = low + mid
        if hi_start >= n_mels:
            hi_start = max(8, n_mels - 8)
            low = max(8, min(low, hi_start - 8))
        x_low = mel_norm[:, :low, :]
        x_mid = mel_norm[:, low:hi_start, :]
        x_high = mel_norm[:, hi_start:, :]
        return x_low, x_mid, x_high

    def forward(
        self,
        mel_norm: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        x_low, x_mid, x_high = self._split(mel_norm)
        if return_features:
            l0, f0 = self.d_low(x_low, cond, return_features=True)
            l1, f1 = self.d_mid(x_mid, cond, return_features=True)
            l2, f2 = self.d_high(x_high, cond, return_features=True)
            logit = torch.stack([l0, l1, l2], dim=0).mean(dim=0)
            feat = torch.cat([f0, f1, f2], dim=-1)
            return logit, feat
        l0 = self.d_low(x_low, cond, return_features=False)
        l1 = self.d_mid(x_mid, cond, return_features=False)
        l2 = self.d_high(x_high, cond, return_features=False)
        return torch.stack([l0, l1, l2], dim=0).mean(dim=0)


class MultiPeriodMelDiscriminator(nn.Module):
    """
    Time-period discriminator:
    each branch views mel at a different temporal stride.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        cond_dim: int = 0,
        periods: tuple[int, ...] = (1, 2, 3, 5),
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.periods = tuple(int(max(1, p)) for p in periods)
        self.discriminators = nn.ModuleList(
            [
                MelDiscriminator(
                    in_channels=in_channels,
                    base_channels=base_channels,
                    cond_dim=cond_dim,
                    spectral_norm=spectral_norm,
                )
                for _ in self.periods
            ]
        )

    def forward(
        self,
        mel_norm: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        logits = []
        feats = []
        for p, disc in zip(self.periods, self.discriminators):
            x = mel_norm[:, :, ::p]
            # Guard tiny temporal views.
            if x.shape[-1] < 16:
                x = mel_norm
            if return_features:
                l, f = disc(x, cond, return_features=True)
                feats.append(f)
            else:
                l = disc(x, cond, return_features=False)
            logits.append(l)
        logit = torch.stack(logits, dim=0).mean(dim=0)
        if return_features:
            return logit, torch.cat(feats, dim=-1)
        return logit


class HybridMelDiscriminator(nn.Module):
    """
    Hybrid critic combining multiscale and multiperiod branches.
    """

    def __init__(
        self,
        cond_dim: int = 0,
        num_scales: int = 3,
        periods: tuple[int, ...] = (1, 2, 3, 5),
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.ms = MultiScaleMelDiscriminator(cond_dim=cond_dim, num_scales=num_scales, spectral_norm=spectral_norm)
        self.mp = MultiPeriodMelDiscriminator(cond_dim=cond_dim, periods=periods, spectral_norm=spectral_norm)

    def forward(
        self,
        mel_norm: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        if return_features:
            l_ms, f_ms = self.ms(mel_norm, cond, return_features=True)
            l_mp, f_mp = self.mp(mel_norm, cond, return_features=True)
            logit = 0.5 * (l_ms + l_mp)
            feat = torch.cat([f_ms, f_mp], dim=-1)
            return logit, feat
        l_ms = self.ms(mel_norm, cond, return_features=False)
        l_mp = self.mp(mel_norm, cond, return_features=False)
        return 0.5 * (l_ms + l_mp)


class StyleCritic(nn.Module):
    """
    Train-time style critic operating on frozen-encoder z_style features.
    This is separate from the frozen Lab1 branch used for audit metrics.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(n_classes)),
        )

    def forward(self, z_style: torch.Tensor) -> torch.Tensor:
        return self.net(z_style)


def bce_logits(
    logits: torch.Tensor,
    target_is_real: bool,
    real_label: float = 1.0,
    fake_label: float = 0.0,
) -> torch.Tensor:
    target_val = float(real_label) if target_is_real else float(fake_label)
    t = torch.full_like(logits, fill_value=target_val)
    return F.binary_cross_entropy_with_logits(logits, t)
