from __future__ import annotations

from typing import Iterable, Tuple
import math

import torch
import torch.nn.functional as F


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1)


def multi_resolution_stft_loss(
    mel_pred: torch.Tensor,
    mel_true: torch.Tensor,
    resolutions: Iterable[Tuple[int, int, int]] = ((64, 16, 64), (128, 32, 128), (256, 64, 256)),
) -> torch.Tensor:
    """
    Differentiable MR-STFT continuity loss on mel trajectories.
    Treats each mel band as a 1D signal over time.
    Inputs:
      mel_pred, mel_true: [B, n_mels, T] (normalized or dB-space, but same domain)
    """
    if mel_pred.shape != mel_true.shape:
        raise ValueError(f"Shape mismatch: {mel_pred.shape} vs {mel_true.shape}")

    b, n_mels, t = mel_pred.shape
    x = mel_pred.reshape(b * n_mels, t)
    y = mel_true.reshape(b * n_mels, t)
    device = x.device
    total = 0.0
    count = 0

    for n_fft, hop, win in resolutions:
        n_fft = int(n_fft)
        hop = int(hop)
        win = int(win)
        if t < n_fft:
            continue
        window = torch.hann_window(win, device=device)
        sx = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
            center=True,
        )
        sy = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True,
            center=True,
        )
        mx = torch.abs(sx) + 1e-8
        my = torch.abs(sy) + 1e-8
        sc = torch.norm(my - mx, p="fro") / (torch.norm(my, p="fro") + 1e-8)
        lm = F.l1_loss(torch.log(mx), torch.log(my))
        total = total + sc + lm
        count += 1

    if count == 0:
        return F.l1_loss(mel_pred, mel_true)
    return total / float(count)


def spectral_flatness_penalty_from_db(mel_db: torch.Tensor, target_max: float = 0.12) -> torch.Tensor:
    """
    Penalize over-flat (noise-like) mel spectra.
    mel_db: [B, n_mels, T] in dB space.
    """
    # Convert dB -> power and compute per-frame spectral flatness over mel bins.
    p = torch.pow(10.0, mel_db / 10.0).clamp_min(1e-10)
    gm = torch.exp(torch.mean(torch.log(p), dim=1))  # [B, T]
    am = torch.mean(p, dim=1).clamp_min(1e-10)        # [B, T]
    flat = gm / am
    # Penalize only when too flat; lets musical spectra stay unconstrained below target.
    return F.relu(flat - float(target_max)).mean()


def timbre_balance_penalty_from_db(
    mel_db: torch.Tensor,
    hf_top_frac: float = 0.25,
    lf_bottom_frac: float = 0.25,
    hf_max_ratio: float = 0.45,
    lf_min_ratio: float = 0.10,
) -> torch.Tensor:
    """
    Discourage screechy HF-dominant outputs in mel space.
    mel_db: [B, n_mels, T] in dB.
    """
    p = torch.pow(10.0, mel_db / 10.0).clamp_min(1e-10)
    n_mels = int(p.shape[1])
    top = max(1, int(round(n_mels * float(hf_top_frac))))
    bot = max(1, int(round(n_mels * float(lf_bottom_frac))))

    total = p.sum(dim=1).mean(dim=1).clamp_min(1e-10)        # [B]
    hf = p[:, -top:, :].sum(dim=1).mean(dim=1) / total       # [B]
    lf = p[:, :bot, :].sum(dim=1).mean(dim=1) / total        # [B]

    loss_hf = F.relu(hf - float(hf_max_ratio))
    loss_lf = F.relu(float(lf_min_ratio) - lf)
    return (loss_hf + loss_lf).mean()


def lowpass_mel_db(mel_db: torch.Tensor, keep_bins: int = 80, fill_db: float = -80.0) -> torch.Tensor:
    """
    Zero (to fill_db) high mel bins to suppress HF-style shortcuts.
    mel_db: [B, n_mels, T]
    """
    if mel_db.ndim != 3:
        raise ValueError(f"Expected [B, n_mels, T], got {tuple(mel_db.shape)}")
    k = int(max(1, min(int(keep_bins), int(mel_db.shape[1]))))
    out = mel_db.clone()
    if k < out.shape[1]:
        out[:, k:, :] = float(fill_db)
    return out


def estimate_lowpass_keep_bins(
    n_mels: int,
    sample_rate: int,
    cutoff_hz: float,
    fmin: float = 20.0,
    fmax: float | None = None,
) -> int:
    """
    Approximate number of mel bins whose center frequencies are <= cutoff_hz.
    """
    if n_mels <= 1:
        return 1
    sr = float(sample_rate)
    hi = float(sr * 0.5 if fmax is None else fmax)
    lo = float(max(1e-3, fmin))
    cutoff = float(max(lo, min(cutoff_hz, hi)))

    def hz_to_mel(h: float) -> float:
        return 2595.0 * math.log10(1.0 + h / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(lo)
    mel_max = hz_to_mel(hi)
    keep = 0
    for i in range(int(n_mels)):
        # Uniform mel centers.
        t = float(i) / float(max(1, n_mels - 1))
        m = mel_min + (mel_max - mel_min) * t
        hz = mel_to_hz(m)
        if hz <= cutoff:
            keep += 1
    return int(max(1, min(n_mels, keep)))


def bandpass_mel_db(
    mel_db: torch.Tensor,
    low_bin: int = 8,
    high_bin: int = 56,
    fill_db: float = -80.0,
) -> torch.Tensor:
    """
    Keep only [low_bin, high_bin) mel bins; set others to fill_db.
    """
    if mel_db.ndim != 3:
        raise ValueError(f"Expected [B, n_mels, T], got {tuple(mel_db.shape)}")
    n = int(mel_db.shape[1])
    lo = int(max(0, min(int(low_bin), n - 1)))
    hi = int(max(lo + 1, min(int(high_bin), n)))
    out = mel_db.clone()
    if lo > 0:
        out[:, :lo, :] = float(fill_db)
    if hi < n:
        out[:, hi:, :] = float(fill_db)
    return out


def frequency_weighted_l1_loss(
    mel_pred: torch.Tensor,
    mel_true: torch.Tensor,
    split_bin: int = 80,
    low_gain: float = 5.0,
    high_gain: float = 0.5,
) -> torch.Tensor:
    """
    Weighted L1 over mel bins, emphasizing bass/mids and deemphasizing upper bins.
    Inputs are [B, n_mels, T] in same domain.
    """
    if mel_pred.shape != mel_true.shape:
        raise ValueError(f"Shape mismatch: {tuple(mel_pred.shape)} vs {tuple(mel_true.shape)}")
    n_mels = int(mel_pred.shape[1])
    s = int(max(1, min(int(split_bin), n_mels)))
    w = torch.full((n_mels,), float(high_gain), device=mel_pred.device, dtype=mel_pred.dtype)
    w[:s] = float(low_gain)
    w = w.view(1, n_mels, 1)
    return torch.mean(torch.abs(mel_pred - mel_true) * w)


def spectral_tilt_penalty_from_db(
    mel_db: torch.Tensor,
    split_frac: float = 0.5,
    hf_to_lf_max_ratio: float = 0.7,
) -> torch.Tensor:
    """
    Enforce a downward spectral tilt (LF/Mid energy should dominate HF).
    """
    p = torch.pow(10.0, mel_db / 10.0).clamp_min(1e-10)
    n_mels = int(p.shape[1])
    split = int(max(1, min(n_mels - 1, round(n_mels * float(split_frac)))))
    lf = p[:, :split, :].mean(dim=(1, 2))
    hf = p[:, split:, :].mean(dim=(1, 2))
    ratio = hf / (lf + 1e-10)
    return F.relu(ratio - float(hf_to_lf_max_ratio)).mean()


def zcr_proxy_penalty_from_db(
    mel_db: torch.Tensor,
    target_max: float = 0.18,
    smooth_k: float = 8.0,
) -> torch.Tensor:
    """
    Differentiable proxy for waveform ZCR using mel temporal trajectory.
    """
    p = torch.pow(10.0, mel_db / 10.0).clamp_min(1e-10)
    # Collapse frequency -> 1D temporal envelope.
    env = torch.log(p.mean(dim=1) + 1e-10)  # [B, T]
    if env.shape[1] < 3:
        return torch.zeros((), device=mel_db.device, dtype=mel_db.dtype)
    d = env[:, 1:] - env[:, :-1]  # [B, T-1]
    s = torch.tanh(float(smooth_k) * d)
    cross = F.relu(-(s[:, 1:] * s[:, :-1]))  # ~sign flips
    zcrp = cross.mean(dim=1)
    return F.relu(zcrp - float(target_max)).mean()


def highpass_recon_l1_loss(
    mel_pred: torch.Tensor,
    mel_true: torch.Tensor,
    low_cut_bin: int = 12,
    gain: float = 2.0,
) -> torch.Tensor:
    """
    Anchor harmonics above low_cut_bin so model cannot satisfy style by bass-only mud.
    """
    if mel_pred.shape != mel_true.shape:
        raise ValueError(f"Shape mismatch: {tuple(mel_pred.shape)} vs {tuple(mel_true.shape)}")
    n_mels = int(mel_pred.shape[1])
    k = int(max(0, min(int(low_cut_bin), n_mels - 1)))
    w = torch.ones((n_mels,), device=mel_pred.device, dtype=mel_pred.dtype)
    w[:k] = 0.0
    w[k:] = float(gain)
    w = w.view(1, n_mels, 1)
    return torch.mean(torch.abs(mel_pred - mel_true) * w)


def hf_excess_penalty_from_db(
    mel_fake_db: torch.Tensor,
    mel_real_db: torch.Tensor,
    top_frac: float = 0.20,
    max_ratio_multiplier: float = 1.20,
) -> torch.Tensor:
    """
    Penalize fake high-frequency energy ratio when it exceeds real-batch ratio by margin.
    """
    if mel_fake_db.shape != mel_real_db.shape:
        raise ValueError(f"Shape mismatch: {tuple(mel_fake_db.shape)} vs {tuple(mel_real_db.shape)}")

    pf = torch.pow(10.0, mel_fake_db / 10.0).clamp_min(1e-10)
    pr = torch.pow(10.0, mel_real_db / 10.0).clamp_min(1e-10)
    n_mels = int(pf.shape[1])
    top = int(max(1, min(n_mels, round(n_mels * float(top_frac)))))

    fake_total = pf.sum(dim=1).mean(dim=1).clamp_min(1e-10)
    real_total = pr.sum(dim=1).mean(dim=1).clamp_min(1e-10)
    fake_hf = pf[:, -top:, :].sum(dim=1).mean(dim=1) / fake_total
    real_hf = pr[:, -top:, :].sum(dim=1).mean(dim=1) / real_total
    target = real_hf * float(max_ratio_multiplier)
    return F.relu(fake_hf - target).mean()
