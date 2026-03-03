"""Diffusion training loop with classifier-free guidance (CFG).

Training:
  - DDPM with epsilon prediction on mel spectrograms
  - CFG: independently drop z_content (10%) and z_style (10%) with zeros
  - EMA on model weights (decay 0.9999)
  - Augmentation handled by DiffusionMelDataset (time-shift, freq mask, gain)

Inference:
  - DDIM sampling (configurable steps, default 50)
  - CFG at inference: eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
  - BigVGAN vocoder for mel → audio
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .lab3_diffusion_model import DiffusionUNet, DiffusionUNetV2, EMA, NoiseSchedule


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class DiffusionTrainConfig:
    epochs: int = 100
    lr: float = 1e-4
    batch_size: int = 8
    num_workers: int = 0
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    cfg_dropout_p: float = 0.10  # probability of dropping each conditioning independently
    log_every: int = 50  # log every N batches
    save_every: int = 5  # save checkpoint every N epochs
    warmup_steps: int = 500


def train_one_epoch(
    model: DiffusionUNet,
    schedule: NoiseSchedule,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    ema: EMA,
    cfg: DiffusionTrainConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, int]:
    """Train for one epoch.  Returns (avg_loss, updated_global_step)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None

    for batch_idx, batch in enumerate(loader):
        mel = batch["mel"].to(device)           # [B, 1, 80, 432]
        cond_feat = batch["cond_feat"].to(device)  # [B, 14, 80, 432]
        z_content = batch["z_content"].to(device)  # [B, 128]
        z_style = batch["z_style"].to(device)      # [B, 128]

        B = mel.shape[0]

        # -- CFG dropout: independently zero-out each conditioning --
        if cfg.cfg_dropout_p > 0:
            mask_content = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            mask_style = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            z_content = z_content * mask_content
            z_style = z_style * mask_style

        # -- Sample random timesteps --
        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)

        # -- Forward diffusion --
        noise = torch.randn_like(mel)
        x_t = schedule.q_sample(mel, t, noise)

        # -- UNet input: concat noisy mel with conditioning features --
        unet_input = torch.cat([x_t, cond_feat], dim=1)  # [B, 15, 80, 432]

        optimizer.zero_grad(set_to_none=True)

        # -- Forward + loss with AMP --
        with torch.amp.autocast("cuda", enabled=use_amp):
            eps_pred = model(unet_input, t, z_content, z_style)
            loss = F.mse_loss(eps_pred, noise)

        # -- Backward --
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        else:
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        # -- LR warmup --
        if global_step < cfg.warmup_steps:
            warmup_factor = float(global_step + 1) / float(cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * warmup_factor

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        ema.update(model)

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if cfg.log_every > 0 and (batch_idx + 1) % cfg.log_every == 0:
            avg = total_loss / n_batches
            print(f"  [epoch {epoch}] batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.5f}  avg={avg:.5f}  step={global_step}")

    return total_loss / max(1, n_batches), global_step


# ---------------------------------------------------------------------------
# Inference (DDIM with CFG)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    schedule: NoiseSchedule,
    cond_feat: torch.Tensor,   # [B, 14, 80, 432]
    z_content: torch.Tensor,    # [B, 128]
    z_style: torch.Tensor,      # [B, 128]
    n_steps: int = 50,
    guidance_scale: float = 4.0,
    eta: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    """DDIM sampling with classifier-free guidance.

    Returns denoised mel [B, 1, 80, 432] in [-1, 1] normalized space.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = cond_feat.shape[0]
    shape = (B, 1, 80, 432)

    # Timestep sub-sequence for DDIM
    step_size = schedule.T // n_steps
    timesteps = list(range(0, schedule.T, step_size))[::-1]  # descending

    x_t = torch.randn(shape, device=device)

    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        unet_input = torch.cat([x_t, cond_feat], dim=1)

        # Conditional prediction
        eps_cond = model(unet_input, t_batch, z_content, z_style)

        if guidance_scale != 1.0:
            # Unconditional prediction (zero out both z vectors)
            z_zero = torch.zeros_like(z_content)
            eps_uncond = model(unet_input, t_batch, z_zero, z_zero)
            # CFG
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_cond

        x_t = schedule.ddim_sample_step(eps, x_t, t_cur, t_prev, eta=eta)

    return torch.clamp(x_t, -1.0, 1.0)


# ---------------------------------------------------------------------------
# BigVGAN vocoding
# ---------------------------------------------------------------------------

def vocode_bigvgan(
    mel_norm: torch.Tensor,
    mel_min: float,
    mel_max: float,
    bigvgan_model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Convert normalized mel [B, 1, 80, T] → audio [B, S] via BigVGAN.

    Returns numpy float32 array.
    """
    from .lab3_diffusion_data import denormalize_mel

    # Denormalize to BigVGAN log mel space
    log_mel = denormalize_mel(mel_norm.squeeze(1), mel_min, mel_max)  # [B, 80, T]
    log_mel = log_mel.to(device)

    with torch.no_grad():
        wav = bigvgan_model(log_mel)  # [B, 1, S]

    return wav.squeeze(1).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_pyin_pitch(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract pitch contour using pyin.  Returns [T] Hz (0 = unvoiced)."""
    import librosa
    f0, voiced, _ = librosa.pyin(
        audio.astype(np.float64), fmin=50, fmax=2000, sr=sr,
        hop_length=256,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    return f0


def pitch_correlation(source_audio: np.ndarray, gen_audio: np.ndarray, sr: int = 22050) -> float:
    """Pearson correlation between source and generated pitch contours."""
    f0_src = compute_pyin_pitch(source_audio, sr)
    f0_gen = compute_pyin_pitch(gen_audio, sr)
    # Align lengths
    n = min(len(f0_src), len(f0_gen))
    f0_src, f0_gen = f0_src[:n], f0_gen[:n]
    # Only compare voiced frames
    voiced = (f0_src > 0) & (f0_gen > 0)
    if voiced.sum() < 10:
        return 0.0
    src_v = f0_src[voiced]
    gen_v = f0_gen[voiced]
    if src_v.std() < 1e-6 or gen_v.std() < 1e-6:
        return 0.0
    corr = float(np.corrcoef(src_v, gen_v)[0, 1])
    return corr if np.isfinite(corr) else 0.0


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: DiffusionUNet,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_loss: float,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, str(path))


def load_checkpoint(
    path: Path,
    model,
    ema: EMA,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = None,
) -> Dict:
    ckpt = torch.load(str(path), map_location=device or "cpu")
    model.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ===========================================================================
# V2: v-prediction + gradient accumulation + cosine LR + epoch sampling
# ===========================================================================

def train_one_epoch_v2(
    model: nn.Module,
    schedule: NoiseSchedule,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    ema: EMA,
    cfg: DiffusionTrainConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 4,
    scheduler: Optional[object] = None,
    save_fn=None,
) -> Tuple[float, int]:
    """V2 training: v-prediction, gradient accumulation, cosine LR.

    save_fn(global_step): called every save_every steps to save latest checkpoint.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        mel = batch["mel"].to(device)
        cond_feat = batch["cond_feat"].to(device)
        z_content = batch["z_content"].to(device)
        z_style = batch["z_style"].to(device)
        B = mel.shape[0]

        # CFG dropout: independently zero each conditioning
        if cfg.cfg_dropout_p > 0:
            mask_c = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            mask_s = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            z_content = z_content * mask_c
            z_style = z_style * mask_s

        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(mel)
        x_t = schedule.q_sample(mel, t, noise)

        # v-prediction target
        v_target = schedule.compute_v_target(mel, t, noise)

        unet_input = torch.cat([x_t, cond_feat], dim=1)

        with torch.amp.autocast("cuda", enabled=use_amp):
            v_pred = model(unet_input, t, z_content, z_style)
            loss = F.mse_loss(v_pred, v_target) / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        real_loss = loss.item() * grad_accum_steps
        total_loss += real_loss
        n_batches += 1
        global_step += 1

        # Optimizer step every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            if scheduler is not None:
                scheduler.step()

        if cfg.log_every > 0 and (batch_idx + 1) % cfg.log_every == 0:
            avg = total_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            print(f"  [epoch {epoch}] batch {batch_idx+1}/{len(loader)}  "
                  f"loss={real_loss:.5f}  avg={avg:.5f}  lr={lr:.2e}  step={global_step}")

        # Periodic save within epoch
        if save_fn is not None and global_step % 500 == 0:
            save_fn(global_step)

    # Handle leftover gradients
    if (batch_idx + 1) % grad_accum_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        if cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    return total_loss / max(1, n_batches), global_step


@torch.no_grad()
def ddim_sample_v2(
    model: nn.Module,
    schedule: NoiseSchedule,
    cond_feat: torch.Tensor,
    z_content: torch.Tensor,
    z_style: torch.Tensor,
    n_steps: int = 50,
    guidance_scale: float = 2.0,
    eta: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    """DDIM sampling with v-prediction + CFG."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = cond_feat.shape[0]
    T_frames = cond_feat.shape[3]
    shape = (B, 1, 80, T_frames)

    step_size = schedule.T // n_steps
    timesteps = list(range(0, schedule.T, step_size))[::-1]

    x_t = torch.randn(shape, device=device)

    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        unet_input = torch.cat([x_t, cond_feat], dim=1)

        v_cond = model(unet_input, t_batch, z_content, z_style)

        if guidance_scale != 1.0:
            z_zero_c = torch.zeros_like(z_content)
            z_zero_s = torch.zeros_like(z_style)
            v_uncond = model(unet_input, t_batch, z_zero_c, z_zero_s)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        # Convert v → eps for DDIM step
        eps = schedule.v_to_eps(x_t, t_cur, v)
        x_t = schedule.ddim_sample_step(eps, x_t, t_cur, t_prev, eta=eta)

    return torch.clamp(x_t, -1.0, 1.0)


@torch.no_grad()
def ddim_sample_v2_constrained(
    model: nn.Module,
    schedule: NoiseSchedule,
    cond_feat: torch.Tensor,
    z_content: torch.Tensor,
    z_style: torch.Tensor,
    *,
    source_mel: Optional[torch.Tensor] = None,
    t_start: int = 350,
    prefix_x0: Optional[torch.Tensor] = None,
    prefix_frames: int = 0,
    prefix_blend: float = 1.0,
    source_prefix_x0: Optional[torch.Tensor] = None,
    source_prefix_blend: float = 0.0,
    n_steps: int = 50,
    guidance_scale: float = 2.0,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """DDIM v2 sampler with longform continuity constraints.

    Coherence controls:
      1. SDEdit-style source anchoring via `source_mel` + `t_start`
      2. Overlap locking via `prefix_x0` on the first `prefix_frames` each step
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = cond_feat.shape[0]
    T_frames = cond_feat.shape[3]
    shape = (B, 1, 80, T_frames)

    step_size = max(1, schedule.T // max(1, int(n_steps)))
    base_steps = list(range(0, schedule.T, step_size))

    use_sdedit = source_mel is not None
    if use_sdedit:
        t0 = int(np.clip(int(t_start), 1, schedule.T - 1))
        timesteps = sorted(set([t0] + [int(t) for t in base_steps if int(t) < t0]), reverse=True)
        noise = torch.randn(shape, device=device)
        t_batch = torch.full((B,), t0, device=device, dtype=torch.long)
        x_t = schedule.q_sample(source_mel.to(device), t_batch, noise)
    else:
        timesteps = list(reversed(base_steps))
        x_t = torch.randn(shape, device=device)

    lock_prefix = (prefix_x0 is not None) and int(prefix_frames) > 0
    lock_frames = 0
    prefix_noise = None
    prefix_ref = None
    source_prefix_ref = None
    if lock_prefix:
        lock_frames = min(int(prefix_frames), T_frames, int(prefix_x0.shape[-1]))
        if lock_frames > 0:
            prefix_ref = prefix_x0[..., :lock_frames].to(device)
            prefix_noise = torch.randn((B, 1, 80, lock_frames), device=device)
            if source_prefix_x0 is not None:
                source_prefix_ref = source_prefix_x0[..., :lock_frames].to(device)
        else:
            lock_prefix = False

    alpha = float(np.clip(float(prefix_blend), 0.0, 1.0))
    source_alpha = float(np.clip(float(source_prefix_blend), 0.0, 1.0))
    if lock_prefix and source_prefix_ref is not None and source_alpha > 0.0 and prefix_ref is not None:
        # Blend previous generated overlap with current source overlap in x0 space
        prefix_ref = (1.0 - source_alpha) * prefix_ref + source_alpha * source_prefix_ref
    z_zero_c = torch.zeros_like(z_content)
    z_zero_s = torch.zeros_like(z_style)

    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((B,), int(t_cur), device=device, dtype=torch.long)

        unet_input = torch.cat([x_t, cond_feat], dim=1)
        v_cond = model(unet_input, t_batch, z_content, z_style)

        if guidance_scale != 1.0:
            v_uncond = model(unet_input, t_batch, z_zero_c, z_zero_s)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

        eps = schedule.v_to_eps(x_t, int(t_cur), v)
        x_t = schedule.ddim_sample_step(eps, x_t, int(t_cur), int(t_prev), eta=eta)

        # Lock overlap region at current noise level to previous chunk tail.
        if lock_prefix and t_prev >= 0 and prefix_ref is not None and prefix_noise is not None:
            t_prev_batch = torch.full((B,), int(t_prev), device=device, dtype=torch.long)
            x_prefix = schedule.q_sample(prefix_ref, t_prev_batch, prefix_noise)
            if alpha >= 0.999:
                x_t[..., :lock_frames] = x_prefix
            elif alpha > 0.0:
                x_t[..., :lock_frames] = (
                    alpha * x_prefix + (1.0 - alpha) * x_t[..., :lock_frames]
                )

    return torch.clamp(x_t, -1.0, 1.0)


# ===========================================================================
# V3: v-prediction + adversarial training (fine-tune from V2)
# ===========================================================================


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator."""
    return 0.5 * (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean())


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator."""
    return -fake_logits.mean()


def feature_matching_loss(feat_real: torch.Tensor, feat_fake: torch.Tensor) -> torch.Tensor:
    """L1 feature matching loss (detaches real features)."""
    return F.l1_loss(feat_fake, feat_real.detach())


def train_one_epoch_v3(
    model: nn.Module,
    schedule: NoiseSchedule,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    ema: EMA,
    cfg: DiffusionTrainConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    discriminator: nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler] = None,
    disc_scaler: Optional[torch.amp.GradScaler] = None,
    grad_accum_steps: int = 4,
    scheduler: Optional[object] = None,
    disc_scheduler: Optional[object] = None,
    adv_weight: float = 0.1,
    fm_weight: float = 0.5,
    disc_warmup_steps: int = 500,
    save_fn=None,
) -> Tuple[float, float, float, int]:
    """V3 training: v-prediction + adversarial (hinge) + feature matching.

    Fine-tunes from a V2 checkpoint with an added HybridMelDiscriminator.
    Returns (avg_gen_loss, avg_disc_loss, avg_mse_loss, updated_global_step).
    """
    model.train()
    discriminator.train()
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    total_mse_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        mel = batch["mel"].to(device)            # [B, 1, 80, T]
        cond_feat = batch["cond_feat"].to(device)
        z_content = batch["z_content"].to(device)
        z_style = batch["z_style"].to(device)
        B = mel.shape[0]

        # CFG dropout
        if cfg.cfg_dropout_p > 0:
            mask_c = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            mask_s = (torch.rand(B, 1, device=device) > cfg.cfg_dropout_p).float()
            z_content = z_content * mask_c
            z_style = z_style * mask_s

        t = torch.randint(0, schedule.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(mel)
        x_t = schedule.q_sample(mel, t, noise)
        v_target = schedule.compute_v_target(mel, t, noise)
        unet_input = torch.cat([x_t, cond_feat], dim=1)

        # ---- Generator forward ----
        with torch.amp.autocast("cuda", enabled=use_amp):
            v_pred = model(unet_input, t, z_content, z_style)
            mse_loss = F.mse_loss(v_pred, v_target)

        # Recover x0_pred from v-prediction (batched timesteps)
        s_a = schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        s_om = schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x0_pred = torch.clamp(s_a * x_t - s_om * v_pred.float(), -1.0, 1.0)

        use_adv = global_step >= disc_warmup_steps
        d_loss_val = 0.0
        g_adv_val = 0.0
        fm_val = 0.0

        if use_adv:
            real_mel = mel.squeeze(1)                   # [B, 80, T]
            fake_mel_det = x0_pred.squeeze(1).detach()  # no grad to gen
            fake_mel_live = x0_pred.squeeze(1)           # grad flows to gen

            # ---- Discriminator step ----
            disc_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                d_real = discriminator(real_mel)
                d_fake = discriminator(fake_mel_det)
                d_loss = hinge_d_loss(d_real, d_fake)

            if disc_scaler is not None:
                disc_scaler.scale(d_loss).backward()
                disc_scaler.unscale_(disc_optimizer)
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                disc_scaler.step(disc_optimizer)
                disc_scaler.update()
            else:
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                disc_optimizer.step()

            if disc_scheduler is not None:
                disc_scheduler.step()
            d_loss_val = d_loss.item()

            # ---- Generator adversarial + feature matching loss ----
            with torch.amp.autocast("cuda", enabled=use_amp):
                d_fake_g, feat_fake = discriminator(fake_mel_live, return_features=True)
                with torch.no_grad():
                    _, feat_real = discriminator(real_mel, return_features=True)
                g_adv = hinge_g_loss(d_fake_g)
                fm_loss = feature_matching_loss(feat_real, feat_fake)
                gen_loss = (mse_loss + adv_weight * g_adv + fm_weight * fm_loss) / grad_accum_steps

            g_adv_val = g_adv.item()
            fm_val = fm_loss.item()
        else:
            gen_loss = mse_loss / grad_accum_steps

        # ---- Generator backward (accumulated) ----
        if use_amp:
            scaler.scale(gen_loss).backward()
        else:
            gen_loss.backward()

        real_gen_loss = gen_loss.item() * grad_accum_steps
        total_gen_loss += real_gen_loss
        total_disc_loss += d_loss_val
        total_mse_loss += mse_loss.item()
        n_batches += 1
        global_step += 1

        # Generator optimizer step every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            if scheduler is not None:
                scheduler.step()

        if cfg.log_every > 0 and (batch_idx + 1) % cfg.log_every == 0:
            avg_g = total_gen_loss / n_batches
            avg_d = total_disc_loss / n_batches
            avg_m = total_mse_loss / n_batches
            lr = optimizer.param_groups[0]["lr"]
            adv_str = (f"d={d_loss_val:.4f} g_adv={g_adv_val:.4f} fm={fm_val:.4f}"
                       if use_adv else "warmup")
            print(f"  [epoch {epoch}] batch {batch_idx+1}/{len(loader)}  "
                  f"mse={mse_loss.item():.5f}  {adv_str}  "
                  f"avg_gen={avg_g:.5f}  avg_mse={avg_m:.5f}  lr={lr:.2e}  step={global_step}")

        if save_fn is not None and global_step % 500 == 0:
            save_fn(global_step)

    # Handle leftover gradients
    if n_batches > 0 and (batch_idx + 1) % grad_accum_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
        if cfg.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)

    avg_gen = total_gen_loss / max(1, n_batches)
    avg_disc = total_disc_loss / max(1, n_batches)
    avg_mse = total_mse_loss / max(1, n_batches)
    return avg_gen, avg_disc, avg_mse, global_step


def generate_epoch_samples(
    ema: EMA,
    schedule: NoiseSchedule,
    val_ds,
    genre_to_idx: Dict,
    mel_min: float,
    mel_max: float,
    device: torch.device,
    epoch: int,
    out_dir: Path,
    n_samples: int = 4,
    ddim_steps: int = 50,
    guidance_scale: float = 2.0,
) -> None:
    """Generate sample WAVs at end of epoch for quality monitoring."""
    import soundfile as sf
    try:
        import bigvgan as bvg
    except ImportError:
        print("  [epoch samples] bigvgan not available, skipping")
        return

    sample_dir = out_dir / "epoch_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    idx_to_genre = {v: k for k, v in genre_to_idx.items()}

    # Load vocoder temporarily
    vocoder = bvg.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    from .lab3_diffusion_data import DIFFUSION_SR

    n = min(n_samples, len(val_ds))
    for i in range(n):
        batch = val_ds[i]
        genre_name = idx_to_genre.get(int(batch["genre_idx"].item()), "unk")

        mel_gt = batch["mel"].unsqueeze(0).to(device)
        cond_feat = batch["cond_feat"].unsqueeze(0).to(device)
        z_c = batch["z_content"].unsqueeze(0).to(device)
        z_s = batch["z_style"].unsqueeze(0).to(device)

        # GT
        wav_gt = vocode_bigvgan(mel_gt, mel_min, mel_max, vocoder, device)
        sf.write(str(sample_dir / f"e{epoch:03d}_{i:02d}_{genre_name}_GT.wav"),
                 wav_gt[0], DIFFUSION_SR)

        # Generated
        mel_gen = ddim_sample_v2(
            ema.shadow, schedule, cond_feat, z_c, z_s,
            n_steps=ddim_steps, guidance_scale=guidance_scale, device=device,
        )
        wav_gen = vocode_bigvgan(mel_gen, mel_min, mel_max, vocoder, device)
        sf.write(str(sample_dir / f"e{epoch:03d}_{i:02d}_{genre_name}_gen.wav"),
                 wav_gen[0], DIFFUSION_SR)

    del vocoder
    torch.cuda.empty_cache()
    print(f"  Epoch {epoch} samples saved to {sample_dir}")
