#!/usr/bin/env python3
"""SDEdit v2: Spectral envelope transfer + style-only CFG on Diffusion V2.

Two-stage approach:
  1. Spectral envelope swap: transfer the average frequency shape from target genre
     to source mel.  This directly changes timbre (which bands are loud/quiet).
  2. SDEdit with style-only CFG: denoise the swapped mel, guiding specifically
     on z_style to amplify timbral differences.

Usage:
  python quick_eval_sdedit_v2.py
  python quick_eval_sdedit_v2.py --t-start 300 --style-guidance 3.0
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parent

from src.lab3_diffusion_data import (
    DIFFUSION_SR, DiffusionMelDataset, load_diffusion_cache, denormalize_mel,
)
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import load_checkpoint, vocode_bigvgan
from src.lab3_data import stratified_group_split_indices


def spectral_envelope_transfer(
    mel_source: torch.Tensor,    # [B, 1, 80, T] normalized [-1, 1]
    mel_target: torch.Tensor,    # [B, 1, 80, T] normalized [-1, 1]
    strength: float = 1.0,       # 0=no change, 1=full transfer
) -> torch.Tensor:
    """Transfer spectral envelope (average freq shape) from target to source.

    This changes the timbre (which frequency bands are emphasized) while
    preserving the temporal structure (melody, rhythm).
    """
    # Compute average spectral shape (mean across time)
    src_shape = mel_source.mean(dim=-1, keepdim=True)  # [B, 1, 80, 1]
    tgt_shape = mel_target.mean(dim=-1, keepdim=True)  # [B, 1, 80, 1]

    # Remove source shape, add target shape (scaled by strength)
    delta = (tgt_shape - src_shape) * strength
    mel_transferred = mel_source + delta

    return torch.clamp(mel_transferred, -1.0, 1.0)


@torch.no_grad()
def sdedit_style_guided(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    mel_source: torch.Tensor,    # [B, 1, 80, 432]
    cond_feat: torch.Tensor,     # [B, 14, 80, 432]
    z_content: torch.Tensor,     # [B, 128]
    z_style_target: torch.Tensor,  # [B, 128] — TARGET style
    t_start: int = 350,
    n_ddim_steps: int = 50,
    style_guidance: float = 3.0,
    device: torch.device = None,
) -> torch.Tensor:
    """SDEdit with style-only CFG.

    Instead of standard CFG (uncond vs full-cond), we guide on:
      eps_guided = eps_no_style + w * (eps_with_style - eps_no_style)

    This amplifies the effect of z_style specifically.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = mel_source.shape[0]

    # Forward diffuse
    noise = torch.randn_like(mel_source)
    t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
    x_t = schedule.q_sample(mel_source, t_tensor, noise)

    # DDIM timesteps from t_start down
    full_step_size = schedule.T // n_ddim_steps
    all_steps = list(range(0, schedule.T, full_step_size))
    timesteps = sorted([t for t in all_steps if t <= t_start], reverse=True)

    z_style_zero = torch.zeros_like(z_style_target)

    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        unet_input = torch.cat([x_t, cond_feat], dim=1)

        # With target style
        v_styled = model(unet_input, t_batch, z_content, z_style_target)

        if style_guidance != 1.0:
            # Without style (but keep content!)
            v_no_style = model(unet_input, t_batch, z_content, z_style_zero)
            # Guide on style difference only
            v = v_no_style + style_guidance * (v_styled - v_no_style)
        else:
            v = v_styled

        eps = schedule.v_to_eps(x_t, t_cur, v)
        x_t = schedule.ddim_sample_step(eps, x_t, t_cur, t_prev, eta=0.0)

    return torch.clamp(x_t, -1.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002"))
    p.add_argument("--checkpoint", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--t-start", type=int, default=350)
    p.add_argument("--style-guidance", type=float, default=3.0)
    p.add_argument("--envelope-strengths", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                   help="Spectral envelope transfer strengths to test")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=3)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = run_dir / "cache"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "sdedit_v2"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load cache + val split
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"])
    track_ids = index_df["track_id"].to_numpy()
    _, val_idx = stratified_group_split_indices(genre_idx, track_ids, val_ratio=0.1, seed=328)

    val_ds = DiffusionMelDataset(
        arrays, val_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=False, seed=329, style_source="lab1",
    )
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}
    print(f"Val set: {len(val_ds)} samples")
    print(f"Genres: {list(genre_to_idx.keys())}")

    # Load model
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "latest.pt"

    model = DiffusionUNetV2(
        in_channels=15, out_channels=1, base_ch=64,
        ch_mults=(1, 2, 4, 4), n_res=2, attn_levels=(2, 3),
        z_content_dim=128, z_style_dim=128, dropout=0.1,
    ).to(device)
    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=0.9999)
    ckpt = load_checkpoint(ckpt_path, model, ema, device=device)
    print(f"Loaded {ckpt_path.name}: epoch={ckpt.get('epoch','?')}")

    # Load BigVGAN
    print("Loading BigVGAN vocoder...")
    import bigvgan as bvg
    import soundfile as sf
    vocoder = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    # Build per-genre style banks + mel exemplars
    print("\nBuilding genre style banks...")
    genre_styles = {}   # {genre_idx: [B=1, 128]}
    genre_mels = {}     # {genre_idx: [B=1, 1, 80, 432]} averaged mel shape
    for gi, gname in idx_to_genre.items():
        styles = []
        mels = []
        for j in range(min(len(val_ds), 3000)):
            if int(val_ds.genre_idx[j]) == gi:
                b = val_ds[j]
                styles.append(b["z_style"].numpy())
                mels.append(b["mel"].numpy())
                if len(styles) >= 30:
                    break
        if styles:
            genre_styles[gi] = torch.from_numpy(np.mean(styles, axis=0)).unsqueeze(0).to(device)
            genre_mels[gi] = torch.from_numpy(np.mean(mels, axis=0)).unsqueeze(0).to(device)
            print(f"  {gname}: {len(styles)} exemplars")

    # ---- Cross-genre style transfer ----
    print(f"\n{'='*60}")
    print("CROSS-GENRE STYLE TRANSFER")
    print(f"  t_start={args.t_start}, style_guidance={args.style_guidance}")
    print(f"  envelope_strengths={args.envelope_strengths}")
    print(f"{'='*60}\n")

    for i in range(min(args.n_samples, len(val_ds))):
        batch = val_ds[i]
        src_gi = int(batch["genre_idx"].item())
        src_name = idx_to_genre.get(src_gi, "unk")

        mel_src = batch["mel"].unsqueeze(0).to(device)
        cond_feat = batch["cond_feat"].unsqueeze(0).to(device)
        z_c = batch["z_content"].unsqueeze(0).to(device)

        # Save source GT
        wav_gt = vocode_bigvgan(mel_src, meta.mel_min, meta.mel_max, vocoder, device)
        sf.write(str(eval_dir / f"{i:02d}_{src_name}_source.wav"), wav_gt[0], DIFFUSION_SR)
        print(f"[{i:02d}] Source: {src_name}")

        for tgt_gi, tgt_style in genre_styles.items():
            if tgt_gi == src_gi:
                continue
            tgt_name = idx_to_genre.get(tgt_gi, "unk")
            tgt_mel = genre_mels[tgt_gi]

            for env_str in args.envelope_strengths:
                # Step 1: spectral envelope transfer
                if env_str > 0:
                    mel_swapped = spectral_envelope_transfer(mel_src, tgt_mel, strength=env_str)
                else:
                    mel_swapped = mel_src

                # Step 2: SDEdit with style-only guidance
                t0 = time.time()
                mel_out = sdedit_style_guided(
                    ema.shadow, schedule, mel_swapped, cond_feat,
                    z_c, tgt_style,
                    t_start=args.t_start,
                    n_ddim_steps=args.ddim_steps,
                    style_guidance=args.style_guidance,
                    device=device,
                )
                wav_out = vocode_bigvgan(mel_out, meta.mel_min, meta.mel_max, vocoder, device)
                dt = time.time() - t0

                tag = f"e{env_str:.1f}" if env_str > 0 else "noenv"
                fname = f"{i:02d}_{src_name}_to_{tgt_name}_{tag}.wav"
                sf.write(str(eval_dir / fname), wav_out[0], DIFFUSION_SR)
                print(f"  → {tgt_name} [{tag}] {dt:.1f}s")

            # Also save envelope-only (no diffusion) for comparison
            mel_env_only = spectral_envelope_transfer(mel_src, tgt_mel, strength=1.0)
            wav_env = vocode_bigvgan(mel_env_only, meta.mel_min, meta.mel_max, vocoder, device)
            sf.write(str(eval_dir / f"{i:02d}_{src_name}_to_{tgt_name}_envonly.wav"),
                     wav_env[0], DIFFUSION_SR)

    print(f"\nAll files in: {eval_dir}")
    print("\nCompare these:")
    print("  *_source.wav           — original")
    print("  *_noenv.wav            — SDEdit style-only (no envelope swap)")
    print("  *_e0.5.wav             — envelope 50% + SDEdit")
    print("  *_e1.0.wav             — envelope 100% + SDEdit")
    print("  *_envonly.wav          — envelope only (no diffusion, pure signal processing)")


if __name__ == "__main__":
    main()
