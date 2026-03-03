#!/usr/bin/env python3
"""SDEdit-style mel style transfer using Diffusion V2 (run_d002).

Instead of generating from pure noise (which sounds harsh), we:
  1. Take a real mel spectrogram (content source)
  2. Forward-diffuse it to an intermediate timestep t_start
  3. Denoise from t_start → 0 with the TARGET style embedding

t_start controls content/style tradeoff:
  - Low  (200-300): preserves most content, subtle style change
  - Mid  (400-500): balanced
  - High (600-700): more style change, less content fidelity

Usage:
  python quick_eval_sdedit.py
  python quick_eval_sdedit.py --t-starts 200 400 600
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


@torch.no_grad()
def sdedit_transfer(
    model: torch.nn.Module,
    schedule: NoiseSchedule,
    mel_source: torch.Tensor,    # [B, 1, 80, 432] normalized [-1, 1]
    cond_feat: torch.Tensor,     # [B, 14, 80, 432]
    z_content: torch.Tensor,     # [B, 128]
    z_style: torch.Tensor,       # [B, 128]  — TARGET style
    t_start: int = 400,
    n_ddim_steps: int = 50,
    guidance_scale: float = 1.5,
    device: torch.device = None,
) -> torch.Tensor:
    """SDEdit: forward-diffuse source mel to t_start, then denoise with target style."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    B = mel_source.shape[0]

    # 1. Forward diffuse source mel to t_start
    noise = torch.randn_like(mel_source)
    t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
    x_t = schedule.q_sample(mel_source, t_tensor, noise)

    # 2. Build DDIM timestep sub-sequence from t_start down to 0
    # Only denoise from t_start, not from T=1000
    full_step_size = schedule.T // n_ddim_steps
    # Get all DDIM timesteps <= t_start, descending
    all_steps = list(range(0, schedule.T, full_step_size))
    timesteps = sorted([t for t in all_steps if t <= t_start], reverse=True)

    # 3. DDIM reverse from t_start to 0
    for i, t_cur in enumerate(timesteps):
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)

        unet_input = torch.cat([x_t, cond_feat], dim=1)

        v_cond = model(unet_input, t_batch, z_content, z_style)

        if guidance_scale != 1.0 and guidance_scale != 0.0:
            z_zero = torch.zeros_like(z_content)
            v_uncond = model(unet_input, t_batch, z_zero, z_zero)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = v_cond

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
    p.add_argument("--t-starts", nargs="+", type=int, default=[200, 350, 500],
                   help="Noise levels to test (lower=more content, higher=more style)")
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=4)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = run_dir / "cache"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "sdedit_eval"
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
    print(f"Val set: {len(val_ds)} samples, genres: {list(genre_to_idx.keys())}")

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
    print(f"Loaded {ckpt_path.name}: epoch={ckpt.get('epoch','?')}, loss={ckpt.get('best_loss','?')}")

    # Load BigVGAN
    print("Loading BigVGAN vocoder...")
    import bigvgan as bvg
    import soundfile as sf
    vocoder = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    # ---- Test 1: Reconstruction (same style) at different noise levels ----
    print(f"\n{'='*60}")
    print("TEST 1: Reconstruction quality at different noise levels")
    print(f"  t_starts = {args.t_starts}")
    print(f"  guidance = {args.guidance_scale}")
    print(f"{'='*60}\n")

    for i in range(min(args.n_samples, len(val_ds))):
        batch = val_ds[i]
        genre_name = idx_to_genre.get(int(batch["genre_idx"].item()), "unk")

        mel_src = batch["mel"].unsqueeze(0).to(device)
        cond_feat = batch["cond_feat"].unsqueeze(0).to(device)
        z_c = batch["z_content"].unsqueeze(0).to(device)
        z_s = batch["z_style"].unsqueeze(0).to(device)

        # Save ground truth
        wav_gt = vocode_bigvgan(mel_src, meta.mel_min, meta.mel_max, vocoder, device)
        sf.write(str(eval_dir / f"{i:02d}_{genre_name}_GT.wav"), wav_gt[0], DIFFUSION_SR)

        for t_start in args.t_starts:
            t0 = time.time()
            mel_recon = sdedit_transfer(
                ema.shadow, schedule, mel_src, cond_feat, z_c, z_s,
                t_start=t_start, n_ddim_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale, device=device,
            )
            wav_recon = vocode_bigvgan(mel_recon, meta.mel_min, meta.mel_max, vocoder, device)
            dt = time.time() - t0

            fname = f"{i:02d}_{genre_name}_recon_t{t_start}.wav"
            sf.write(str(eval_dir / fname), wav_recon[0], DIFFUSION_SR)
            print(f"  [{fname}] {dt:.1f}s")

    # ---- Test 2: Cross-genre style transfer ----
    print(f"\n{'='*60}")
    print("TEST 2: Cross-genre style transfer")
    print(f"{'='*60}\n")

    # Build a style exemplar bank: average z_style per genre from val set
    genre_styles = {}
    for gi, gname in idx_to_genre.items():
        styles = []
        for j in range(min(len(val_ds), 2000)):
            if int(val_ds.genre_idx[j]) == gi:
                s = val_ds[j]["z_style"].numpy()
                styles.append(s)
                if len(styles) >= 20:
                    break
        if styles:
            genre_styles[gi] = torch.from_numpy(np.mean(styles, axis=0)).unsqueeze(0).to(device)
            print(f"  Genre '{gname}': averaged {len(styles)} style vectors")

    # Transfer first few samples to other genres
    best_t = args.t_starts[len(args.t_starts) // 2]  # use middle t_start
    print(f"\n  Using t_start={best_t} for transfers\n")

    for i in range(min(2, len(val_ds))):
        batch = val_ds[i]
        src_genre_idx = int(batch["genre_idx"].item())
        src_genre_name = idx_to_genre.get(src_genre_idx, "unk")

        mel_src = batch["mel"].unsqueeze(0).to(device)
        cond_feat = batch["cond_feat"].unsqueeze(0).to(device)
        z_c = batch["z_content"].unsqueeze(0).to(device)

        for tgt_gi, tgt_style in genre_styles.items():
            if tgt_gi == src_genre_idx:
                continue
            tgt_name = idx_to_genre.get(tgt_gi, "unk")

            mel_xfer = sdedit_transfer(
                ema.shadow, schedule, mel_src, cond_feat, z_c, tgt_style,
                t_start=best_t, n_ddim_steps=args.ddim_steps,
                guidance_scale=args.guidance_scale, device=device,
            )
            wav_xfer = vocode_bigvgan(mel_xfer, meta.mel_min, meta.mel_max, vocoder, device)
            fname = f"xfer_{i:02d}_{src_genre_name}_to_{tgt_name}_t{best_t}.wav"
            sf.write(str(eval_dir / fname), wav_xfer[0], DIFFUSION_SR)
            print(f"  [{fname}]")

    print(f"\nAll files in: {eval_dir}")
    print("\nWhat to listen for:")
    print("  - GT vs recon_t200: should sound very similar (low noise = faithful)")
    print("  - GT vs recon_t350: moderate reconstruction")
    print("  - GT vs recon_t500: more diffused, model has more creative freedom")
    print("  - xfer_*: source melody + target genre style")


if __name__ == "__main__":
    main()
