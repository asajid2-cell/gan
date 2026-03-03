#!/usr/bin/env python3
"""Quick eval: generate a few diffusion V2 samples and vocode them.

Run this to hear what the model produces BEFORE committing to more training.
Defaults to run_d002 epoch_006 checkpoint (best perceived quality).

Usage:
  python quick_eval_diffusion.py
  python quick_eval_diffusion.py --device cpu   # if GPU is busy with training
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch.nn.functional as F

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parent

from src.lab3_diffusion_data import (
    DIFFUSION_SR, DiffusionMelDataset, load_diffusion_cache,
)
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import ddim_sample_v2, vocode_bigvgan, load_checkpoint
from src.lab3_data import stratified_group_split_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002"))
    p.add_argument("--checkpoint", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt"))
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ddim-steps", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of val samples to reconstruct")
    p.add_argument("--guidance-scales", nargs="+", type=float, default=[0.0, 1.0, 1.5, 2.0, 3.0])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cache_dir = run_dir / "cache"
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "quick_eval"
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
    print(f"Val set: {len(val_ds)} samples")

    # Load model from explicit checkpoint (fallback: best/latest)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "latest.pt"
    print(f"Loading checkpoint: {ckpt_path}")

    model = DiffusionUNetV2(
        in_channels=15, out_channels=1, base_ch=64,
        ch_mults=(1, 2, 4, 4), n_res=2, attn_levels=(2, 3),
        z_content_dim=128, z_style_dim=128, dropout=0.1,
    ).to(device)
    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=0.9999)
    ckpt = load_checkpoint(ckpt_path, model, ema, device=device)
    print(f"  epoch={ckpt.get('epoch','?')}, best_loss={ckpt.get('best_loss','?')}")

    # Load BigVGAN vocoder
    print("Loading BigVGAN vocoder...")
    import bigvgan as bvg
    import soundfile as sf
    vocoder = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    # Build index of genres for nice filenames
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}

    n = min(args.n_samples, len(val_ds))
    print(f"\nGenerating {n} samples x {len(args.guidance_scales)} guidance scales...")
    print(f"DDIM steps: {args.ddim_steps}")
    print(f"Guidance scales: {args.guidance_scales}\n")

    for i in range(n):
        batch = val_ds[i]
        genre_name = idx_to_genre.get(int(batch["genre_idx"].item()), "unk")

        # Ground truth vocoded
        mel_gt = batch["mel"].unsqueeze(0).to(device)
        wav_gt = vocode_bigvgan(mel_gt, meta.mel_min, meta.mel_max, vocoder, device)
        sf.write(str(eval_dir / f"{i:02d}_{genre_name}_GT.wav"), wav_gt[0], DIFFUSION_SR)

        cond_feat = batch["cond_feat"].unsqueeze(0).to(device)
        z_content = batch["z_content"].unsqueeze(0).to(device)
        z_style = batch["z_style"].unsqueeze(0).to(device)

        for w in args.guidance_scales:
            t0 = time.time()
            mel_gen = ddim_sample_v2(
                ema.shadow, schedule, cond_feat, z_content, z_style,
                n_steps=args.ddim_steps, guidance_scale=w, device=device,
            )
            wav_gen = vocode_bigvgan(mel_gen, meta.mel_min, meta.mel_max, vocoder, device)
            dt = time.time() - t0

            fname = f"{i:02d}_{genre_name}_w{w:.1f}.wav"
            sf.write(str(eval_dir / fname), wav_gen[0], DIFFUSION_SR)

            # Also save a smoothed version (reduce harsh high-freq artifacts)
            from src.lab3_diffusion_data import denormalize_mel
            mel_log = denormalize_mel(mel_gen.squeeze(1), meta.mel_min, meta.mel_max)
            # Mild temporal smoothing (3-frame avg) to reduce crackling
            kernel = torch.ones(1, 1, 1, 5, device=device) / 5.0
            mel_smooth = F.conv2d(mel_log.unsqueeze(1), kernel, padding=(0, 2))
            # Re-normalize back to [-1, 1] for vocoding
            span = meta.mel_max - meta.mel_min
            mel_smooth_norm = (2.0 * (mel_smooth.squeeze(1) - meta.mel_min) / span - 1.0)
            mel_smooth_norm = mel_smooth_norm.unsqueeze(1)  # [B, 1, 80, T]
            wav_smooth = vocode_bigvgan(mel_smooth_norm, meta.mel_min, meta.mel_max, vocoder, device)
            fname_s = f"{i:02d}_{genre_name}_w{w:.1f}_smooth.wav"
            sf.write(str(eval_dir / fname_s), wav_smooth[0], DIFFUSION_SR)

            print(f"  [{fname}] {dt:.1f}s")

    # Cross-genre style transfer: take sample 0, apply style from different genres
    print("\n--- Cross-genre style transfer ---")
    src = val_ds[0]
    src_genre = idx_to_genre.get(int(src["genre_idx"].item()), "unk")
    src_cond = src["cond_feat"].unsqueeze(0).to(device)
    src_zc = src["z_content"].unsqueeze(0).to(device)

    # Find exemplars from other genres
    for gi, gname in idx_to_genre.items():
        if gi == int(src["genre_idx"].item()):
            continue
        # Find a val sample from this genre
        for j in range(len(val_ds)):
            if int(val_ds.genre_idx[j]) == gi:
                tgt = val_ds[j]
                tgt_zs = tgt["z_style"].unsqueeze(0).to(device)
                mel_xfer = ddim_sample_v2(
                    ema.shadow, schedule, src_cond, src_zc, tgt_zs,
                    n_steps=args.ddim_steps, guidance_scale=3.0, device=device,
                )
                wav_xfer = vocode_bigvgan(mel_xfer, meta.mel_min, meta.mel_max, vocoder, device)
                fname = f"xfer_{src_genre}_to_{gname}.wav"
                sf.write(str(eval_dir / fname), wav_xfer[0], DIFFUSION_SR)
                print(f"  [{fname}]")
                break

    print(f"\nAll files saved to: {eval_dir}")
    print("Listen to GT vs w1/w3/w5 to judge quality.")
    print("Listen to xfer_* files to judge style transfer.")


if __name__ == "__main__":
    main()
