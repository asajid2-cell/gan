#!/usr/bin/env python3
"""Lab 3 — Diffusion mel spectrogram generator with BigVGAN vocoder.

Pipeline stages:
  0. BigVGAN quality check  — vocode 10 real mels, save audio for listening
  1. Build diffusion cache  — extract (mel, chroma, onset, beat, z_content, z_style) from ALL audio
  2. Train diffusion UNet   — DDPM + CFG + EMA + augmentation
  3. Evaluate               — DDIM sample → BigVGAN → metrics (style_conf, MPS, pitch_corr)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# -- Ensure lab 3/src is importable --
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from src.lab3_bridge import FrozenLab1Encoder, load_audio_chunk
from src.lab3_diffusion_data import (
    DIFFUSION_CHUNK_SEC,
    DIFFUSION_HOP,
    DIFFUSION_N_FRAMES,
    DIFFUSION_N_MELS,
    DIFFUSION_SR,
    DiffusionMelDataset,
    build_diffusion_cache,
    denormalize_mel,
    extract_bigvgan_mel_np,
    load_diffusion_cache,
)
from src.lab3_diffusion_model import DiffusionUNet, EMA, NoiseSchedule
from src.lab3_diffusion_train import (
    DiffusionTrainConfig,
    ddim_sample,
    load_checkpoint,
    pitch_correlation,
    save_checkpoint,
    train_one_epoch,
    vocode_bigvgan,
)
from src.lab3_data import stratified_group_split_indices


def _device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


# ===================================================================
# Stage 0: BigVGAN quality check
# ===================================================================

def stage0_bigvgan_check(args, device: torch.device):
    """Vocode 10 real mel spectrograms through BigVGAN and save audio."""
    import bigvgan as bvg
    import soundfile as sf

    out_dir = Path(args.out_dir) / "bigvgan_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STAGE 0: BigVGAN quality check")
    print("=" * 60)

    model = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    model.remove_weight_norm()
    model.eval().to(device)

    # Load a few real audio files
    from src.lab3_data import assign_genres, load_manifests
    df = load_manifests(Path(args.manifests_root), args.manifest_files)
    df = assign_genres(df)
    df = df[df["genre"] != "unassigned"].reset_index(drop=True)
    # Pick 10 samples (spread across genres)
    picks = df.groupby("genre").apply(lambda g: g.head(3)).reset_index(drop=True).head(10)

    for i, rec in picks.iterrows():
        p = Path(str(rec["path"]))
        if not p.exists():
            continue
        try:
            y = load_audio_chunk(p, sample_rate=DIFFUSION_SR, seconds=DIFFUSION_CHUNK_SEC)
            mel = extract_bigvgan_mel_np(y, sr=DIFFUSION_SR)  # [80, T]
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)  # [1, 80, T]
            with torch.no_grad():
                wav_recon = model(mel_t).squeeze(0).squeeze(0).cpu().numpy()  # [S]
            # Save original and reconstructed
            tag = f"{rec['genre']}_{i:03d}"
            sf.write(str(out_dir / f"{tag}_original.wav"), y, DIFFUSION_SR)
            sf.write(str(out_dir / f"{tag}_bigvgan_recon.wav"), wav_recon, DIFFUSION_SR)
            print(f"  [{tag}] original={len(y)} samples, recon={len(wav_recon)} samples  OK")
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")

    print(f"\nBigVGAN check files saved to: {out_dir}")
    print("Listen to the _original vs _bigvgan_recon pairs to verify quality.")
    del model
    torch.cuda.empty_cache()
    return out_dir


# ===================================================================
# Stage 1: Build diffusion cache
# ===================================================================

def stage1_build_cache(args, device: torch.device):
    """Build large-scale cache from all manifest audio."""
    cache_dir = Path(args.out_dir) / "cache"

    if (cache_dir / "diff_meta.json").exists() and not args.rebuild_cache:
        print(f"\n[stage1] Cache already exists at {cache_dir}, skipping (use --rebuild-cache to force)")
        return cache_dir

    print("\n" + "=" * 60)
    print("STAGE 1: Build diffusion cache")
    print("=" * 60)

    lab1 = FrozenLab1Encoder(checkpoint_path=Path(args.lab1_checkpoint), device=str(device))
    print(f"  Lab1 encoder loaded: sr={lab1.cfg.sample_rate}, z_dim={lab1.cfg.z_dim}")

    mert = None
    if args.use_mert:
        from src.lab3_mert_bridge import FrozenMERT
        mert = FrozenMERT(model_id=args.mert_model_id, device=str(device))
        print(f"  MERT loaded: sr={mert.cfg.sample_rate}, hidden={mert.cfg.hidden_size}")

    index_df, arrays, genre_to_idx, meta = build_diffusion_cache(
        manifests_root=Path(args.manifests_root),
        manifest_files=args.manifest_files,
        lab1_encoder=lab1,
        cache_dir=cache_dir,
        mert=mert,
        chunk_sec=DIFFUSION_CHUNK_SEC,
        max_chunks_per_track=int(args.max_chunks_per_track),
        seed=args.seed,
    )
    # build_diffusion_cache saves everything internally (shard-based)
    print(f"\n  Cache saved: {meta.n_samples} samples to {cache_dir}")

    del lab1, mert
    torch.cuda.empty_cache()
    return cache_dir


# ===================================================================
# Stage 2: Train
# ===================================================================

def stage2_train(args, device: torch.device, cache_dir: Path):
    """Train diffusion UNet."""
    print("\n" + "=" * 60)
    print("STAGE 2: Train diffusion model")
    print("=" * 60)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load cache
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    print(f"  Loaded cache: {meta.n_samples} samples, mel_range=[{meta.mel_min:.3f}, {meta.mel_max:.3f}]")

    # Split: stratified by genre, grouped by track_id (no leakage)
    genre_idx = np.asarray(arrays["genre_idx"])
    track_ids = index_df["track_id"].to_numpy()
    train_idx, val_idx = stratified_group_split_indices(
        genre_idx, track_ids, val_ratio=0.1, seed=args.seed,
    )
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}")

    train_ds = DiffusionMelDataset(
        arrays, train_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=True, seed=args.seed, style_source=args.style_source,
    )
    val_ds = DiffusionMelDataset(
        arrays, val_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=False, seed=args.seed + 1, style_source=args.style_source,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # Model
    z_style_dim = 768 if args.style_source == "mert" else 128
    model = DiffusionUNet(
        in_channels=15,  # 1 noisy mel + 12 chroma + 1 onset + 1 beat
        out_channels=1,
        base_ch=args.base_ch,
        ch_mults=tuple(args.ch_mults),
        n_res=args.n_res,
        attn_levels=tuple(args.attn_levels),
        z_content_dim=128,
        z_style_dim=z_style_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  UNet: {n_params/1e6:.2f}M parameters")

    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # AMP scaler for mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Using mixed precision (AMP)")

    train_cfg = DiffusionTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        cfg_dropout_p=args.cfg_dropout_p,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        save_every=args.save_every,
        warmup_steps=args.warmup_steps,
    )

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists() and not args.restart_training:
        print(f"  Resuming from {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, model, ema, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(f"  Resumed at epoch {start_epoch}, step {global_step}, best_loss={best_loss:.5f}")

    # -- Training loop --
    history = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        avg_loss, global_step = train_one_epoch(
            model, schedule, optimizer, train_loader, ema, train_cfg, device, epoch, global_step,
            scaler=scaler,
        )
        dt = time.time() - t0

        # -- Validation loss --
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                cond_feat = batch["cond_feat"].to(device)
                z_c = batch["z_content"].to(device)
                z_s = batch["z_style"].to(device)
                B = mel.shape[0]
                t = torch.randint(0, schedule.T, (B,), device=device)
                noise = torch.randn_like(mel)
                x_t = schedule.q_sample(mel, t, noise)
                unet_input = torch.cat([x_t, cond_feat], dim=1)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    eps_pred = ema.forward(unet_input, t, z_c, z_s)
                    vl = F.mse_loss(eps_pred, noise)
                val_loss += vl.item() * B
                n_val += B
        val_loss /= max(1, n_val)

        print(f"[epoch {epoch}] train_loss={avg_loss:.5f}  val_loss={val_loss:.5f}  "
              f"time={dt:.1f}s  step={global_step}")

        history.append({
            "epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss,
            "time_sec": dt, "global_step": global_step,
        })

        # Save
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(latest_ckpt, model, ema, optimizer, epoch, global_step, best_loss, scaler)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", model, ema, optimizer, epoch, global_step, best_loss, scaler)
        if (epoch + 1) % train_cfg.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", model, ema, optimizer, epoch, global_step, best_loss, scaler)

        # Save history
        with open(out_dir / "diffusion_history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete: {args.epochs} epochs, best_val_loss={best_loss:.5f}")
    return ckpt_dir


# ===================================================================
# Stage 3: Evaluate
# ===================================================================

def stage3_evaluate(args, device: torch.device, cache_dir: Path, ckpt_dir: Path):
    """Generate samples with CFG sweep, vocode with BigVGAN, compute metrics."""
    print("\n" + "=" * 60)
    print("STAGE 3: Evaluate (DDIM + BigVGAN + metrics)")
    print("=" * 60)

    import bigvgan as bvg
    import soundfile as sf

    out_dir = Path(args.out_dir)
    eval_dir = out_dir / "eval_samples"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load cache for val samples
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"])
    track_ids = index_df["track_id"].to_numpy()
    _, val_idx = stratified_group_split_indices(genre_idx, track_ids, val_ratio=0.1, seed=args.seed)

    val_ds = DiffusionMelDataset(
        arrays, val_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=False, seed=args.seed + 1, style_source=args.style_source,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False)

    # Load EMA model
    z_style_dim = 768 if args.style_source == "mert" else 128
    model = DiffusionUNet(
        in_channels=15, out_channels=1, base_ch=args.base_ch,
        ch_mults=tuple(args.ch_mults), n_res=args.n_res,
        attn_levels=tuple(args.attn_levels),
        z_content_dim=128, z_style_dim=z_style_dim, dropout=0.0,
    ).to(device)
    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=args.ema_decay)

    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        ckpt = load_checkpoint(best_ckpt, model, ema, device=device)
        print(f"  Loaded best checkpoint: epoch={ckpt.get('epoch', '?')}, loss={ckpt.get('best_loss', '?')}")
    else:
        ckpt = load_checkpoint(ckpt_dir / "latest.pt", model, ema, device=device)

    # BigVGAN vocoder
    vocoder = bvg.BigVGAN.from_pretrained("nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    # -- Style transfer evaluation --
    # For each val sample: generate with own z_style (reconstruction)
    # and with swapped z_style from a different genre (cross-genre transfer)
    guidance_scales = [float(w) for w in args.cfg_sweep]
    print(f"  CFG sweep: {guidance_scales}")
    print(f"  DDIM steps: {args.ddim_steps}")

    all_results = {}
    n_eval = min(args.n_eval_samples, len(val_ds))

    for w in guidance_scales:
        pitch_corrs = []
        generated_audios = []
        source_audios = []

        for batch_idx, batch in enumerate(val_loader):
            if batch_idx * args.eval_batch_size >= n_eval:
                break

            cond_feat = batch["cond_feat"].to(device)
            z_content = batch["z_content"].to(device)
            z_style = batch["z_style"].to(device)
            mel_gt = batch["mel"].to(device)

            # -- Reconstruction: same z_content + z_style --
            mel_gen = ddim_sample(
                ema.shadow, schedule, cond_feat, z_content, z_style,
                n_steps=args.ddim_steps, guidance_scale=w, device=device,
            )

            # Vocode both GT and generated
            wav_gen = vocode_bigvgan(mel_gen, meta.mel_min, meta.mel_max, vocoder, device)
            wav_gt = vocode_bigvgan(mel_gt, meta.mel_min, meta.mel_max, vocoder, device)

            for bi in range(wav_gen.shape[0]):
                idx_global = batch_idx * args.eval_batch_size + bi
                if idx_global >= n_eval:
                    break
                pc = pitch_correlation(wav_gt[bi], wav_gen[bi], sr=DIFFUSION_SR)
                pitch_corrs.append(pc)
                generated_audios.append(wav_gen[bi])
                source_audios.append(wav_gt[bi])

                # Save a few samples
                if idx_global < 8:
                    sf.write(str(eval_dir / f"w{w:.1f}_recon_{idx_global:03d}_gt.wav"),
                             wav_gt[bi], DIFFUSION_SR)
                    sf.write(str(eval_dir / f"w{w:.1f}_recon_{idx_global:03d}_gen.wav"),
                             wav_gen[bi], DIFFUSION_SR)

        result = {
            "guidance_scale": w,
            "n_eval": len(pitch_corrs),
            "mean_pitch_corr": float(np.mean(pitch_corrs)) if pitch_corrs else 0.0,
            "median_pitch_corr": float(np.median(pitch_corrs)) if pitch_corrs else 0.0,
        }

        # -- Style evaluation using Lab1 encoder --
        if generated_audios:
            lab1 = FrozenLab1Encoder(checkpoint_path=Path(args.lab1_checkpoint), device=str(device))
            from src.lab3_bridge import extract_log_mel, fix_log_mel_frames
            style_matches = 0
            content_coses = []
            n_style_eval = min(len(generated_audios), 64)
            for si in range(n_style_eval):
                try:
                    mel_src = extract_log_mel(source_audios[si], sr=DIFFUSION_SR)
                    mel_src = fix_log_mel_frames(mel_src, n_frames=256)
                    lat_src = lab1.infer_log_mel(mel_src)

                    mel_gen_ = extract_log_mel(generated_audios[si], sr=DIFFUSION_SR)
                    mel_gen_ = fix_log_mel_frames(mel_gen_, n_frames=256)
                    lat_gen = lab1.infer_log_mel(mel_gen_)

                    # Content similarity (MPS proxy)
                    cos_c = float(np.dot(lat_src["z_content"], lat_gen["z_content"]))
                    content_coses.append(cos_c)

                    # Style match
                    cos_s = float(np.dot(lat_src["z_style"], lat_gen["z_style"]))
                    if cos_s > 0.5:
                        style_matches += 1
                except Exception:
                    continue

            result["mps_mean"] = float(np.mean(content_coses)) if content_coses else 0.0
            result["style_acc"] = float(style_matches / max(1, n_style_eval))
            del lab1

        all_results[f"w={w}"] = result
        print(f"  [w={w:.1f}] pitch_corr={result['mean_pitch_corr']:.4f}  "
              f"mps={result.get('mps_mean', 0):.4f}  style_acc={result.get('style_acc', 0):.4f}")

    # -- Cross-genre transfer samples --
    print("\n  Generating cross-genre transfer samples...")
    idx_to_genre = {v: k for k, v in genre_to_idx.items()}
    # Build genre exemplar bank from val set
    genre_exemplars = {}
    for gi in np.unique(genre_idx[val_idx]):
        mask = genre_idx[val_idx] == gi
        exemplar_indices = np.where(mask)[0][:4]  # 4 exemplars per genre
        if len(exemplar_indices) > 0:
            genre_exemplars[int(gi)] = exemplar_indices

    # Take first 4 val samples, transfer to each other genre
    n_transfer = min(4, len(val_ds))
    for src_i in range(n_transfer):
        src_batch = val_ds[src_i]
        src_genre = int(src_batch["genre_idx"].item())
        src_cond = src_batch["cond_feat"].unsqueeze(0).to(device)
        src_zc = src_batch["z_content"].unsqueeze(0).to(device)
        src_mel = src_batch["mel"].unsqueeze(0).to(device)

        # Save source audio
        wav_src = vocode_bigvgan(src_mel, meta.mel_min, meta.mel_max, vocoder, device)
        sf.write(str(eval_dir / f"transfer_{src_i:02d}_source_{idx_to_genre.get(src_genre, src_genre)}.wav"),
                 wav_src[0], DIFFUSION_SR)

        for tgt_gi, tgt_exemplar_indices in genre_exemplars.items():
            if tgt_gi == src_genre:
                continue
            # Average z_style from target genre exemplars
            tgt_styles = []
            for ei in tgt_exemplar_indices:
                tgt_batch = val_ds[ei]
                tgt_styles.append(tgt_batch["z_style"].numpy())
            tgt_z_style = torch.from_numpy(np.mean(tgt_styles, axis=0)).unsqueeze(0).to(device)

            mel_transfer = ddim_sample(
                ema.shadow, schedule, src_cond, src_zc, tgt_z_style,
                n_steps=args.ddim_steps, guidance_scale=args.eval_guidance_scale,
                device=device,
            )
            wav_transfer = vocode_bigvgan(mel_transfer, meta.mel_min, meta.mel_max, vocoder, device)
            tgt_name = idx_to_genre.get(tgt_gi, str(tgt_gi))
            sf.write(str(eval_dir / f"transfer_{src_i:02d}_to_{tgt_name}.wav"),
                     wav_transfer[0], DIFFUSION_SR)

    # Save results
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nEval results saved to {out_dir / 'eval_results.json'}")
    print(f"Audio samples in {eval_dir}")

    del vocoder
    torch.cuda.empty_cache()


# ===================================================================
# CLI
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lab 3 Diffusion mel generator")

    # Modes
    p.add_argument("--stage", type=str, default="all",
                   choices=["bigvgan-check", "cache", "train", "eval", "all"],
                   help="Which stage to run")
    p.add_argument("--out-dir", type=str, required=True, help="Output directory for this run")

    # Data
    p.add_argument("--manifests-root", type=str, default="Z:\\DataSets\\_lab1_manifests")
    p.add_argument("--manifest-files", nargs="+", default=[
        "xtc_audio_clean.csv", "hh_lfbb_audio_clean.csv",
        "cc0_audio_clean.csv", "phase1_symbolic_audio_manifest.csv",
    ])
    p.add_argument("--lab1-checkpoint", type=str,
                   default="Z:\\328\\CMPUT328-A2\\codexworks\\301\\414-pl1\\saves\\lab1_run_combo_af_gate_exit_v2\\latest.pt")
    p.add_argument("--max-chunks-per-track", type=int, default=10)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--use-mert", action="store_true", help="Extract MERT features (slow)")
    p.add_argument("--mert-model-id", type=str, default="m-a-p/MERT-v1-95M")
    p.add_argument("--style-source", type=str, default="lab1", choices=["lab1", "mert"])
    p.add_argument("--seed", type=int, default=328)

    # Model
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--ch-mults", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--attn-levels", nargs="+", type=int, default=[2, 3])
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--cfg-dropout-p", type=float, default=0.10)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--restart-training", action="store_true")

    # Eval
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--cfg-sweep", nargs="+", type=float, default=[2.0, 3.0, 4.0, 5.0])
    p.add_argument("--eval-guidance-scale", type=float, default=4.0)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--n-eval-samples", type=int, default=64)

    # Hardware
    p.add_argument("--device", type=str, default="auto")

    return p


def main():
    args = build_parser().parse_args()
    device = _device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "diffusion_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    stage = args.stage
    cache_dir = out_dir / "cache"
    ckpt_dir = out_dir / "checkpoints"

    if stage in ("bigvgan-check", "all"):
        stage0_bigvgan_check(args, device)

    if stage in ("cache", "all"):
        cache_dir = stage1_build_cache(args, device)

    if stage in ("train", "all"):
        if not (cache_dir / "diff_meta.json").exists():
            print("ERROR: Cache not found. Run --stage cache first.")
            sys.exit(1)
        stage2_train(args, device, cache_dir)

    if stage in ("eval", "all"):
        if not ckpt_dir.exists():
            print("ERROR: No checkpoints found. Run --stage train first.")
            sys.exit(1)
        stage3_evaluate(args, device, cache_dir, ckpt_dir)


if __name__ == "__main__":
    main()
