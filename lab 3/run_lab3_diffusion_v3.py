#!/usr/bin/env python3
"""Lab 3 — Diffusion V3: adversarial fine-tuning from V2 epoch 6.

Key improvements over V2:
  - Fine-tunes from V2 epoch 6 (best perceptual quality)
  - Adds HybridMelDiscriminator (hinge loss + feature matching)
  - Lower LR (1e-4) and faster EMA (0.999)
  - Higher CFG dropout (0.15) for stronger regularization
  - Discriminator warmup: first 500 steps are pure MSE
  - Generates 8 samples per epoch for quality monitoring
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parent

from src.lab3_diffusion_data import (
    DIFFUSION_SR,
    DiffusionMelDataset,
    load_diffusion_cache,
)
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import (
    DiffusionTrainConfig,
    ddim_sample_v2,
    generate_epoch_samples,
    save_checkpoint,
    train_one_epoch_v3,
    vocode_bigvgan,
)
from src.lab3_models import HybridMelDiscriminator
from src.lab3_data import stratified_group_split_indices


def main():
    p = argparse.ArgumentParser(description="Diffusion V3: adversarial fine-tuning")

    # Paths
    p.add_argument("--cache-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"))
    p.add_argument("--out-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d003"))
    p.add_argument("--v2-checkpoint", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt"),
                   help="V2 checkpoint to fine-tune from")

    # Model (same architecture as V2)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--ch-mults", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--attn-levels", nargs="+", type=int, default=[2, 3])
    p.add_argument("--dropout", type=float, default=0.1)

    # Discriminator
    p.add_argument("--disc-num-scales", type=int, default=3)
    p.add_argument("--disc-periods", nargs="+", type=int, default=[1, 2, 3, 5])
    p.add_argument("--disc-spectral-norm", action="store_true")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Generator LR (half of V2)")
    p.add_argument("--disc-lr", type=float, default=2e-4,
                   help="Discriminator LR")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=256)
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="Faster EMA tracking (10x less lag than V2's 0.9999)")
    p.add_argument("--cfg-dropout-p", type=float, default=0.15,
                   help="Stronger CFG dropout (V2 used 0.10)")
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--disc-warmup-steps", type=int, default=500,
                   help="Steps of pure MSE before activating adversarial loss")
    p.add_argument("--adv-weight", type=float, default=0.1)
    p.add_argument("--fm-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=328)

    # Eval
    p.add_argument("--epoch-samples", type=int, default=8,
                   help="More samples per epoch for quality monitoring")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=2.0)

    # Hardware
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--restart", action="store_true",
                   help="Ignore existing V3 checkpoint and restart from V2")

    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "v3_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Load cache ----
    cache_dir = Path(args.cache_dir)
    print(f"Loading cache from {cache_dir} ...")
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    print(f"  {meta.n_samples} samples, mel=[{meta.mel_min:.3f}, {meta.mel_max:.3f}]")

    # ---- Split ----
    genre_idx = np.asarray(arrays["genre_idx"])
    track_ids = index_df["track_id"].to_numpy()
    train_idx, val_idx = stratified_group_split_indices(
        genre_idx, track_ids, val_ratio=0.1, seed=args.seed)
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}")

    # ---- Datasets ----
    train_ds = DiffusionMelDataset(
        arrays, train_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=True, seed=args.seed, style_source="lab1",
        max_frames=args.max_frames,
    )
    val_ds = DiffusionMelDataset(
        arrays, val_idx, mel_min=meta.mel_min, mel_max=meta.mel_max,
        augment=False, seed=args.seed + 1, style_source="lab1",
        max_frames=args.max_frames,
    )
    print(f"  max_frames={args.max_frames} ({args.max_frames * 256 / 22050:.1f}s)")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ---- Generator (same architecture as V2) ----
    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=args.base_ch,
        ch_mults=tuple(args.ch_mults),
        n_res=args.n_res,
        attn_levels=tuple(args.attn_levels),
        z_content_dim=128,
        z_style_dim=128,
        dropout=args.dropout,
    ).to(device)

    n_gen_params = sum(p.numel() for p in model.parameters())
    print(f"  DiffusionUNetV2: {n_gen_params / 1e6:.2f}M parameters")

    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=args.ema_decay)

    # ---- Discriminator ----
    discriminator = HybridMelDiscriminator(
        cond_dim=0,
        num_scales=args.disc_num_scales,
        periods=tuple(args.disc_periods),
        spectral_norm=args.disc_spectral_norm,
    ).to(device)

    n_disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  HybridMelDiscriminator: {n_disc_params / 1e6:.2f}M parameters")

    # ---- Optimizers ----
    gen_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    disc_optimizer = torch.optim.AdamW(
        discriminator.parameters(), lr=args.disc_lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # AMP
    use_amp = device.type == "cuda"
    gen_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    disc_scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled (separate scalers for gen/disc)")

    # Cosine LR with warmup (generator)
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_gen_steps = steps_per_epoch * args.epochs
    print(f"  Gen steps/epoch={steps_per_epoch}, total={total_gen_steps}, warmup={args.warmup_steps}")

    def gen_lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        progress = float(step - args.warmup_steps) / float(max(1, total_gen_steps - args.warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, gen_lr_lambda)

    # Cosine LR for discriminator (no warmup, steps every batch)
    total_disc_steps = len(train_loader) * args.epochs

    def disc_lr_lambda(step):
        progress = float(step) / float(max(1, total_disc_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, disc_lr_lambda)

    train_cfg = DiffusionTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        cfg_dropout_p=args.cfg_dropout_p,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        warmup_steps=args.warmup_steps,
    )

    # ---- Load V2 checkpoint or resume V3 ----
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    latest_v3_ckpt = ckpt_dir / "latest.pt"
    if latest_v3_ckpt.exists() and not args.restart:
        # Resume V3 training
        print(f"  Resuming V3 from {latest_v3_ckpt}")
        ckpt = torch.load(str(latest_v3_ckpt), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        gen_optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if gen_scaler is not None and "scaler" in ckpt:
            gen_scaler.load_state_dict(ckpt["scaler"])
        # Load discriminator state
        if "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
        if "disc_optimizer" in ckpt:
            disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        if disc_scaler is not None and "disc_scaler" in ckpt:
            disc_scaler.load_state_dict(ckpt["disc_scaler"])
        # Advance schedulers
        for _ in range(global_step // args.grad_accum):
            gen_scheduler.step()
        for _ in range(global_step):
            disc_scheduler.step()
        print(f"  Resumed: epoch={start_epoch}, step={global_step}, best_loss={best_loss:.5f}")
    else:
        # Load V2 epoch 6 checkpoint
        v2_ckpt_path = Path(args.v2_checkpoint)
        if not v2_ckpt_path.exists():
            print(f"ERROR: V2 checkpoint not found: {v2_ckpt_path}")
            sys.exit(1)
        print(f"  Loading V2 checkpoint: {v2_ckpt_path}")
        v2_ckpt = torch.load(str(v2_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(v2_ckpt["model"])
        ema.load_state_dict(v2_ckpt["ema"])
        # Don't load V2 optimizer — we use a fresh one with lower LR
        v2_epoch = v2_ckpt.get("epoch", "?")
        v2_step = v2_ckpt.get("global_step", "?")
        print(f"  Loaded V2 epoch={v2_epoch}, step={v2_step}")
        print(f"  Discriminator initialized from scratch")

    # ---- Extended save function (includes disc state) ----
    def save_v3_checkpoint(path, ep, step, b_loss):
        ckpt = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": gen_optimizer.state_dict(),
            "discriminator": discriminator.state_dict(),
            "disc_optimizer": disc_optimizer.state_dict(),
            "epoch": ep,
            "global_step": step,
            "best_loss": b_loss,
        }
        if gen_scaler is not None:
            ckpt["scaler"] = gen_scaler.state_dict()
        if disc_scaler is not None:
            ckpt["disc_scaler"] = disc_scaler.state_dict()
        torch.save(ckpt, str(path))

    def save_fn(step):
        save_v3_checkpoint(latest_v3_ckpt, epoch, step, best_loss)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"V3 ADVERSARIAL TRAINING: {args.epochs} epochs, "
          f"effective batch={args.batch_size * args.grad_accum}")
    print(f"  adv_weight={args.adv_weight}, fm_weight={args.fm_weight}, "
          f"disc_warmup={args.disc_warmup_steps} steps")
    print(f"{'='*60}\n")

    history = []
    hist_path = out_dir / "v3_history.json"
    if hist_path.exists() and not args.restart:
        with open(hist_path) as f:
            history = json.load(f)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        avg_gen, avg_disc, avg_mse, global_step = train_one_epoch_v3(
            model, schedule, gen_optimizer, train_loader, ema, train_cfg, device,
            epoch, global_step,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
            scaler=gen_scaler,
            disc_scaler=disc_scaler,
            grad_accum_steps=args.grad_accum,
            scheduler=gen_scheduler,
            disc_scheduler=disc_scheduler,
            adv_weight=args.adv_weight,
            fm_weight=args.fm_weight,
            disc_warmup_steps=args.disc_warmup_steps,
            save_fn=save_fn,
        )
        dt = time.time() - t0

        # ---- Validation (v-prediction MSE only) ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                vmel = batch["mel"].to(device)
                vcond = batch["cond_feat"].to(device)
                vz_c = batch["z_content"].to(device)
                vz_s = batch["z_style"].to(device)
                vB = vmel.shape[0]
                vt = torch.randint(0, schedule.T, (vB,), device=device)
                vnoise = torch.randn_like(vmel)
                vx_t = schedule.q_sample(vmel, vt, vnoise)
                vv_target = schedule.compute_v_target(vmel, vt, vnoise)
                vunet_input = torch.cat([vx_t, vcond], dim=1)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vv_pred = ema.forward(vunet_input, vt, vz_c, vz_s)
                    vl = F.mse_loss(vv_pred, vv_target)
                val_loss += vl.item() * vB
                n_val += vB
        val_loss /= max(1, n_val)

        gen_lr = gen_optimizer.param_groups[0]["lr"]
        disc_lr = disc_optimizer.param_groups[0]["lr"]
        print(f"\n[epoch {epoch}] gen={avg_gen:.5f}  mse={avg_mse:.5f}  disc={avg_disc:.5f}  "
              f"val={val_loss:.5f}  gen_lr={gen_lr:.2e}  disc_lr={disc_lr:.2e}  "
              f"time={dt:.0f}s  step={global_step}")

        history.append({
            "epoch": epoch, "gen_loss": avg_gen, "mse_loss": avg_mse,
            "disc_loss": avg_disc, "val_loss": val_loss,
            "gen_lr": gen_lr, "disc_lr": disc_lr,
            "time_sec": dt, "global_step": global_step,
        })

        # ---- Save checkpoints ----
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_v3_checkpoint(latest_v3_ckpt, epoch, global_step, best_loss)
        save_v3_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, global_step, best_loss)
        if is_best:
            save_v3_checkpoint(ckpt_dir / "best.pt", epoch, global_step, best_loss)
            print(f"  ** New best: {best_loss:.5f}")

        # Save history
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        # ---- Generate epoch samples ----
        if args.epoch_samples > 0:
            print(f"  Generating {args.epoch_samples} samples...")
            try:
                generate_epoch_samples(
                    ema, schedule, val_ds, genre_to_idx,
                    meta.mel_min, meta.mel_max, device, epoch, out_dir,
                    n_samples=args.epoch_samples, ddim_steps=args.ddim_steps,
                    guidance_scale=args.guidance_scale,
                )
            except Exception as e:
                print(f"  [sample gen failed: {e}]")

    print(f"\nV3 Training complete. Best val loss: {best_loss:.5f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Samples: {out_dir / 'epoch_samples'}")


if __name__ == "__main__":
    main()
