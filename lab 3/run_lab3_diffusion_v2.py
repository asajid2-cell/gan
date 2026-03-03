#!/usr/bin/env python3
"""Lab 3 — Diffusion V2: v-prediction + StyleAdaIN + grad accum + cosine LR.

Reuses existing cache from run_d001. Key improvements:
  - v-prediction instead of epsilon
  - Separate style AdaIN (not mixed into FiLM)
  - Gradient accumulation (effective batch 16)
  - Cosine LR with warmup
  - 3-second chunks (256 frames) instead of 5s
  - Attention at levels 2+3
  - Sample generation at end of every epoch
  - Checkpoints saved every epoch + every 500 steps
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
    generate_epoch_samples,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch_v2,
)
from src.lab3_data import stratified_group_split_indices


def main():
    p = argparse.ArgumentParser(description="Diffusion V2 training")

    # Paths
    p.add_argument("--cache-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"))
    p.add_argument("--out-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002"))

    # Model
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--ch-mults", nargs="+", type=int, default=[1, 2, 4, 4])
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--attn-levels", nargs="+", type=int, default=[2, 3])
    p.add_argument("--dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max-frames", type=int, default=256,
                   help="Trim mel to this many frames (256 = ~3s)")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--cfg-dropout-p", type=float, default=0.10)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=328)

    # Eval
    p.add_argument("--epoch-samples", type=int, default=4,
                   help="Number of samples to generate at end of each epoch")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=2.0)

    # Hardware
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--restart", action="store_true", help="Ignore existing checkpoint")

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
    with open(out_dir / "v2_config.json", "w") as f:
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

    # ---- Model ----
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  DiffusionUNetV2: {n_params / 1e6:.2f}M parameters")

    schedule = NoiseSchedule(T=1000).to(device)
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # AMP
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("  Mixed precision (AMP) enabled")

    # Cosine LR with warmup
    # Total optimizer steps = (len(train_loader) / grad_accum) * epochs
    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    print(f"  Steps/epoch={steps_per_epoch}, total={total_steps}, warmup={args.warmup_steps}")

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(args.warmup_steps)
        progress = float(step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_cfg = DiffusionTrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        ema_decay=args.ema_decay,
        cfg_dropout_p=args.cfg_dropout_p,
        grad_clip_norm=args.grad_clip_norm,
        log_every=args.log_every,
        warmup_steps=args.warmup_steps,  # handled by scheduler now
    )

    # ---- Resume ----
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists() and not args.restart:
        print(f"  Resuming from {latest_ckpt}")
        ckpt = load_checkpoint(latest_ckpt, model, ema, optimizer, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        # Advance scheduler to correct position
        for _ in range(global_step // args.grad_accum):
            scheduler.step()
        print(f"  Resumed: epoch={start_epoch}, step={global_step}, best_loss={best_loss:.5f}")

    # ---- Save callback ----
    def save_fn(step):
        save_checkpoint(latest_ckpt, model, ema, optimizer, epoch, step, best_loss, scaler)

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"V2 TRAINING: {args.epochs} epochs, effective batch={args.batch_size * args.grad_accum}")
    print(f"{'='*60}\n")

    history = []
    # Load existing history if resuming
    hist_path = out_dir / "v2_history.json"
    if hist_path.exists() and not args.restart:
        with open(hist_path) as f:
            history = json.load(f)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        avg_loss, global_step = train_one_epoch_v2(
            model, schedule, optimizer, train_loader, ema, train_cfg, device,
            epoch, global_step, scaler=scaler,
            grad_accum_steps=args.grad_accum, scheduler=scheduler,
            save_fn=save_fn,
        )
        dt = time.time() - t0

        # ---- Validation (v-prediction) ----
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
                v_target = schedule.compute_v_target(mel, t, noise)
                unet_input = torch.cat([x_t, cond_feat], dim=1)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    v_pred = ema.forward(unet_input, t, z_c, z_s)
                    vl = F.mse_loss(v_pred, v_target)
                val_loss += vl.item() * B
                n_val += B
        val_loss /= max(1, n_val)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\n[epoch {epoch}] train={avg_loss:.5f}  val={val_loss:.5f}  "
              f"lr={lr_now:.2e}  time={dt:.0f}s  step={global_step}")

        history.append({
            "epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss,
            "lr": lr_now, "time_sec": dt, "global_step": global_step,
        })

        # ---- Save checkpoints ----
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
        save_checkpoint(latest_ckpt, model, ema, optimizer, epoch, global_step, best_loss, scaler)
        save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", model, ema, optimizer,
                        epoch, global_step, best_loss, scaler)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", model, ema, optimizer,
                            epoch, global_step, best_loss, scaler)
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

    print(f"\nTraining complete. Best val loss: {best_loss:.5f}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Samples: {out_dir / 'epoch_samples'}")


if __name__ == "__main__":
    main()
