from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.lab3_bridge import FrozenLab1Encoder
from src.lab3_codec_bridge import FrozenEncodec
from src.lab3_codec_data import (
    DEFAULT_MANIFESTS,
    CachedCodecDataset,
    assign_genres,
    build_codec_cache,
    load_codec_cache,
    load_manifests,
    materialize_genre_samples,
    save_codec_cache,
    stratified_group_split_indices,
    stratified_split_indices,
)
from src.lab3_codec_models import CodecLatentTranslator, CodecTrainWeights, MultiScaleWaveDiscriminator
from src.lab3_codec_train import (
    CodecStageTrainConfig,
    build_q_exemplar_bank,
    build_style_centroid_bank,
    build_style_exemplar_bank,
    train_codec_stage,
)
from src.lab3_sampling import resolve_next_run_name, validate_run_name

try:
    import soundfile as sf

    HAS_SF = True
except Exception:
    HAS_SF = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_out_root() -> Path:
    return _repo_root() / "saves2" / "lab3_codec_transfer"


def _default_lab1_checkpoint() -> Path:
    root = _repo_root()
    candidates = [
        root / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        root / "saves" / "lab1_run_combo_af_gate" / "latest.pt",
        root / "saves" / "lab1_run_a" / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _save_json(obj: Dict, path: Path) -> None:
    def _default(v):
        if isinstance(v, Path):
            return str(v)
        return v

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_history(csv_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        out = df_new
    out.to_csv(csv_path, index=False)


def _load_models_from_ckpt(
    ckpt_path: Path,
    generator: CodecLatentTranslator,
    discriminator: MultiScaleWaveDiscriminator,
    device: torch.device,
) -> bool:
    if not Path(ckpt_path).exists():
        return False
    try:
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(ckpt_path), map_location=device)
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab3 codec-latent style transfer training (EnCodec + Lab1 content).")
    p.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    p.add_argument("--out-root", type=Path, default=_default_out_root())
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--strict-run-naming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-custom-run-name", action="store_true")
    p.add_argument("--resume-dir", type=Path, default=None)
    p.add_argument("--reuse-cache-dir", type=Path, default=None)

    p.add_argument("--manifests-root", type=Path, default=Path("Z:/DataSets/_lab1_manifests"))
    p.add_argument("--manifest-files", nargs="*", default=DEFAULT_MANIFESTS)
    p.add_argument("--per-genre-samples", type=int, default=600)
    p.add_argument("--chunks-per-track", type=int, default=2)
    p.add_argument("--chunk-sampling", choices=["uniform", "random"], default="uniform")
    p.add_argument("--min-start-sec", type=float, default=0.0)
    p.add_argument("--max-start-sec", type=float, default=None)
    p.add_argument("--split-by-track", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=328)

    p.add_argument("--lab1-checkpoint", type=Path, default=_default_lab1_checkpoint())
    p.add_argument("--lab1-n-frames", type=int, default=256)
    p.add_argument("--codec-model-id", type=str, default="facebook/encodec_24khz")
    p.add_argument("--codec-bandwidth", type=float, default=6.0)
    p.add_argument("--codec-chunk-seconds", type=float, default=5.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--translator-hidden-channels", type=int, default=256)
    p.add_argument("--translator-blocks", type=int, default=10)
    p.add_argument("--translator-noise-dim", type=int, default=32)
    p.add_argument("--translator-residual-scale", type=float, default=0.5)
    p.add_argument("--discriminator-scales", type=int, default=3)

    p.add_argument("--stage1-epochs", type=int, default=8)
    p.add_argument("--stage2-epochs", type=int, default=16)
    p.add_argument("--stage3-epochs", type=int, default=8)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-stage3", action="store_true")

    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--stage2-d-lr-mult", type=float, default=0.2)
    p.add_argument("--stage3-d-lr-mult", type=float, default=0.2)
    p.add_argument("--adv-weight", type=float, default=0.4)
    p.add_argument("--feature-match-weight", type=float, default=1.0)
    p.add_argument("--latent-l1-weight", type=float, default=4.0)
    p.add_argument("--continuity-weight", type=float, default=1.0)
    p.add_argument("--content-weight", type=float, default=2.0)
    p.add_argument("--style-weight", type=float, default=3.0)
    p.add_argument("--mrstft-weight", type=float, default=2.0)
    p.add_argument("--mode-seeking-weight", type=float, default=1.0)

    p.add_argument("--stage1-adv-weight", type=float, default=0.0)
    p.add_argument("--stage1-style-weight", type=float, default=0.5)
    p.add_argument("--stage1-content-weight", type=float, default=1.0)
    p.add_argument("--stage1-mrstft-weight", type=float, default=2.0)
    p.add_argument("--stage1-latent-l1-weight", type=float, default=6.0)

    p.add_argument("--stage2-cond-mode", choices=["centroid", "exemplar", "mix"], default="mix")
    p.add_argument("--stage2-cond-alpha-start", type=float, default=0.8)
    p.add_argument("--stage2-cond-alpha-end", type=float, default=0.4)
    p.add_argument("--stage2-target-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage2-style-dropout-p", type=float, default=0.0)
    p.add_argument("--stage2-style-jitter-std", type=float, default=0.03)
    p.add_argument("--stage2-exemplar-noise-std", type=float, default=0.03)

    p.add_argument("--stage3-cond-mode", choices=["centroid", "exemplar", "mix"], default="mix")
    p.add_argument("--stage3-cond-alpha-start", type=float, default=0.5)
    p.add_argument("--stage3-cond-alpha-end", type=float, default=0.2)
    p.add_argument("--stage3-target-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage3-style-dropout-p", type=float, default=0.25)
    p.add_argument("--stage3-style-jitter-std", type=float, default=0.08)
    p.add_argument("--stage3-exemplar-noise-std", type=float, default=0.05)

    p.add_argument("--r1-gamma", type=float, default=1.0)
    p.add_argument("--r1-interval", type=int, default=16)
    p.add_argument("--stage1-d-step-period", type=int, default=1)
    p.add_argument("--stage2-d-step-period", type=int, default=2)
    p.add_argument("--stage3-d-step-period", type=int, default=2)
    p.add_argument("--g-grad-clip-norm", type=float, default=5.0)
    p.add_argument("--d-grad-clip-norm", type=float, default=5.0)

    p.add_argument("--auto-sample-export", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-count", type=int, default=24)
    p.add_argument("--sample-export-tag", type=str, default="posttrain_samples")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def _export_codec_samples(
    out_dir: Path,
    generator: CodecLatentTranslator,
    codec: FrozenEncodec,
    arrays: Dict[str, np.ndarray],
    val_idx: np.ndarray,
    style_centroid_bank: torch.Tensor,
    style_exemplar_bank: Dict[int, torch.Tensor],
    seed: int = 328,
    n_samples: int = 24,
    cond_mode: str = "mix",
    cond_alpha: float = 0.35,
) -> Optional[Path]:
    if not HAS_SF:
        print("[sample-export] soundfile not installed; skipping export.")
        return None
    if len(val_idx) == 0:
        return None
    sample_dir = Path(out_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    n_genres = int(style_centroid_bank.shape[0])
    rows: List[Dict] = []
    device = next(generator.parameters()).device
    generator.eval()
    for i in range(int(n_samples)):
        ridx = int(val_idx[int(rng.integers(0, len(val_idx)))])
        src_genre = int(arrays["genre_idx"][ridx])
        tgt_genre = int(rng.integers(0, n_genres))
        if tgt_genre == src_genre and n_genres > 1:
            tgt_genre = int((tgt_genre + 1) % n_genres)
        q_src = torch.from_numpy(arrays["q_emb"][ridx : ridx + 1]).to(device).float()
        zc = torch.from_numpy(arrays["z_content"][ridx : ridx + 1]).to(device).float()

        z_cent = style_centroid_bank[tgt_genre : tgt_genre + 1].to(device).float()
        z_ex_bank = style_exemplar_bank.get(int(tgt_genre))
        if z_ex_bank is None or len(z_ex_bank) == 0:
            z_ex = z_cent
        else:
            ex_i = int(rng.integers(0, int(z_ex_bank.shape[0])))
            z_ex = z_ex_bank[ex_i : ex_i + 1].to(device).float()
        if str(cond_mode) == "centroid":
            z_tgt = z_cent
        elif str(cond_mode) == "exemplar":
            z_tgt = z_ex
        else:
            z_tgt = float(cond_alpha) * z_cent + (1.0 - float(cond_alpha)) * z_ex
        z_tgt = torch.nn.functional.normalize(z_tgt, dim=-1)

        with torch.no_grad():
            q_hat = generator(q_src=q_src, z_content=zc, z_style_tgt=z_tgt)
            wav = codec.decode_embeddings(q_hat)[0, 0].detach().cpu().numpy().astype(np.float32)
            wav = wav / (np.max(np.abs(wav)) + 1e-8)
        out_wav = sample_dir / f"sample_{i:04d}_src{src_genre}_tgt{tgt_genre}.wav"
        sf.write(str(out_wav), wav, int(codec.cfg.sample_rate))
        rows.append(
            {
                "sample_id": int(i),
                "cache_row": int(ridx),
                "source_genre_idx": int(src_genre),
                "target_genre_idx": int(tgt_genre),
                "fake_wav": str(out_wav),
            }
        )
    pd.DataFrame(rows).to_csv(sample_dir / "generation_summary.csv", index=False)
    return sample_dir


def main() -> None:
    args = _parse_args()
    if args.smoke:
        args.per_genre_samples = min(int(args.per_genre_samples), 48)
        args.chunks_per_track = 1
        args.batch_size = min(int(args.batch_size), 6)
        args.stage1_epochs = min(int(args.stage1_epochs), 1)
        args.stage2_epochs = min(int(args.stage2_epochs), 1)
        args.stage3_epochs = min(int(args.stage3_epochs), 1)
        args.max_batches_per_epoch = 2
        args.sample_count = min(int(args.sample_count), 8)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "fresh":
        run_name = validate_run_name(
            run_name=args.run_name.strip() if args.run_name else resolve_next_run_name(out_root),
            strict_run_naming=bool(args.strict_run_naming),
            force_custom_run_name=bool(args.force_custom_run_name),
        )
        out_dir = out_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if args.resume_dir is None:
            raise ValueError("--mode resume requires --resume-dir")
        out_dir = Path(args.resume_dir)
        run_name = out_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "run_state.json"
    history_path = out_dir / "history.csv"
    cache_dir = out_dir / "cache"
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    state = _load_json(state_path) if args.mode == "resume" else {}
    if not state:
        state = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "run_name": run_name,
            "out_dir": str(out_dir),
            "current_stage": "init",
            "stage1_done": False,
            "stage2_done": False,
            "stage3_done": False,
        }
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    state["config"] = vars(args)
    _save_json(state, state_path)

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else str(args.device)
    device = torch.device(device)
    print(f"[setup] run={run_name} out={out_dir}")
    print(f"[setup] device={device}")

    lab1 = FrozenLab1Encoder(checkpoint_path=Path(args.lab1_checkpoint), device=str(device))
    codec = FrozenEncodec(
        model_id=str(args.codec_model_id),
        bandwidth=float(args.codec_bandwidth),
        chunk_seconds=float(args.codec_chunk_seconds),
        device=str(device),
    )

    if cache_dir.exists() and (cache_dir / "codec_cache_index.csv").exists():
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=cache_dir)
    elif args.reuse_cache_dir is not None:
        index_df, arrays, genre_to_idx, cache_meta = load_codec_cache(cache_dir=Path(args.reuse_cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
    else:
        manifests_df = load_manifests(Path(args.manifests_root), manifest_files=args.manifest_files)
        assigned_df = assign_genres(manifests_df)
        samples_df = materialize_genre_samples(
            assigned_df=assigned_df,
            per_genre_samples=int(args.per_genre_samples),
            seed=int(args.seed),
            drop_unassigned=True,
            require_existing_paths=True,
        )
        index_df, arrays, genre_to_idx, cache_meta = build_codec_cache(
            samples_df=samples_df,
            codec=codec,
            lab1_encoder=lab1,
            cache_dir=cache_dir,
            lab1_n_frames=int(args.lab1_n_frames),
            chunks_per_track=int(args.chunks_per_track),
            chunk_sampling=str(args.chunk_sampling),
            min_start_sec=float(args.min_start_sec),
            max_start_sec=args.max_start_sec,
            seed=int(args.seed),
            progress_every=50,
        )
        save_codec_cache(cache_dir=cache_dir, index_df=index_df, arrays=arrays, genre_to_idx=genre_to_idx, meta=cache_meta)
    print(f"[cache] rows={len(index_df)} genres={len(genre_to_idx)}")

    genre_idx = arrays["genre_idx"]
    if bool(args.split_by_track) and "track_id" in index_df.columns:
        train_idx, val_idx = stratified_group_split_indices(
            genre_idx=genre_idx,
            group_ids=index_df["track_id"].to_numpy(),
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    else:
        train_idx, val_idx = stratified_split_indices(
            genre_idx=genre_idx,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed),
        )
    print(f"[split] train={len(train_idx)} val={len(val_idx)}")

    train_ds = CachedCodecDataset(arrays=arrays, indices=train_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(device.type == "cuda"),
        drop_last=True,
    )
    if len(train_loader) == 0:
        raise RuntimeError("Empty training loader. Increase samples or reduce batch-size.")

    n_genres = len(genre_to_idx)
    z_style_train = arrays["z_style"][train_idx]
    genre_train = arrays["genre_idx"][train_idx]
    q_train = arrays["q_emb"][train_idx]

    style_centroid_bank = build_style_centroid_bank(z_style_train, genre_train, n_genres=n_genres).to(device)
    style_exemplar_bank = build_style_exemplar_bank(z_style_train, genre_train, n_genres=n_genres)
    q_exemplar_bank = build_q_exemplar_bank(q_train, genre_train, n_genres=n_genres)

    gen = CodecLatentTranslator(
        in_channels=int(cache_meta.codec_channels),
        z_content_dim=int(arrays["z_content"].shape[1]),
        z_style_dim=int(arrays["z_style"].shape[1]),
        hidden_channels=int(args.translator_hidden_channels),
        n_blocks=int(args.translator_blocks),
        noise_dim=int(args.translator_noise_dim),
        residual_scale=float(args.translator_residual_scale),
    ).to(device)
    disc = MultiScaleWaveDiscriminator(n_scales=int(args.discriminator_scales)).to(device)

    stage1_weights = CodecTrainWeights(
        adv=float(args.stage1_adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.stage1_latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.stage1_content_weight),
        style=float(args.stage1_style_weight),
        mrstft=float(args.stage1_mrstft_weight),
        mode_seeking=0.0,
    )
    stage2_weights = CodecTrainWeights(
        adv=float(args.adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.content_weight),
        style=float(args.style_weight),
        mrstft=float(args.mrstft_weight),
        mode_seeking=0.0,
    )
    stage3_weights = CodecTrainWeights(
        adv=float(args.adv_weight),
        feature_match=float(args.feature_match_weight),
        latent_l1=float(args.latent_l1_weight),
        latent_continuity=float(args.continuity_weight),
        content=float(args.content_weight),
        style=float(args.style_weight),
        mrstft=float(args.mrstft_weight),
        mode_seeking=float(args.mode_seeking_weight),
    )

    if not args.skip_stage1 and not bool(state.get("stage1_done", False)):
        state["current_stage"] = "stage1"
        _save_json(state, state_path)
        cfg1 = CodecStageTrainConfig(
            stage_name="stage1",
            epochs=int(args.stage1_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode="centroid",
            cond_alpha_start=1.0,
            cond_alpha_end=1.0,
            target_balance=True,
            style_dropout_p=0.0,
            style_jitter_std=0.0,
            exemplar_noise_std=0.0,
            d_step_period=int(args.stage1_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage1_weights,
        )
        hist = train_codec_stage(
            stage_cfg=cfg1,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume"),
        )
        _append_history(history_path, hist)
        state["stage1_done"] = True
        _save_json(state, state_path)

    if not args.skip_stage2 and not bool(state.get("stage2_done", False)):
        stage2_ckpt = checkpoints_dir / "stage2_latest.pt"
        if args.mode == "resume" and not stage2_ckpt.exists():
            _load_models_from_ckpt(
                ckpt_path=checkpoints_dir / "stage1_latest.pt",
                generator=gen,
                discriminator=disc,
                device=device,
            )
        state["current_stage"] = "stage2"
        _save_json(state, state_path)
        cfg2 = CodecStageTrainConfig(
            stage_name="stage2",
            epochs=int(args.stage2_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d) * float(args.stage2_d_lr_mult),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode=str(args.stage2_cond_mode),
            cond_alpha_start=float(args.stage2_cond_alpha_start),
            cond_alpha_end=float(args.stage2_cond_alpha_end),
            target_balance=bool(args.stage2_target_balance),
            style_dropout_p=float(args.stage2_style_dropout_p),
            style_jitter_std=float(args.stage2_style_jitter_std),
            exemplar_noise_std=float(args.stage2_exemplar_noise_std),
            d_step_period=int(args.stage2_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage2_weights,
        )
        hist = train_codec_stage(
            stage_cfg=cfg2,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume" and stage2_ckpt.exists()),
        )
        _append_history(history_path, hist)
        state["stage2_done"] = True
        _save_json(state, state_path)

    if not args.skip_stage3 and not bool(state.get("stage3_done", False)):
        stage3_ckpt = checkpoints_dir / "stage3_latest.pt"
        if args.mode == "resume" and not stage3_ckpt.exists():
            loaded = _load_models_from_ckpt(
                ckpt_path=checkpoints_dir / "stage2_latest.pt",
                generator=gen,
                discriminator=disc,
                device=device,
            )
            if not loaded:
                _load_models_from_ckpt(
                    ckpt_path=checkpoints_dir / "stage1_latest.pt",
                    generator=gen,
                    discriminator=disc,
                    device=device,
                )
        state["current_stage"] = "stage3"
        _save_json(state, state_path)
        cfg3 = CodecStageTrainConfig(
            stage_name="stage3",
            epochs=int(args.stage3_epochs),
            lr_g=float(args.lr_g),
            lr_d=float(args.lr_d) * float(args.stage3_d_lr_mult),
            max_batches_per_epoch=args.max_batches_per_epoch,
            cond_mode=str(args.stage3_cond_mode),
            cond_alpha_start=float(args.stage3_cond_alpha_start),
            cond_alpha_end=float(args.stage3_cond_alpha_end),
            target_balance=bool(args.stage3_target_balance),
            style_dropout_p=float(args.stage3_style_dropout_p),
            style_jitter_std=float(args.stage3_style_jitter_std),
            exemplar_noise_std=float(args.stage3_exemplar_noise_std),
            d_step_period=int(args.stage3_d_step_period),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            g_grad_clip_norm=float(args.g_grad_clip_norm),
            d_grad_clip_norm=float(args.d_grad_clip_norm),
            weights=stage3_weights,
            mode_seeking_noise_scale=1.0,
        )
        hist = train_codec_stage(
            stage_cfg=cfg3,
            generator=gen,
            discriminator=disc,
            codec=codec,
            lab1_encoder=lab1,
            train_loader=train_loader,
            n_genres=n_genres,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            q_exemplar_bank=q_exemplar_bank,
            out_ckpt_dir=checkpoints_dir,
            device=device,
            resume=(args.mode == "resume" and stage3_ckpt.exists()),
        )
        _append_history(history_path, hist)
        state["stage3_done"] = True
        _save_json(state, state_path)

    if bool(args.auto_sample_export):
        sample_dir = _export_codec_samples(
            out_dir=out_dir / "samples" / str(args.sample_export_tag),
            generator=gen,
            codec=codec,
            arrays=arrays,
            val_idx=val_idx,
            style_centroid_bank=style_centroid_bank,
            style_exemplar_bank=style_exemplar_bank,
            seed=int(args.seed),
            n_samples=int(args.sample_count),
            cond_mode=str(args.stage3_cond_mode),
            cond_alpha=float(args.stage3_cond_alpha_end),
        )
        if sample_dir is not None:
            print(f"[sample-export] wrote {sample_dir}")

    state["current_stage"] = "done"
    state["updated_at"] = datetime.utcnow().isoformat() + "Z"
    _save_json(state, state_path)
    print(f"[done] run={run_name}")


if __name__ == "__main__":
    main()
