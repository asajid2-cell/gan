from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.lab3_bridge import FrozenLab1Encoder, denormalize_log_mel
from src.lab3_data import (
    DEFAULT_MANIFESTS,
    CachedSynthesisDataset,
    assign_genres,
    build_latent_cache,
    genre_count_table,
    load_cache,
    load_manifests,
    materialize_genre_samples,
    save_cache,
    stratified_group_split_indices,
    stratified_split_indices,
)
from src.lab3_eval import evaluate_classifier_quality, evaluate_genre_shift, fit_third_party_style_classifier
from src.lab3_models import (
    HybridMelDiscriminator,
    MelDiscriminator,
    MultiPeriodMelDiscriminator,
    MultiScaleMelDiscriminator,
    ReconstructionDecoder,
    SubBandMelDiscriminator,
)
from src.lab3_sampling import (
    export_posttrain_samples,
    resolve_next_run_name,
    validate_run_name,
)
from src.lab3_train import (
    StageTrainConfig,
    TrainWeights,
    build_condition_bank,
    build_style_exemplar_bank,
    build_style_centroid_bank,
    load_target_centroids,
    train_stage1,
    train_stage2,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_out_root() -> Path:
    return _repo_root() / "saves2" / "lab3_synthesis"


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


def _default_lab2_centroids_json() -> Path:
    root = _repo_root() / "saves" / "lab2_calibration"
    if not root.exists():
        return _repo_root() / "saves" / "lab2_calibration" / "target_centroids.json"
    dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        p = d / "target_centroids.json"
        if p.exists():
            return p
    return root / "target_centroids.json"


def _save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _parse_mrstft_resolutions(spec: str) -> tuple[tuple[int, int, int], ...]:
    """
    Parse "64,16,64;128,32,128;256,64,256" -> ((64,16,64), (128,32,128), (256,64,256))
    """
    chunks = [c.strip() for c in str(spec).split(";") if c.strip()]
    out: list[tuple[int, int, int]] = []
    for c in chunks:
        vals = [v.strip() for v in c.split(",") if v.strip()]
        if len(vals) != 3:
            raise ValueError(f"Invalid MRSTFT resolution chunk '{c}'. Expected n_fft,hop,win.")
        n_fft, hop, win = (int(vals[0]), int(vals[1]), int(vals[2]))
        out.append((n_fft, hop, win))
    if not out:
        raise ValueError("MRSTFT resolutions must not be empty.")
    return tuple(out)


def _load_json(path: Path, default: Optional[Dict] = None) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {} if default is None else default


def _append_history(csv_path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if csv_path.exists():
        prev = pd.read_csv(csv_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _checkpoint_epoch(ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        return 0
    payload = torch.load(str(ckpt_path), map_location="cpu")
    return int(payload.get("epoch", 0))


def _build_genre_to_source_idx(
    index_df: pd.DataFrame,
    genre_to_idx: Dict[str, int],
    source_to_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Map each lab3 genre to a lab1 source-class index for style CE supervision.
    Uses dominant source per genre; falls back to heuristics when needed.
    """
    mapping = np.full((len(genre_to_idx),), fill_value=-1, dtype=np.int64)

    source_alias = {
        "xtc_audio_clean": "xtc_hiphop",
        "hh_lfbb_audio_clean": "hh_lfbb",
        "cc0_audio_clean": "cc0_music",
        "phase1_pdmx": "phase1_pdmx",
    }

    def _resolve_source_idx(candidates: List[str]) -> int:
        for name in candidates:
            if name in source_to_idx:
                return int(source_to_idx[name])
        keys = list(source_to_idx.keys())
        lower_keys = [k.lower() for k in keys]
        for name in candidates:
            n = name.lower()
            for i, k in enumerate(lower_keys):
                if n in k or k in n:
                    return int(source_to_idx[keys[i]])
        return -1

    heuristics = {
        "hiphop_xtc": ["xtc_hiphop", "xtc_audio_clean", "xtc"],
        "lofi_hh_lfbb": ["hh_lfbb", "hh_lfbb_audio_clean", "lfbb"],
        "cc0_other": ["cc0_music", "cc0_audio_clean", "cc0"],
        "baroque_classical": ["phase1_pdmx", "pdmx"],
    }

    for genre, gidx in genre_to_idx.items():
        chosen_idx = -1
        if "genre" in index_df.columns and "source" in index_df.columns:
            sub = index_df[index_df["genre"] == genre]
            if len(sub) > 0:
                vc = sub["source"].value_counts()
                for src_name in vc.index.tolist():
                    src_norm = source_alias.get(src_name, src_name)
                    idx = _resolve_source_idx([src_norm, src_name])
                    if idx >= 0:
                        chosen_idx = idx
                        break
        if chosen_idx < 0 and genre in heuristics:
            chosen_idx = _resolve_source_idx(heuristics[genre])
        mapping[int(gidx)] = int(chosen_idx)
    return torch.from_numpy(mapping)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab 3 - Reconstruction Decoder synthesis pipeline")
    p.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    p.add_argument("--out-root", type=Path, default=_default_out_root())
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--strict-run-naming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-custom-run-name", action="store_true")
    p.add_argument("--resume-dir", type=Path, default=None)
    p.add_argument("--reuse-cache-dir", type=Path, default=None)

    p.add_argument("--manifests-root", type=Path, default=Path("Z:/DataSets/_lab1_manifests"))
    p.add_argument("--manifest-files", nargs="*", default=DEFAULT_MANIFESTS)
    p.add_argument("--per-genre-samples", type=int, default=800)
    p.add_argument("--chunks-per-track", type=int, default=4)
    p.add_argument("--chunk-sampling", choices=["uniform", "random"], default="uniform")
    p.add_argument("--min-start-sec", type=float, default=0.0)
    p.add_argument("--max-start-sec", type=float, default=None)
    p.add_argument("--split-by-track", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=328)
    p.add_argument("--val-ratio", type=float, default=0.15)

    p.add_argument("--lab1-checkpoint", type=Path, default=_default_lab1_checkpoint())
    p.add_argument("--lab2-centroids-json", type=Path, default=_default_lab2_centroids_json())
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--n-frames", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--generator-norm", choices=["instance", "batch"], default="instance")
    p.add_argument("--generator-spectral-norm", action="store_true")
    p.add_argument("--generator-upsample", choices=["transpose", "pixelshuffle", "nearest"], default="transpose")
    p.add_argument("--generator-mrf", action="store_true")
    p.add_argument("--generator-mrf-kernels", type=str, default="3,7,11")
    p.add_argument(
        "--discriminator-arch",
        choices=["single", "multiscale", "subband", "multiperiod", "hybrid"],
        default="single",
    )
    p.add_argument("--discriminator-scales", type=int, default=3)
    p.add_argument("--discriminator-subband-low-bins", type=int, default=32)
    p.add_argument("--discriminator-subband-mid-bins", type=int, default=32)
    p.add_argument("--discriminator-periods", type=str, default="1,2,3,5")
    p.add_argument("--discriminator-spectral-norm", action="store_true")

    p.add_argument("--stage1-epochs", type=int, default=20)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)

    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)
    p.add_argument("--gan-loss", choices=["bce", "hinge"], default="bce")
    p.add_argument("--adv-weight", type=float, default=0.5)
    p.add_argument("--r1-gamma", type=float, default=0.0)
    p.add_argument("--r1-interval", type=int, default=16)
    p.add_argument("--recon-weight", type=float, default=8.0)
    p.add_argument("--content-weight", type=float, default=2.0)
    p.add_argument("--style-weight", type=float, default=1.0)
    p.add_argument("--continuity-weight", type=float, default=1.0)
    p.add_argument("--mrstft-weight", type=float, default=0.0)
    p.add_argument("--stage1-mrstft-weight", type=float, default=None)
    p.add_argument("--stage2-mrstft-weight", type=float, default=None)
    p.add_argument("--mrstft-resolutions", type=str, default="64,16,64;128,32,128;256,64,256")
    p.add_argument("--flatness-weight", type=float, default=0.5)
    p.add_argument("--feature-match-weight", type=float, default=0.0)
    p.add_argument("--perceptual-weight", type=float, default=0.0)
    p.add_argument("--style-hinge-weight", type=float, default=0.0)
    p.add_argument("--contrastive-weight", type=float, default=0.0)
    p.add_argument("--batch-infonce-div-weight", type=float, default=0.0)
    p.add_argument("--pfm-style-weight", type=float, default=0.0)
    p.add_argument("--diversity-weight", type=float, default=0.0)
    p.add_argument("--timbre-balance-weight", type=float, default=0.0)
    p.add_argument("--lowmid-recon-weight", type=float, default=0.0)
    p.add_argument("--spectral-tilt-weight", type=float, default=0.0)
    p.add_argument("--zcr-proxy-weight", type=float, default=0.0)
    p.add_argument("--style-mid-weight", type=float, default=0.0)
    p.add_argument("--hf-muzzle-weight", type=float, default=0.0)
    p.add_argument("--highpass-anchor-weight", type=float, default=0.0)
    p.add_argument("--mel-diversity-weight", type=float, default=0.0)
    p.add_argument("--target-profile-weight", type=float, default=0.0)
    p.add_argument("--stage2-d-lr-mult", type=float, default=1.0)
    p.add_argument("--stage2-content-start", type=float, default=None)
    p.add_argument("--stage2-content-end", type=float, default=None)
    p.add_argument("--stage2-style-label-smoothing", type=float, default=0.0)
    p.add_argument("--stage2-style-only-warmup-epochs", type=int, default=0)
    p.add_argument("--stage2-g-lr-warmup-epochs", type=int, default=0)
    p.add_argument("--stage2-g-lr-start-mult", type=float, default=1.0)
    p.add_argument("--stage2-cond-noise-std", type=float, default=0.0)
    p.add_argument("--stage2-style-jitter-std", type=float, default=0.0)
    p.add_argument("--stage2-style-hinge-target-conf", type=float, default=0.85)
    p.add_argument("--stage2-cond-mode", choices=["centroid", "exemplar", "mix"], default="mix")
    p.add_argument("--stage2-cond-alpha-start", type=float, default=0.8)
    p.add_argument("--stage2-cond-alpha-end", type=float, default=0.4)
    p.add_argument("--stage2-cond-exemplar-noise-std", type=float, default=0.03)
    p.add_argument("--stage2-target-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stage2-adaptive-content", action="store_true")
    p.add_argument("--stage2-adaptive-content-low", type=float, default=0.05)
    p.add_argument("--stage2-adaptive-content-high", type=float, default=0.5)
    p.add_argument("--stage2-adaptive-conf-low", type=float, default=0.5)
    p.add_argument("--stage2-adaptive-conf-high", type=float, default=0.8)
    p.add_argument("--reset-stage2-out-layer", action="store_true")
    p.add_argument("--stage2-style-critic-lr", type=float, default=2e-4)
    p.add_argument("--stage2-contrastive-temp", type=float, default=0.10)
    p.add_argument("--stage2-batch-infonce-temp", type=float, default=0.15)
    p.add_argument("--stage2-diversity-margin", type=float, default=0.90)
    p.add_argument("--stage2-diversity-max-pairs", type=int, default=64)
    p.add_argument("--stage2-mel-diversity-margin", type=float, default=0.85)
    p.add_argument("--stage2-mel-diversity-max-pairs", type=int, default=64)
    p.add_argument("--stage2-style-lowpass-keep-bins", type=int, default=80)
    p.add_argument("--stage2-style-lowpass-cutoff-hz", type=float, default=None)
    p.add_argument("--stage2-style-mid-low-bin", type=int, default=8)
    p.add_argument("--stage2-style-mid-high-bin", type=int, default=56)
    p.add_argument("--stage2-lowmid-split-bin", type=int, default=80)
    p.add_argument("--stage2-lowmid-gain", type=float, default=5.0)
    p.add_argument("--stage2-high-gain", type=float, default=0.5)
    p.add_argument("--stage2-spectral-tilt-max-ratio", type=float, default=0.7)
    p.add_argument("--stage2-zcr-proxy-target-max", type=float, default=0.18)
    p.add_argument("--stage2-style-thaw-last-epochs", type=int, default=0)
    p.add_argument("--stage2-style-thaw-lr", type=float, default=1e-6)
    p.add_argument(
        "--stage2-style-thaw-scope",
        choices=["style_head", "shared_style", "full_style_path"],
        default="style_head",
    )
    p.add_argument("--stage2-disc-use-content-cond", action="store_true")
    p.add_argument("--d-real-label", type=float, default=1.0)
    p.add_argument("--d-fake-label", type=float, default=0.0)
    p.add_argument("--g-real-label", type=float, default=1.0)

    p.add_argument("--mps-threshold", type=float, default=0.90)
    p.add_argument("--sf-threshold", type=float, default=0.85)
    p.add_argument("--eval-max-batches", type=int, default=30)

    p.add_argument("--skip-stage1", action="store_true")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--auto-sample-export", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sample-count", type=int, default=100)
    p.add_argument(
        "--sample-target-mode",
        choices=["balanced_random", "random", "round_robin"],
        default="balanced_random",
    )
    p.add_argument("--sample-griffin-lim-iters", type=int, default=48)
    p.add_argument("--sample-export-tag", type=str, default="posttrain_samples")
    p.add_argument("--sample-write-real-audio", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)

    if args.smoke:
        args.per_genre_samples = min(args.per_genre_samples, 24)
        args.stage1_epochs = min(args.stage1_epochs, 1)
        args.stage2_epochs = min(args.stage2_epochs, 1)
        if args.max_batches_per_epoch is None:
            args.max_batches_per_epoch = 4
        args.eval_max_batches = min(args.eval_max_batches, 4)
        args.sample_count = min(int(args.sample_count), 8)

    if args.mode == "fresh":
        requested_name = str(args.run_name).strip()
        if requested_name:
            run_name = validate_run_name(
                requested_name,
                strict_run_naming=bool(args.strict_run_naming),
                force_custom_run_name=bool(args.force_custom_run_name),
            )
        else:
            run_name = resolve_next_run_name(out_root)
        if bool(args.strict_run_naming) and not bool(args.force_custom_run_name):
            if not re.match(r"^run\d+$", run_name):
                raise ValueError(
                    f"Strict run naming is enabled; run name must match runN. Got: {run_name}"
                )
        out_dir = out_root / run_name
        if out_dir.exists():
            raise FileExistsError(f"Run directory already exists: {out_dir}")
    else:
        if args.resume_dir is None:
            raise ValueError("--mode resume requires --resume-dir")
        out_dir = Path(args.resume_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    ckpt_dir = out_dir / "checkpoints"
    state_path = out_dir / "run_state.json"
    history_path = out_dir / "history.csv"

    state = _load_json(
        state_path,
        default={
            "stage_cache_done": False,
            "stage1_done": False,
            "stage2_done": False,
            "eval_done": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    state["config"] = {
        "mode": args.mode,
        "out_root": str(out_root),
        "out_dir": str(out_dir),
        "strict_run_naming": bool(args.strict_run_naming),
        "force_custom_run_name": bool(args.force_custom_run_name),
        "reuse_cache_dir": str(args.reuse_cache_dir) if args.reuse_cache_dir else "",
        "manifests_root": str(args.manifests_root),
        "manifest_files": list(args.manifest_files),
        "per_genre_samples": int(args.per_genre_samples),
        "chunks_per_track": int(args.chunks_per_track),
        "chunk_sampling": str(args.chunk_sampling),
        "min_start_sec": float(args.min_start_sec),
        "max_start_sec": None if args.max_start_sec is None else float(args.max_start_sec),
        "split_by_track": bool(args.split_by_track),
        "seed": int(args.seed),
        "val_ratio": float(args.val_ratio),
        "lab1_checkpoint": str(args.lab1_checkpoint),
        "lab2_centroids_json": str(args.lab2_centroids_json),
        "device": device,
        "n_frames": int(args.n_frames),
        "batch_size": int(args.batch_size),
        "generator_norm": str(args.generator_norm),
        "generator_spectral_norm": bool(args.generator_spectral_norm),
        "generator_mrf": bool(args.generator_mrf),
        "generator_mrf_kernels": str(args.generator_mrf_kernels),
        "generator_upsample": str(args.generator_upsample),
        "discriminator_arch": str(args.discriminator_arch),
        "discriminator_scales": int(args.discriminator_scales),
        "discriminator_subband_low_bins": int(args.discriminator_subband_low_bins),
        "discriminator_subband_mid_bins": int(args.discriminator_subband_mid_bins),
        "discriminator_periods": str(args.discriminator_periods),
        "discriminator_spectral_norm": bool(args.discriminator_spectral_norm),
        "stage1_epochs": int(args.stage1_epochs),
        "stage2_epochs": int(args.stage2_epochs),
        "max_batches_per_epoch": None if args.max_batches_per_epoch is None else int(args.max_batches_per_epoch),
        "lr_g": float(args.lr_g),
        "lr_d": float(args.lr_d),
        "gan_loss": str(args.gan_loss),
        "adv_weight": float(args.adv_weight),
        "r1_gamma": float(args.r1_gamma),
        "r1_interval": int(args.r1_interval),
        "recon_weight": float(args.recon_weight),
        "content_weight": float(args.content_weight),
        "style_weight": float(args.style_weight),
        "continuity_weight": float(args.continuity_weight),
        "mrstft_weight": float(args.mrstft_weight),
        "stage1_mrstft_weight": None if args.stage1_mrstft_weight is None else float(args.stage1_mrstft_weight),
        "stage2_mrstft_weight": None if args.stage2_mrstft_weight is None else float(args.stage2_mrstft_weight),
        "mrstft_resolutions": str(args.mrstft_resolutions),
        "flatness_weight": float(args.flatness_weight),
        "feature_match_weight": float(args.feature_match_weight),
        "perceptual_weight": float(args.perceptual_weight),
        "style_hinge_weight": float(args.style_hinge_weight),
        "contrastive_weight": float(args.contrastive_weight),
        "batch_infonce_div_weight": float(args.batch_infonce_div_weight),
        "pfm_style_weight": float(args.pfm_style_weight),
        "diversity_weight": float(args.diversity_weight),
        "timbre_balance_weight": float(args.timbre_balance_weight),
        "lowmid_recon_weight": float(args.lowmid_recon_weight),
        "spectral_tilt_weight": float(args.spectral_tilt_weight),
        "zcr_proxy_weight": float(args.zcr_proxy_weight),
        "style_mid_weight": float(args.style_mid_weight),
        "hf_muzzle_weight": float(args.hf_muzzle_weight),
        "highpass_anchor_weight": float(args.highpass_anchor_weight),
        "mel_diversity_weight": float(args.mel_diversity_weight),
        "target_profile_weight": float(args.target_profile_weight),
        "stage2_d_lr_mult": float(args.stage2_d_lr_mult),
        "stage2_content_start": None if args.stage2_content_start is None else float(args.stage2_content_start),
        "stage2_content_end": None if args.stage2_content_end is None else float(args.stage2_content_end),
        "stage2_style_label_smoothing": float(args.stage2_style_label_smoothing),
        "stage2_style_only_warmup_epochs": int(args.stage2_style_only_warmup_epochs),
        "stage2_g_lr_warmup_epochs": int(args.stage2_g_lr_warmup_epochs),
        "stage2_g_lr_start_mult": float(args.stage2_g_lr_start_mult),
        "stage2_cond_noise_std": float(args.stage2_cond_noise_std),
        "stage2_style_jitter_std": float(args.stage2_style_jitter_std),
        "stage2_style_hinge_target_conf": float(args.stage2_style_hinge_target_conf),
        "stage2_cond_mode": str(args.stage2_cond_mode),
        "stage2_cond_alpha_start": float(args.stage2_cond_alpha_start),
        "stage2_cond_alpha_end": float(args.stage2_cond_alpha_end),
        "stage2_cond_exemplar_noise_std": float(args.stage2_cond_exemplar_noise_std),
        "stage2_target_balance": bool(args.stage2_target_balance),
        "stage2_adaptive_content": bool(args.stage2_adaptive_content),
        "stage2_adaptive_content_low": float(args.stage2_adaptive_content_low),
        "stage2_adaptive_content_high": float(args.stage2_adaptive_content_high),
        "stage2_adaptive_conf_low": float(args.stage2_adaptive_conf_low),
        "stage2_adaptive_conf_high": float(args.stage2_adaptive_conf_high),
        "reset_stage2_out_layer": bool(args.reset_stage2_out_layer),
        "stage2_style_critic_lr": float(args.stage2_style_critic_lr),
        "stage2_contrastive_temp": float(args.stage2_contrastive_temp),
        "stage2_batch_infonce_temp": float(args.stage2_batch_infonce_temp),
        "stage2_diversity_margin": float(args.stage2_diversity_margin),
        "stage2_diversity_max_pairs": int(args.stage2_diversity_max_pairs),
        "stage2_mel_diversity_margin": float(args.stage2_mel_diversity_margin),
        "stage2_mel_diversity_max_pairs": int(args.stage2_mel_diversity_max_pairs),
        "stage2_style_lowpass_keep_bins": int(args.stage2_style_lowpass_keep_bins),
        "stage2_style_lowpass_cutoff_hz": None
        if args.stage2_style_lowpass_cutoff_hz is None
        else float(args.stage2_style_lowpass_cutoff_hz),
        "stage2_style_mid_low_bin": int(args.stage2_style_mid_low_bin),
        "stage2_style_mid_high_bin": int(args.stage2_style_mid_high_bin),
        "stage2_lowmid_split_bin": int(args.stage2_lowmid_split_bin),
        "stage2_lowmid_gain": float(args.stage2_lowmid_gain),
        "stage2_high_gain": float(args.stage2_high_gain),
        "stage2_spectral_tilt_max_ratio": float(args.stage2_spectral_tilt_max_ratio),
        "stage2_zcr_proxy_target_max": float(args.stage2_zcr_proxy_target_max),
        "stage2_style_thaw_last_epochs": int(args.stage2_style_thaw_last_epochs),
        "stage2_style_thaw_lr": float(args.stage2_style_thaw_lr),
        "stage2_style_thaw_scope": str(args.stage2_style_thaw_scope),
        "stage2_disc_use_content_cond": bool(args.stage2_disc_use_content_cond),
        "d_real_label": float(args.d_real_label),
        "d_fake_label": float(args.d_fake_label),
        "g_real_label": float(args.g_real_label),
        "auto_sample_export": bool(args.auto_sample_export),
        "sample_count": int(args.sample_count),
        "sample_target_mode": str(args.sample_target_mode),
        "sample_griffin_lim_iters": int(args.sample_griffin_lim_iters),
        "sample_export_tag": str(args.sample_export_tag),
        "sample_write_real_audio": bool(args.sample_write_real_audio),
        "smoke": bool(args.smoke),
    }
    _save_json(state, state_path)

    # ---------------------------
    # Stage A - Build or Load Cache
    # ---------------------------
    if state.get("stage_cache_done", False):
        print("[lab3] loading existing cache from run dir...")
        index_df, arrays, genre_to_idx = load_cache(cache_dir)
    else:
        if args.reuse_cache_dir is not None:
            print(f"[lab3] reusing cache from {args.reuse_cache_dir}")
            index_df, arrays, genre_to_idx = load_cache(Path(args.reuse_cache_dir))
            save_cache(cache_dir, index_df, arrays, genre_to_idx)
        else:
            print("[lab3] materializing samples from manifests...")
            raw_df = load_manifests(args.manifests_root, args.manifest_files)
            assigned_df = assign_genres(raw_df)
            assigned_counts = genre_count_table(assigned_df)
            samples_df = materialize_genre_samples(
                assigned_df,
                per_genre_samples=args.per_genre_samples,
                seed=args.seed,
            )
            samples_df.to_csv(out_dir / "genre_samples.csv", index=False)
            sampled_counts = genre_count_table(samples_df)
            state["assigned_genre_counts"] = assigned_counts
            state["sampled_genre_counts"] = sampled_counts
            _save_json(state, state_path)
            print(f"[lab3] assigned genres: {assigned_counts}")
            print(f"[lab3] sampled genres: {sampled_counts}")

            print("[lab3] extracting latent cache with frozen Lab1 encoder...")
            enc = FrozenLab1Encoder(args.lab1_checkpoint, device=device)
            index_df, arrays, genre_to_idx = build_latent_cache(
                samples_df=samples_df,
                encoder=enc,
                cache_dir=cache_dir,
                n_frames=args.n_frames,
                chunks_per_track=args.chunks_per_track,
                chunk_sampling=args.chunk_sampling,
                min_start_sec=args.min_start_sec,
                max_start_sec=args.max_start_sec,
                seed=args.seed,
                progress_every=100,
            )
            save_cache(cache_dir, index_df, arrays, genre_to_idx)

        state["stage_cache_done"] = True
        state["n_samples"] = int(len(index_df))
        state["genre_to_idx"] = {k: int(v) for k, v in genre_to_idx.items()}
        _save_json(state, state_path)

    # ---------------------------
    # Dataset split/loaders
    # ---------------------------
    if bool(args.split_by_track) and "track_id" in index_df.columns:
        train_idx, val_idx = stratified_group_split_indices(
            arrays["genre_idx"],
            index_df["track_id"].astype(str).to_numpy(),
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        train_idx, val_idx = stratified_split_indices(arrays["genre_idx"], val_ratio=args.val_ratio, seed=args.seed)
    ds_train = CachedSynthesisDataset(arrays=arrays, indices=train_idx)
    ds_val = CachedSynthesisDataset(arrays=arrays, indices=val_idx)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---------------------------
    # Conditioning banks
    # ---------------------------
    frozen_encoder = FrozenLab1Encoder(args.lab1_checkpoint, device=device)
    target_centroids = load_target_centroids(args.lab2_centroids_json)
    cond_bank = build_condition_bank(genre_to_idx, target_centroids)
    style_proto_bank = build_style_centroid_bank(
        z_style=arrays["z_style"],
        genre_idx=arrays["genre_idx"],
        n_genres=len(genre_to_idx),
    )
    style_exemplar_bank = build_style_exemplar_bank(
        z_style=arrays["z_style"][train_idx],
        genre_idx=arrays["genre_idx"][train_idx],
        n_genres=len(genre_to_idx),
    )
    genre_to_source_idx = _build_genre_to_source_idx(index_df, genre_to_idx, frozen_encoder.source_to_idx)
    state["genre_to_lab1_source_idx"] = {
        str(g): int(genre_to_source_idx[int(i)].item()) for g, i in genre_to_idx.items()
    }
    _save_json(state, state_path)

    # ---------------------------
    # Models
    # ---------------------------
    generator = ReconstructionDecoder(
        zc_dim=arrays["z_content"].shape[1],
        cond_dim=cond_bank.shape[1],
        n_mels=arrays["mel_norm"].shape[1],
        n_frames=arrays["mel_norm"].shape[2],
        norm=str(args.generator_norm),
        upsample=str(args.generator_upsample),
        spectral_norm=bool(args.generator_spectral_norm),
        mrf=bool(args.generator_mrf),
        mrf_kernels=tuple(int(x.strip()) for x in str(args.generator_mrf_kernels).split(",") if x.strip()),
    ).to(device)
    disc_cond_dim = int(cond_bank.shape[1]) + (
        int(arrays["z_content"].shape[1]) if bool(args.stage2_disc_use_content_cond) else 0
    )
    if str(args.discriminator_arch) == "multiscale":
        discriminator = MultiScaleMelDiscriminator(
            cond_dim=disc_cond_dim,
            num_scales=int(args.discriminator_scales),
            spectral_norm=bool(args.discriminator_spectral_norm),
        ).to(device)
    elif str(args.discriminator_arch) == "subband":
        discriminator = SubBandMelDiscriminator(
            cond_dim=disc_cond_dim,
            low_bins=int(args.discriminator_subband_low_bins),
            mid_bins=int(args.discriminator_subband_mid_bins),
            spectral_norm=bool(args.discriminator_spectral_norm),
        ).to(device)
    elif str(args.discriminator_arch) == "multiperiod":
        periods = tuple(int(x.strip()) for x in str(args.discriminator_periods).split(",") if x.strip())
        discriminator = MultiPeriodMelDiscriminator(
            cond_dim=disc_cond_dim,
            periods=periods if periods else (1, 2, 3, 5),
            spectral_norm=bool(args.discriminator_spectral_norm),
        ).to(device)
    elif str(args.discriminator_arch) == "hybrid":
        periods = tuple(int(x.strip()) for x in str(args.discriminator_periods).split(",") if x.strip())
        discriminator = HybridMelDiscriminator(
            cond_dim=disc_cond_dim,
            num_scales=int(args.discriminator_scales),
            periods=periods if periods else (1, 2, 3, 5),
            spectral_norm=bool(args.discriminator_spectral_norm),
        ).to(device)
    else:
        discriminator = MelDiscriminator(
            cond_dim=disc_cond_dim,
            spectral_norm=bool(args.discriminator_spectral_norm),
        ).to(device)

    weights = TrainWeights(
        adv=args.adv_weight,
        recon_l1=args.recon_weight,
        content=args.content_weight,
        style=args.style_weight,
        continuity=args.continuity_weight,
        mrstft=args.mrstft_weight,
        flatness=args.flatness_weight,
        feature_match=args.feature_match_weight,
        perceptual=args.perceptual_weight,
        style_hinge=args.style_hinge_weight,
        contrastive=args.contrastive_weight,
        batch_infonce_diversity=args.batch_infonce_div_weight,
        pfm_style=args.pfm_style_weight,
        diversity=args.diversity_weight,
        timbre_balance=args.timbre_balance_weight,
        lowmid_recon=args.lowmid_recon_weight,
        spectral_tilt=args.spectral_tilt_weight,
        zcr_proxy=args.zcr_proxy_weight,
        hf_muzzle=args.hf_muzzle_weight,
        highpass_anchor=args.highpass_anchor_weight,
        style_mid=args.style_mid_weight,
        mel_diversity=args.mel_diversity_weight,
        target_profile=args.target_profile_weight,
    )
    stage1_mrstft_weight = (
        float(args.stage1_mrstft_weight) if args.stage1_mrstft_weight is not None else float(args.mrstft_weight)
    )
    stage2_mrstft_weight = (
        float(args.stage2_mrstft_weight) if args.stage2_mrstft_weight is not None else float(args.mrstft_weight)
    )
    weights_stage1 = replace(weights, mrstft=stage1_mrstft_weight)
    weights_stage2 = replace(weights, mrstft=stage2_mrstft_weight)
    mrstft_resolutions = _parse_mrstft_resolutions(args.mrstft_resolutions)

    # Precompute real target-genre spectral profiles in mel-bin space.
    mel_db_all = denormalize_log_mel(torch.from_numpy(arrays["mel_norm"]).float())
    p_all = torch.pow(10.0, mel_db_all / 10.0).clamp_min(1e-10)  # [N, n_mels, T]
    prof_all = p_all.sum(dim=2)  # [N, n_mels]
    prof_all = prof_all / (prof_all.sum(dim=1, keepdim=True) + 1e-8)
    target_profile_bank = torch.zeros((len(genre_to_idx), prof_all.shape[1]), dtype=torch.float32)
    for g in range(len(genre_to_idx)):
        mask = torch.from_numpy((arrays["genre_idx"] == g).astype(np.bool_))
        if bool(mask.any()):
            v = prof_all[mask].mean(dim=0)
            v = v / (v.sum() + 1e-8)
            target_profile_bank[g] = v

    # ---------------------------
    # Stage 1 training
    # ---------------------------
    ckpt_stage1 = ckpt_dir / "stage1_latest.pt"
    stage1_epoch_done = _checkpoint_epoch(ckpt_stage1)
    need_stage1 = (not args.skip_stage1) and (stage1_epoch_done < int(args.stage1_epochs))
    if need_stage1:
        print("[lab3] stage1 training...")
        cfg1 = StageTrainConfig(
            stage_name="stage1",
            epochs=args.stage1_epochs,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
            max_batches_per_epoch=args.max_batches_per_epoch,
            gan_loss=str(args.gan_loss),
            disc_use_content_cond=bool(args.stage2_disc_use_content_cond),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            weights=weights_stage1,
            d_real_label=args.d_real_label,
            d_fake_label=args.d_fake_label,
            g_real_label=args.g_real_label,
            mrstft_resolutions=mrstft_resolutions,
        )
        hist1 = train_stage1(
            generator=generator,
            discriminator=discriminator,
            frozen_encoder=frozen_encoder,
            train_loader=train_loader,
            cond_bank=cond_bank,
            device=device,
            stage_cfg=cfg1,
            checkpoint_path=ckpt_stage1,
            resume=(args.mode == "resume"),
        )
        _append_history(history_path, hist1)
        stage1_epoch_done = _checkpoint_epoch(ckpt_stage1)
        state["stage1_done"] = bool(stage1_epoch_done >= int(args.stage1_epochs))
        state["stage1_epoch_done"] = int(stage1_epoch_done)
        _save_json(state, state_path)
    else:
        if ckpt_stage1.exists():
            payload = torch.load(str(ckpt_stage1), map_location="cpu")
            generator.load_state_dict(payload["generator"], strict=False)
            discriminator.load_state_dict(payload["discriminator"], strict=False)
            print(f"[lab3] stage1 checkpoint loaded (epoch={stage1_epoch_done}).")
            state["stage1_epoch_done"] = int(stage1_epoch_done)
            state["stage1_done"] = bool(stage1_epoch_done >= int(args.stage1_epochs))
            _save_json(state, state_path)

    # ---------------------------
    # Stage 2 training
    # ---------------------------
    ckpt_stage2 = ckpt_dir / "stage2_latest.pt"
    stage2_epoch_done = _checkpoint_epoch(ckpt_stage2)
    need_stage2 = (not args.skip_stage2) and (stage2_epoch_done < int(args.stage2_epochs))
    if need_stage2:
        print("[lab3] stage2 training...")
        if bool(args.reset_stage2_out_layer):
            # Optional "spectral reset": re-init only the final output layer so Stage 2
            # can escape a local minimum while keeping lower-level structure features.
            for m in [generator.out]:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()
            print("[lab3] reset-stage2-out-layer applied.")
        cfg2 = StageTrainConfig(
            stage_name="stage2",
            epochs=args.stage2_epochs,
            lr_g=args.lr_g,
            lr_d=float(args.lr_d) * float(args.stage2_d_lr_mult),
            max_batches_per_epoch=args.max_batches_per_epoch,
            gan_loss=str(args.gan_loss),
            disc_use_content_cond=bool(args.stage2_disc_use_content_cond),
            r1_gamma=float(args.r1_gamma),
            r1_interval=int(args.r1_interval),
            weights=weights_stage2,
            content_weight_start=args.stage2_content_start,
            content_weight_end=args.stage2_content_end,
            style_label_smoothing=args.stage2_style_label_smoothing,
            style_only_warmup_epochs=args.stage2_style_only_warmup_epochs,
            g_lr_warmup_epochs=args.stage2_g_lr_warmup_epochs,
            g_lr_start_mult=args.stage2_g_lr_start_mult,
            cond_noise_std=args.stage2_cond_noise_std,
            style_jitter_std=args.stage2_style_jitter_std,
            cond_mode=args.stage2_cond_mode,
            cond_alpha_start=args.stage2_cond_alpha_start,
            cond_alpha_end=args.stage2_cond_alpha_end,
            cond_exemplar_noise_std=args.stage2_cond_exemplar_noise_std,
            target_balance=bool(args.stage2_target_balance),
            style_hinge_target_conf=args.stage2_style_hinge_target_conf,
            adaptive_content_weight=bool(args.stage2_adaptive_content),
            adaptive_content_low=args.stage2_adaptive_content_low,
            adaptive_content_high=args.stage2_adaptive_content_high,
            adaptive_conf_low=args.stage2_adaptive_conf_low,
            adaptive_conf_high=args.stage2_adaptive_conf_high,
            style_critic_lr=args.stage2_style_critic_lr,
            contrastive_temp=args.stage2_contrastive_temp,
            batch_infonce_temp=args.stage2_batch_infonce_temp,
            diversity_margin=args.stage2_diversity_margin,
            diversity_max_pairs=args.stage2_diversity_max_pairs,
            mel_diversity_margin=args.stage2_mel_diversity_margin,
            mel_diversity_max_pairs=args.stage2_mel_diversity_max_pairs,
            style_lowpass_keep_bins=args.stage2_style_lowpass_keep_bins,
            style_lowpass_cutoff_hz=args.stage2_style_lowpass_cutoff_hz,
            style_mid_low_bin=args.stage2_style_mid_low_bin,
            style_mid_high_bin=args.stage2_style_mid_high_bin,
            lowmid_split_bin=args.stage2_lowmid_split_bin,
            lowmid_gain=args.stage2_lowmid_gain,
            high_gain=args.stage2_high_gain,
            spectral_tilt_max_ratio=args.stage2_spectral_tilt_max_ratio,
            zcr_proxy_target_max=args.stage2_zcr_proxy_target_max,
            style_thaw_last_epochs=args.stage2_style_thaw_last_epochs,
            style_thaw_lr=args.stage2_style_thaw_lr,
            style_thaw_scope=args.stage2_style_thaw_scope,
            d_real_label=args.d_real_label,
            d_fake_label=args.d_fake_label,
            g_real_label=args.g_real_label,
            mrstft_resolutions=mrstft_resolutions,
        )
        hist2 = train_stage2(
            generator=generator,
            discriminator=discriminator,
            frozen_encoder=frozen_encoder,
            train_loader=train_loader,
            cond_bank=cond_bank,
            style_proto_bank=style_proto_bank,
            style_exemplar_bank=style_exemplar_bank,
            target_profile_bank=target_profile_bank,
            device=device,
            stage_cfg=cfg2,
            checkpoint_path=ckpt_stage2,
            resume=(args.mode == "resume"),
        )
        _append_history(history_path, hist2)
        stage2_epoch_done = _checkpoint_epoch(ckpt_stage2)
        state["stage2_done"] = bool(stage2_epoch_done >= int(args.stage2_epochs))
        state["stage2_epoch_done"] = int(stage2_epoch_done)
        _save_json(state, state_path)
    else:
        if ckpt_stage2.exists():
            payload = torch.load(str(ckpt_stage2), map_location="cpu")
            generator.load_state_dict(payload["generator"], strict=False)
            discriminator.load_state_dict(payload["discriminator"], strict=False)
            print(f"[lab3] stage2 checkpoint loaded (epoch={stage2_epoch_done}).")
            state["stage2_epoch_done"] = int(stage2_epoch_done)
            state["stage2_done"] = bool(stage2_epoch_done >= int(args.stage2_epochs))
            _save_json(state, state_path)

    # ---------------------------
    # Final evaluation / exit audit
    # ---------------------------
    if not args.skip_eval:
        print("[lab3] fitting third-party style classifier...")
        style_clf = fit_third_party_style_classifier(
            z_style_train=arrays["z_style"][train_idx],
            genre_idx_train=arrays["genre_idx"][train_idx],
        )
        clf_quality = evaluate_classifier_quality(
            style_classifier=style_clf,
            z_style_val=arrays["z_style"][val_idx],
            genre_idx_val=arrays["genre_idx"][val_idx],
        )
        audit = evaluate_genre_shift(
            generator=generator,
            frozen_encoder=frozen_encoder,
            val_loader=val_loader,
            cond_bank=cond_bank,
            style_classifier=style_clf,
            device=device,
            mps_threshold=args.mps_threshold,
            sf_threshold=args.sf_threshold,
            max_batches=args.eval_max_batches,
        )
        audit["classifier_quality"] = clf_quality
        audit["run_dir"] = str(out_dir)
        _save_json(audit, out_dir / "lab3_exit_audit.json")

        state["eval_done"] = True
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        state["lab3_done"] = bool(audit.get("passes", {}).get("lab3_done", False))
        state["audit"] = audit
        _save_json(state, state_path)
        print(f"[lab3] audit: {audit}")

    # ---------------------------
    # Post-train sample export (always available via CLI)
    # ---------------------------
    if bool(args.auto_sample_export):
        if ckpt_stage2.exists():
            print("[lab3] exporting post-train sample pack...")
            sample_out = out_dir / "samples" / str(args.sample_export_tag)
            genre_source_map = {
                str(g): int(genre_to_source_idx[int(i)].item()) for g, i in genre_to_idx.items()
            }
            try:
                sample_info = export_posttrain_samples(
                    generator=generator,
                    frozen_encoder=frozen_encoder,
                    arrays=arrays,
                    index_df=index_df,
                    genre_to_idx=genre_to_idx,
                    cond_bank=cond_bank,
                    out_dir=sample_out,
                    val_idx=val_idx,
                    n_samples=int(args.sample_count),
                    target_mode=str(args.sample_target_mode),
                    griffin_lim_iters=int(args.sample_griffin_lim_iters),
                    seed=int(args.seed),
                    device=device,
                    genre_to_source_idx=genre_source_map,
                    write_real_audio=bool(args.sample_write_real_audio),
                )
                state["sample_export"] = {
                    "enabled": True,
                    "ok": True,
                    "output_dir": sample_info.get("output_dir", str(sample_out)),
                    "summary_csv": sample_info.get("summary_csv", str(sample_out / "generation_summary.csv")),
                    "meta_json": sample_info.get("meta_json", str(sample_out / "sample_export_meta.json")),
                }
                _save_json(state, state_path)
                print(f"[lab3] sample export done: {sample_info}")
            except Exception as e:
                state["sample_export"] = {
                    "enabled": True,
                    "ok": False,
                    "error": str(e),
                    "output_dir": str(sample_out),
                }
                _save_json(state, state_path)
                raise
        else:
            print("[lab3] sample export skipped: stage2 checkpoint missing.")

    print(f"[lab3] done. run_dir={out_dir}")


if __name__ == "__main__":
    main()
