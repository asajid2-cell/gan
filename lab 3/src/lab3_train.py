from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .lab3_bridge import FrozenLab1Encoder, denormalize_log_mel
from .lab3_losses import (
    bandpass_mel_db,
    cosine_distance,
    estimate_lowpass_keep_bins,
    frequency_weighted_l1_loss,
    hf_excess_penalty_from_db,
    highpass_recon_l1_loss,
    lowpass_mel_db,
    multi_resolution_stft_loss,
    spectral_tilt_penalty_from_db,
    spectral_flatness_penalty_from_db,
    timbre_balance_penalty_from_db,
    zcr_proxy_penalty_from_db,
)
from .lab3_models import MelDiscriminator, ReconstructionDecoder, StyleCritic, bce_logits


@dataclass
class TrainWeights:
    adv: float = 0.5
    recon_l1: float = 8.0
    content: float = 2.0
    style: float = 1.0
    continuity: float = 1.0
    mrstft: float = 0.0
    flatness: float = 0.5
    feature_match: float = 0.0
    perceptual: float = 0.0
    style_hinge: float = 0.0
    contrastive: float = 0.0
    batch_infonce_diversity: float = 0.0
    pfm_style: float = 0.0
    diversity: float = 0.0
    timbre_balance: float = 0.0
    lowmid_recon: float = 0.0
    spectral_tilt: float = 0.0
    zcr_proxy: float = 0.0
    hf_muzzle: float = 0.0
    highpass_anchor: float = 0.0
    style_mid: float = 0.0
    mel_diversity: float = 0.0
    target_profile: float = 0.0


@dataclass
class StageTrainConfig:
    stage_name: str
    epochs: int
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    max_batches_per_epoch: Optional[int] = None
    gan_loss: str = "bce"
    disc_use_content_cond: bool = False
    weights: TrainWeights = field(default_factory=TrainWeights)
    # Stage-2 controls (ignored by stage1)
    content_weight_start: Optional[float] = None
    content_weight_end: Optional[float] = None
    style_label_smoothing: float = 0.0
    style_only_warmup_epochs: int = 0
    g_lr_warmup_epochs: int = 0
    g_lr_start_mult: float = 1.0
    cond_noise_std: float = 0.0
    style_jitter_std: float = 0.0
    cond_mode: str = "mix"  # centroid | exemplar | mix
    cond_alpha_start: float = 0.8
    cond_alpha_end: float = 0.4
    cond_exemplar_noise_std: float = 0.03
    target_balance: bool = True
    style_hinge_target_conf: float = 0.85
    adaptive_content_weight: bool = False
    adaptive_content_low: float = 0.05
    adaptive_content_high: float = 0.5
    adaptive_conf_low: float = 0.5
    adaptive_conf_high: float = 0.8
    style_class_balance: bool = False
    style_critic_lr: float = 2e-4
    contrastive_temp: float = 0.10
    diversity_margin: float = 0.90
    diversity_max_pairs: int = 64
    style_lowpass_keep_bins: int = 80
    style_lowpass_cutoff_hz: Optional[float] = None
    style_mid_low_bin: int = 8
    style_mid_high_bin: int = 56
    lowmid_split_bin: int = 80
    lowmid_gain: float = 5.0
    high_gain: float = 0.5
    spectral_tilt_max_ratio: float = 0.7
    zcr_proxy_target_max: float = 0.18
    hf_top_frac: float = 0.20
    hf_max_ratio_multiplier: float = 1.20
    highpass_cut_bin: int = 12
    highpass_gain: float = 2.0
    mel_diversity_margin: float = 0.85
    mel_diversity_max_pairs: int = 64
    d_real_label: float = 1.0
    d_fake_label: float = 0.0
    g_real_label: float = 1.0
    r1_gamma: float = 0.0
    r1_interval: int = 16
    d_loss_floor_for_step: float = 0.0
    d_step_period: int = 1
    g_grad_clip_norm: float = 0.0
    d_grad_clip_norm: float = 0.0
    style_thaw_last_epochs: int = 0
    style_thaw_lr: float = 1e-6
    style_thaw_scope: str = "style_head"
    batch_infonce_temp: float = 0.15
    mrstft_resolutions: Tuple[Tuple[int, int, int], ...] = (
        (64, 16, 64),
        (128, 32, 128),
        (256, 64, 256),
    )


def load_target_centroids(target_centroids_json: Path) -> Dict[str, np.ndarray]:
    p = Path(target_centroids_json)
    if not p.exists():
        raise FileNotFoundError(f"Target centroid file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, np.ndarray] = {}
    for genre, rec in obj.items():
        if "vector" in rec:
            vec = rec["vector"]
        elif "vector160" in rec:
            vec = rec["vector160"]
        else:
            raise ValueError(f"Missing vector field for genre {genre}")
        v = np.asarray(vec, dtype=np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        out[str(genre)] = v
    return out


def build_condition_bank(genre_to_idx: Dict[str, int], centroid_map: Dict[str, np.ndarray]) -> torch.Tensor:
    missing = [g for g in genre_to_idx.keys() if g not in centroid_map]
    if missing:
        raise ValueError(f"Centroid map missing genres: {missing}")
    dim = int(len(next(iter(centroid_map.values()))))
    bank = np.zeros((len(genre_to_idx), dim), dtype=np.float32)
    for genre, idx in genre_to_idx.items():
        bank[int(idx)] = centroid_map[genre]
    return torch.from_numpy(bank)


def build_style_centroid_bank(z_style: np.ndarray, genre_idx: np.ndarray, n_genres: int) -> torch.Tensor:
    bank = np.zeros((int(n_genres), z_style.shape[1]), dtype=np.float32)
    for g in range(int(n_genres)):
        mask = genre_idx == g
        if np.any(mask):
            v = z_style[mask].mean(axis=0).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            bank[g] = v
    return torch.from_numpy(bank)


def build_style_exemplar_bank(
    z_style: np.ndarray,
    genre_idx: np.ndarray,
    n_genres: int,
) -> Dict[int, torch.Tensor]:
    bank: Dict[int, torch.Tensor] = {}
    for g in range(int(n_genres)):
        mask = genre_idx == g
        if np.any(mask):
            vals = z_style[mask].astype(np.float32)
            vals = vals / (np.linalg.norm(vals, axis=1, keepdims=True) + 1e-8)
            bank[int(g)] = torch.from_numpy(vals)
    return bank


def _sample_shift_targets(source_idx: torch.Tensor, n_genres: int) -> torch.Tensor:
    device = source_idx.device
    n = int(source_idx.shape[0])
    tgt = torch.randint(low=0, high=int(n_genres), size=(n,), device=device)
    clash = tgt == source_idx
    if clash.any():
        tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt


def _sample_shift_targets_balanced(source_idx: torch.Tensor, n_genres: int, offset: int = 0) -> torch.Tensor:
    device = source_idx.device
    n = int(source_idx.shape[0])
    base = (torch.arange(n, device=device, dtype=torch.long) + int(offset)) % int(n_genres)
    tgt = base.clone()
    clash = tgt == source_idx
    if clash.any():
        tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt


def _save_stage_checkpoint(
    ckpt_path: Path,
    epoch: int,
    generator: ReconstructionDecoder,
    discriminator: MelDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    meta: Dict,
    style_critic: Optional[StyleCritic] = None,
    opt_sc: Optional[torch.optim.Optimizer] = None,
    frozen_encoder_model: Optional[torch.nn.Module] = None,
    opt_style_thaw: Optional[torch.optim.Optimizer] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "meta": meta,
    }
    if style_critic is not None:
        payload["style_critic"] = style_critic.state_dict()
    if opt_sc is not None:
        payload["opt_sc"] = opt_sc.state_dict()
    if frozen_encoder_model is not None:
        payload["frozen_encoder_model"] = frozen_encoder_model.state_dict()
    if opt_style_thaw is not None:
        payload["opt_style_thaw"] = opt_style_thaw.state_dict()
    torch.save(payload, str(ckpt_path))


def _load_stage_checkpoint(
    ckpt_path: Path,
    generator: ReconstructionDecoder,
    discriminator: MelDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    style_critic: Optional[StyleCritic] = None,
    opt_sc: Optional[torch.optim.Optimizer] = None,
    frozen_encoder_model: Optional[torch.nn.Module] = None,
    opt_style_thaw: Optional[torch.optim.Optimizer] = None,
) -> int:
    if not ckpt_path.exists():
        return 0
    payload = torch.load(str(ckpt_path), map_location="cpu")
    generator.load_state_dict(payload["generator"], strict=False)
    discriminator.load_state_dict(payload["discriminator"], strict=False)
    try:
        opt_g.load_state_dict(payload["opt_g"])
        opt_d.load_state_dict(payload["opt_d"])
    except Exception:
        # Optimizer shapes can differ across architecture revisions; resume weights only.
        pass
    if style_critic is not None and "style_critic" in payload:
        style_critic.load_state_dict(payload["style_critic"], strict=False)
    if opt_sc is not None and "opt_sc" in payload:
        try:
            opt_sc.load_state_dict(payload["opt_sc"])
        except Exception:
            pass
    if frozen_encoder_model is not None and "frozen_encoder_model" in payload:
        frozen_encoder_model.load_state_dict(payload["frozen_encoder_model"], strict=False)
    if opt_style_thaw is not None and "opt_style_thaw" in payload:
        try:
            opt_style_thaw.load_state_dict(payload["opt_style_thaw"])
        except Exception:
            pass
    return int(payload.get("epoch", 0))


def _epoch_mean(d: Dict[str, float], n: int) -> Dict[str, float]:
    if n <= 0:
        return {k: float("nan") for k in d.keys()}
    return {k: float(v / n) for k, v in d.items()}


def _linear_weight(start: float, end: float, epoch_idx: int, total_epochs: int) -> float:
    if total_epochs <= 1:
        return float(end)
    t = float(epoch_idx) / float(total_epochs - 1)
    return float(start + (end - start) * t)


def _set_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for pg in opt.param_groups:
        pg["lr"] = float(lr)


def _gan_d_loss(
    gan_loss: str,
    logits_real: torch.Tensor,
    logits_fake: torch.Tensor,
    d_real_label: float,
    d_fake_label: float,
) -> torch.Tensor:
    key = str(gan_loss).strip().lower()
    if key == "hinge":
        return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())
    if key == "bce":
        return 0.5 * (
            bce_logits(logits_real, True, real_label=float(d_real_label), fake_label=float(d_fake_label))
            + bce_logits(logits_fake, False, real_label=float(d_real_label), fake_label=float(d_fake_label))
        )
    raise ValueError(f"Unsupported gan_loss: {gan_loss}")


def _gan_g_loss(
    gan_loss: str,
    logits_fake: torch.Tensor,
    g_real_label: float,
    d_fake_label: float,
) -> torch.Tensor:
    key = str(gan_loss).strip().lower()
    if key == "hinge":
        return -logits_fake.mean()
    if key == "bce":
        return bce_logits(logits_fake, True, real_label=float(g_real_label), fake_label=float(d_fake_label))
    raise ValueError(f"Unsupported gan_loss: {gan_loss}")


def _r1_penalty(discriminator: MelDiscriminator, real: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    real_req = real.detach().requires_grad_(True)
    logits = discriminator(real_req, cond)
    grad = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=real_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).reshape(grad.shape[0], -1).sum(dim=1).mean()


def train_stage1(
    generator: ReconstructionDecoder,
    discriminator: MelDiscriminator,
    frozen_encoder: FrozenLab1Encoder,
    train_loader: DataLoader,
    cond_bank: torch.Tensor,
    device: str,
    stage_cfg: StageTrainConfig,
    checkpoint_path: Path,
    resume: bool = False,
) -> List[Dict]:
    generator.train()
    discriminator.train()
    cond_bank = cond_bank.to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=stage_cfg.lr_g, betas=stage_cfg.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=stage_cfg.lr_d, betas=stage_cfg.betas)
    start_epoch = _load_stage_checkpoint(checkpoint_path, generator, discriminator, opt_g, opt_d) if resume else 0
    history: List[Dict] = []
    step_idx = 0
    step_idx = 0

    for epoch in range(start_epoch, int(stage_cfg.epochs)):
        agg = {
            "loss_g": 0.0,
            "loss_d": 0.0,
            "loss_r1": 0.0,
            "loss_adv": 0.0,
            "loss_l1": 0.0,
            "loss_content": 0.0,
            "loss_style": 0.0,
            "loss_continuity": 0.0,
            "loss_mrstft": 0.0,
        }
        nb = 0
        d_steps_taken = 0
        d_steps_skipped = 0
        for bidx, batch in enumerate(train_loader):
            if stage_cfg.max_batches_per_epoch is not None and bidx >= int(stage_cfg.max_batches_per_epoch):
                break
            real = batch["mel_norm"].to(device).float()
            zc = batch["z_content"].to(device).float()
            zs = batch["z_style"].to(device).float()
            gidx = batch["genre_idx"].to(device).long()
            cond = cond_bank[gidx]
            cond_d = torch.cat([cond, zc], dim=1) if bool(stage_cfg.disc_use_content_cond) else cond
            do_d_step = (step_idx % int(max(1, stage_cfg.d_step_period))) == 0

            # D step
            with torch.no_grad():
                fake_det = generator(zc, cond)
            logit_real = discriminator(real, cond_d)
            logit_fake = discriminator(fake_det, cond_d)
            loss_d = _gan_d_loss(
                gan_loss=stage_cfg.gan_loss,
                logits_real=logit_real,
                logits_fake=logit_fake,
                d_real_label=float(stage_cfg.d_real_label),
                d_fake_label=float(stage_cfg.d_fake_label),
            )
            r1 = torch.zeros((), device=real.device)
            if do_d_step:
                if float(stage_cfg.r1_gamma) > 0.0 and (step_idx % int(max(1, stage_cfg.r1_interval)) == 0):
                    r1 = 0.5 * float(stage_cfg.r1_gamma) * _r1_penalty(discriminator, real, cond_d)
                    loss_d = loss_d + r1
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if float(stage_cfg.d_grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=float(stage_cfg.d_grad_clip_norm))
                opt_d.step()
                d_steps_taken += 1
            else:
                d_steps_skipped += 1

            # G step
            fake = generator(zc, cond)
            logit_fake_g = discriminator(fake, cond_d)
            loss_adv = _gan_g_loss(
                gan_loss=stage_cfg.gan_loss,
                logits_fake=logit_fake_g,
                g_real_label=float(stage_cfg.g_real_label),
                d_fake_label=float(stage_cfg.d_fake_label),
            )
            loss_l1 = F.l1_loss(fake, real)
            loss_continuity = multi_resolution_stft_loss(
                fake,
                real,
                resolutions=stage_cfg.mrstft_resolutions,
            )
            loss_mrstft = loss_continuity

            fake_db = denormalize_log_mel(fake)
            enc_out = frozen_encoder.forward_log_mel_tensor(fake_db)
            loss_content = cosine_distance(enc_out["z_content"], zc).mean()
            loss_style = cosine_distance(enc_out["z_style"], zs).mean()

            w = stage_cfg.weights
            loss_g = (
                w.adv * loss_adv
                + w.recon_l1 * loss_l1
                + w.content * loss_content
                + w.style * loss_style
                + w.continuity * loss_continuity
                + w.mrstft * loss_mrstft
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if float(stage_cfg.g_grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=float(stage_cfg.g_grad_clip_norm))
            opt_g.step()

            nb += 1
            agg["loss_g"] += float(loss_g.item())
            agg["loss_d"] += float(loss_d.item())
            agg["loss_r1"] += float(r1.item())
            agg["loss_adv"] += float(loss_adv.item())
            agg["loss_l1"] += float(loss_l1.item())
            agg["loss_content"] += float(loss_content.item())
            agg["loss_style"] += float(loss_style.item())
            agg["loss_continuity"] += float(loss_continuity.item())
            agg["loss_mrstft"] += float(loss_mrstft.item())
            step_idx += 1

        ep = _epoch_mean(agg, nb)
        ep["stage"] = "stage1"
        ep["epoch"] = int(epoch + 1)
        ep["d_steps_taken"] = float(d_steps_taken)
        ep["d_steps_skipped"] = float(d_steps_skipped)
        ep["d_step_rate"] = float(d_steps_taken / max(1, (d_steps_taken + d_steps_skipped)))
        history.append(ep)
        _save_stage_checkpoint(
            checkpoint_path,
            epoch=int(epoch + 1),
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            meta={"stage": "stage1"},
        )
        print(f"[stage1] epoch={epoch + 1} loss_g={ep['loss_g']:.4f} loss_d={ep['loss_d']:.4f}")

    return history


def train_stage2(
    generator: ReconstructionDecoder,
    discriminator: MelDiscriminator,
    frozen_encoder: FrozenLab1Encoder,
    train_loader: DataLoader,
    cond_bank: torch.Tensor,
    style_proto_bank: torch.Tensor,
    style_exemplar_bank: Optional[Dict[int, torch.Tensor]],
    target_profile_bank: Optional[torch.Tensor],
    device: str,
    stage_cfg: StageTrainConfig,
    checkpoint_path: Path,
    resume: bool = False,
) -> List[Dict]:
    generator.train()
    discriminator.train()
    cond_bank = cond_bank.to(device)
    style_proto_bank = F.normalize(style_proto_bank.to(device), dim=-1)
    exemplar_bank_dev: Dict[int, torch.Tensor] = {}
    if style_exemplar_bank:
        for g, t in style_exemplar_bank.items():
            exemplar_bank_dev[int(g)] = F.normalize(t.to(device).float(), dim=-1)
    if target_profile_bank is not None:
        target_profile_bank = target_profile_bank.to(device)
    n_genres = int(cond_bank.shape[0])
    z_style_dim = int(style_proto_bank.shape[1])
    style_critic = StyleCritic(in_dim=z_style_dim, n_classes=n_genres).to(device)
    style_critic.train()
    style_thaw_last_epochs = max(0, int(stage_cfg.style_thaw_last_epochs))
    scope = str(stage_cfg.style_thaw_scope).strip().lower()
    if scope == "style_head":
        style_thaw_params = list(frozen_encoder.model.style_head.parameters())
    elif scope == "shared_style":
        style_thaw_params = list(frozen_encoder.model.shared.parameters()) + list(
            frozen_encoder.model.style_head.parameters()
        )
    elif scope == "full_style_path":
        style_thaw_params = (
            list(frozen_encoder.model.backbone.parameters())
            + list(frozen_encoder.model.shared.parameters())
            + list(frozen_encoder.model.style_head.parameters())
        )
    else:
        raise ValueError(f"Unsupported style_thaw_scope: {stage_cfg.style_thaw_scope}")
    for p in style_thaw_params:
        p.requires_grad = False
    opt_style_thaw = (
        torch.optim.Adam(style_thaw_params, lr=float(stage_cfg.style_thaw_lr), betas=stage_cfg.betas)
        if style_thaw_params
        else None
    )

    opt_g = torch.optim.Adam(generator.parameters(), lr=stage_cfg.lr_g, betas=stage_cfg.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=stage_cfg.lr_d, betas=stage_cfg.betas)
    opt_sc = torch.optim.Adam(style_critic.parameters(), lr=float(stage_cfg.style_critic_lr), betas=stage_cfg.betas)
    # Optional class weights for style losses; disable by default when target sampling is balanced.
    if bool(stage_cfg.style_class_balance):
        ds_genre = getattr(train_loader.dataset, "genre_idx", None)
        if ds_genre is not None:
            y = np.asarray(ds_genre, dtype=np.int64)
            counts = np.bincount(y, minlength=n_genres).astype(np.float32)
            inv = 1.0 / np.maximum(counts, 1.0)
            w_arr = inv / max(float(inv.mean()), 1e-8)
            class_w = torch.from_numpy(w_arr).to(device=device, dtype=torch.float32)
        else:
            class_w = torch.ones((n_genres,), device=device, dtype=torch.float32)
    else:
        class_w = torch.ones((n_genres,), device=device, dtype=torch.float32)
    start_epoch = (
        _load_stage_checkpoint(
            checkpoint_path,
            generator,
            discriminator,
            opt_g,
            opt_d,
            style_critic=style_critic,
            opt_sc=opt_sc,
            frozen_encoder_model=frozen_encoder.model,
            opt_style_thaw=opt_style_thaw,
        )
        if resume
        else 0
    )
    history: List[Dict] = []
    step_idx = 0

    def _contrastive_proto_loss(z_fake: torch.Tensor, tgt_idx: torch.Tensor, temp: float) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.normalize(z_fake, dim=-1)
        logits = torch.matmul(z, style_proto_bank.t()) / float(temp)
        loss = F.cross_entropy(logits, tgt_idx)
        probs = torch.softmax(logits, dim=1)
        conf = probs.gather(1, tgt_idx.unsqueeze(1)).squeeze(1).mean()
        return loss, conf

    def _diversity_loss(feat: torch.Tensor, tgt_idx: torch.Tensor, margin: float, max_pairs: int) -> torch.Tensor:
        if feat.shape[0] < 2:
            return torch.zeros((), device=feat.device)
        f = F.normalize(feat, dim=-1)
        loss_terms = []
        pair_budget = int(max_pairs)
        for g in torch.unique(tgt_idx):
            ids = torch.where(tgt_idx == g)[0]
            if ids.numel() < 2:
                continue
            perm = ids[torch.randperm(ids.numel(), device=feat.device)]
            a = perm[:-1]
            b = perm[1:]
            if a.numel() == 0:
                continue
            if pair_budget > 0 and a.numel() > pair_budget:
                a = a[:pair_budget]
                b = b[:pair_budget]
            sim = torch.sum(f[a] * f[b], dim=-1)
            loss_terms.append(F.relu(sim - float(margin)).mean())
            pair_budget -= int(a.numel())
            if pair_budget <= 0:
                break
        if not loss_terms:
            return torch.zeros((), device=feat.device)
        return torch.stack(loss_terms).mean()

    def _batch_infonce_diversity_loss(
        z_fake: torch.Tensor,
        tgt_idx: torch.Tensor,
        temp: float,
    ) -> torch.Tensor:
        """
        Intra-genre repulsive InfoNCE:
        - Positive: self (detached copy)
        - Negatives: other samples in the same target genre
        Minimization forces within-genre samples to avoid collapsing to identical style embeddings.
        """
        if z_fake.shape[0] < 2:
            return torch.zeros((), device=z_fake.device)
        z = F.normalize(z_fake, dim=-1)
        loss_terms = []
        for g in torch.unique(tgt_idx):
            ids = torch.where(tgt_idx == g)[0]
            if ids.numel() < 2:
                continue
            zg = z[ids]
            logits = torch.matmul(zg, zg.detach().t()) / float(temp)  # [M, M]
            labels = torch.arange(zg.shape[0], device=z_fake.device)
            loss_terms.append(F.cross_entropy(logits, labels))
        if not loss_terms:
            return torch.zeros((), device=z_fake.device)
        return torch.stack(loss_terms).mean()

    def _mel_diversity_loss(fake_mel: torch.Tensor, tgt_idx: torch.Tensor, margin: float, max_pairs: int) -> torch.Tensor:
        if fake_mel.shape[0] < 2:
            return torch.zeros((), device=fake_mel.device)
        f = F.normalize(fake_mel.flatten(1), dim=-1)
        loss_terms = []
        pair_budget = int(max_pairs)
        for g in torch.unique(tgt_idx):
            ids = torch.where(tgt_idx == g)[0]
            if ids.numel() < 2:
                continue
            perm = ids[torch.randperm(ids.numel(), device=fake_mel.device)]
            a = perm[:-1]
            b = perm[1:]
            if a.numel() == 0:
                continue
            if pair_budget > 0 and a.numel() > pair_budget:
                a = a[:pair_budget]
                b = b[:pair_budget]
            sim = torch.sum(f[a] * f[b], dim=-1)
            loss_terms.append(F.relu(sim - float(margin)).mean())
            pair_budget -= int(a.numel())
            if pair_budget <= 0:
                break
        if not loss_terms:
            return torch.zeros((), device=fake_mel.device)
        return torch.stack(loss_terms).mean()

    for epoch in range(start_epoch, int(stage_cfg.epochs)):
        thaw_active = bool(style_thaw_last_epochs > 0 and epoch >= int(stage_cfg.epochs) - style_thaw_last_epochs)
        for p in style_thaw_params:
            p.requires_grad = thaw_active
        style_only_warmup_epochs = max(0, int(stage_cfg.style_only_warmup_epochs))
        g_lr_warmup_epochs = max(0, int(stage_cfg.g_lr_warmup_epochs))
        base_lr_g = float(stage_cfg.lr_g)
        if g_lr_warmup_epochs > 0 and epoch < g_lr_warmup_epochs:
            start_mult = float(stage_cfg.g_lr_start_mult)
            t = float(epoch + 1) / float(g_lr_warmup_epochs)
            lr_mult = start_mult + (1.0 - start_mult) * t
            cur_lr_g = base_lr_g * float(lr_mult)
        else:
            cur_lr_g = base_lr_g
        _set_optimizer_lr(opt_g, cur_lr_g)

        if epoch < style_only_warmup_epochs:
            content_w = 0.0
        elif stage_cfg.content_weight_start is not None and stage_cfg.content_weight_end is not None:
            sched_total = max(1, int(stage_cfg.epochs) - style_only_warmup_epochs)
            sched_epoch = int(epoch) - style_only_warmup_epochs
            content_w = _linear_weight(
                start=float(stage_cfg.content_weight_start),
                end=float(stage_cfg.content_weight_end),
                epoch_idx=int(sched_epoch),
                total_epochs=int(sched_total),
            )
        else:
            content_w = float(stage_cfg.weights.content)
        agg = {
            "loss_g": 0.0,
            "loss_d": 0.0,
            "loss_r1": 0.0,
            "loss_sc": 0.0,
            "loss_adv": 0.0,
            "loss_l1": 0.0,
            "loss_continuity": 0.0,
            "loss_mrstft": 0.0,
            "loss_content": 0.0,
            "loss_style_ce": 0.0,
            "loss_style_mid": 0.0,
            "loss_pfm_style": 0.0,
            "loss_contrastive": 0.0,
            "loss_batch_infonce_diversity": 0.0,
            "loss_style_hinge": 0.0,
            "loss_lowmid_recon": 0.0,
            "loss_flatness": 0.0,
            "loss_timbre_balance": 0.0,
            "loss_spectral_tilt": 0.0,
            "loss_zcr_proxy": 0.0,
            "loss_hf_muzzle": 0.0,
            "loss_highpass_anchor": 0.0,
            "loss_feature_match": 0.0,
            "loss_perceptual": 0.0,
            "loss_diversity": 0.0,
            "loss_mel_diversity": 0.0,
            "loss_target_profile": 0.0,
            "content_weight": 0.0,
            "lr_g": 0.0,
            "style_conf": 0.0,
            "style_thaw_active": 0.0,
            "cond_alpha": 0.0,
        }
        nb = 0
        d_steps_taken = 0
        d_steps_skipped = 0
        ema_style_conf = None
        style_lp_bins = None
        for bidx, batch in enumerate(train_loader):
            if stage_cfg.max_batches_per_epoch is not None and bidx >= int(stage_cfg.max_batches_per_epoch):
                break
            real = batch["mel_norm"].to(device).float()
            zc = batch["z_content"].to(device).float()
            gidx = batch["genre_idx"].to(device).long()
            if bool(stage_cfg.target_balance):
                tgt_idx = _sample_shift_targets_balanced(gidx, n_genres=n_genres, offset=step_idx)
            else:
                tgt_idx = _sample_shift_targets(gidx, n_genres=n_genres)
            cond_centroid = cond_bank[tgt_idx]
            cond_mode = str(stage_cfg.cond_mode).strip().lower()

            if cond_mode in {"exemplar", "mix"} and exemplar_bank_dev:
                ex_rows = []
                for j in range(int(tgt_idx.shape[0])):
                    tg = int(tgt_idx[j].item())
                    ex_pool = exemplar_bank_dev.get(tg)
                    if ex_pool is None or ex_pool.shape[0] == 0:
                        ex_rows.append(cond_centroid[j])
                        continue
                    rid = int(torch.randint(0, int(ex_pool.shape[0]), (1,), device=tgt_idx.device).item())
                    exv = ex_pool[rid]
                    if float(stage_cfg.cond_exemplar_noise_std) > 0.0:
                        exv = F.normalize(
                            exv + torch.randn_like(exv) * float(stage_cfg.cond_exemplar_noise_std),
                            dim=-1,
                        )
                    ex_rows.append(exv)
                cond_exemplar_raw = torch.stack(ex_rows, dim=0)
                # Map exemplar style latents to cond dim.
                # If cond includes extra descriptor dims (e.g., 160 vs 128), keep centroid tail as-is.
                cond_exemplar = cond_centroid.clone()
                k = min(int(cond_exemplar.shape[1]), int(cond_exemplar_raw.shape[1]))
                cond_exemplar[:, :k] = cond_exemplar_raw[:, :k]
                cond_exemplar = F.normalize(cond_exemplar, dim=-1)
            else:
                cond_exemplar = cond_centroid

            if cond_mode == "exemplar":
                cond = cond_exemplar
                cond_alpha = 0.0
            elif cond_mode == "mix":
                cond_alpha = _linear_weight(
                    float(stage_cfg.cond_alpha_start),
                    float(stage_cfg.cond_alpha_end),
                    epoch_idx=int(epoch),
                    total_epochs=int(stage_cfg.epochs),
                )
                cond = F.normalize(
                    float(cond_alpha) * cond_centroid + (1.0 - float(cond_alpha)) * cond_exemplar,
                    dim=-1,
                )
            else:
                cond = cond_centroid
                cond_alpha = 1.0
            noise_parts = []
            if float(stage_cfg.cond_noise_std) > 0.0:
                noise_parts.append(torch.randn_like(cond) * float(stage_cfg.cond_noise_std))
            if float(stage_cfg.style_jitter_std) > 0.0:
                noise_parts.append(torch.randn_like(cond) * float(stage_cfg.style_jitter_std))
            if noise_parts:
                cond = F.normalize(cond + sum(noise_parts), dim=-1)
            cond_d = torch.cat([cond, zc], dim=1) if bool(stage_cfg.disc_use_content_cond) else cond
            do_d_step = (step_idx % int(max(1, stage_cfg.d_step_period))) == 0

            real_db = denormalize_log_mel(real)
            if style_lp_bins is None:
                if stage_cfg.style_lowpass_cutoff_hz is not None and float(stage_cfg.style_lowpass_cutoff_hz) > 0.0:
                    style_lp_bins = estimate_lowpass_keep_bins(
                        n_mels=int(real_db.shape[1]),
                        sample_rate=int(frozen_encoder.cfg.sample_rate),
                        cutoff_hz=float(stage_cfg.style_lowpass_cutoff_hz),
                    )
                else:
                    style_lp_bins = int(stage_cfg.style_lowpass_keep_bins)
            real_db_lp = lowpass_mel_db(real_db, keep_bins=int(style_lp_bins))
            # Style critic step (real domain): keep a train-time critic aligned with current latents.
            with torch.no_grad():
                enc_real_lp = frozen_encoder.forward_log_mel_tensor(real_db_lp)
            z_style_real = enc_real_lp["z_style"].detach()
            sc_real_logits = style_critic(z_style_real)
            loss_sc = F.cross_entropy(sc_real_logits, gidx, weight=class_w)
            opt_sc.zero_grad(set_to_none=True)
            loss_sc.backward()
            opt_sc.step()

            # D step
            with torch.no_grad():
                fake_det = generator(zc, cond)
            logit_real = discriminator(real, cond_d)
            logit_fake = discriminator(fake_det, cond_d)
            loss_d = _gan_d_loss(
                gan_loss=stage_cfg.gan_loss,
                logits_real=logit_real,
                logits_fake=logit_fake,
                d_real_label=float(stage_cfg.d_real_label),
                d_fake_label=float(stage_cfg.d_fake_label),
            )
            r1 = torch.zeros((), device=real.device)
            if do_d_step and float(stage_cfg.r1_gamma) > 0.0 and (step_idx % int(max(1, stage_cfg.r1_interval)) == 0):
                r1 = 0.5 * float(stage_cfg.r1_gamma) * _r1_penalty(discriminator, real, cond_d)
                loss_d = loss_d + r1
            if do_d_step and (float(stage_cfg.d_loss_floor_for_step) <= 0.0 or float(loss_d.item()) >= float(stage_cfg.d_loss_floor_for_step)):
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if float(stage_cfg.d_grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=float(stage_cfg.d_grad_clip_norm))
                opt_d.step()
                d_steps_taken += 1
            else:
                d_steps_skipped += 1

            # G step
            fake = generator(zc, cond)
            logit_fake_g, d_fake_feat = discriminator(fake, cond_d, return_features=True)
            loss_adv = _gan_g_loss(
                gan_loss=stage_cfg.gan_loss,
                logits_fake=logit_fake_g,
                g_real_label=float(stage_cfg.g_real_label),
                d_fake_label=float(stage_cfg.d_fake_label),
            )
            with torch.no_grad():
                _, d_real_feat = discriminator(real, cond_d, return_features=True)
            loss_feature_match = F.l1_loss(d_fake_feat, d_real_feat)

            fake_db = denormalize_log_mel(fake)
            fake_db_lp = lowpass_mel_db(fake_db, keep_bins=int(style_lp_bins))
            fake_db_mid = bandpass_mel_db(
                fake_db,
                low_bin=int(stage_cfg.style_mid_low_bin),
                high_bin=int(stage_cfg.style_mid_high_bin),
            )
            enc_out = frozen_encoder.forward_log_mel_tensor(fake_db_lp)
            enc_mid = frozen_encoder.forward_log_mel_tensor(fake_db_mid)
            loss_content = cosine_distance(enc_out["z_content"], zc).mean()
            fake_zs = enc_out["z_style"]
            style_logits = style_critic(fake_zs)
            loss_style = F.cross_entropy(
                style_logits,
                tgt_idx,
                weight=class_w,
                label_smoothing=float(stage_cfg.style_label_smoothing),
            )
            probs = torch.softmax(style_logits, dim=1)
            conf = probs.gather(1, tgt_idx.unsqueeze(1)).squeeze(1)
            style_conf = conf.mean()
            # Mid-band style objective: force style cues in 250Hz-2k-ish region.
            style_logits_mid = style_critic(enc_mid["z_style"])
            loss_style_mid = F.cross_entropy(
                style_logits_mid,
                tgt_idx,
                weight=class_w,
                label_smoothing=float(stage_cfg.style_label_smoothing),
            )
            # Hinge-style pressure toward target confidence threshold.
            loss_style_hinge = F.relu(float(stage_cfg.style_hinge_target_conf) - conf).mean()
            loss_contrastive, contrastive_conf = _contrastive_proto_loss(
                fake_zs, tgt_idx, temp=float(stage_cfg.contrastive_temp)
            )
            # Target-style perceptual feature matching:
            # For each sample, pick a real reference from the target genre (within batch if available).
            genre_to_ids: Dict[int, List[int]] = {}
            for j in range(int(gidx.shape[0])):
                gj = int(gidx[j].item())
                genre_to_ids.setdefault(gj, []).append(j)
            ref_ids = []
            n_batch = int(gidx.shape[0])
            for j in range(n_batch):
                tg = int(tgt_idx[j].item())
                cands = genre_to_ids.get(tg, [])
                if cands:
                    rid = cands[int(torch.randint(0, len(cands), (1,), device=gidx.device).item())]
                else:
                    rid = int(torch.randint(0, n_batch, (1,), device=gidx.device).item())
                ref_ids.append(rid)
            ref_idx_t = torch.tensor(ref_ids, device=gidx.device, dtype=torch.long)
            real_ref_db_lp = real_db_lp[ref_idx_t]
            with torch.no_grad():
                enc_ref_lp = frozen_encoder.forward_log_mel_tensor(real_ref_db_lp)
            loss_pfm_style = (
                F.mse_loss(
                    F.normalize(enc_out["shared_feat"], dim=-1),
                    F.normalize(enc_ref_lp["shared_feat"].detach(), dim=-1),
                )
                + F.mse_loss(
                    F.normalize(fake_zs, dim=-1),
                    F.normalize(enc_ref_lp["z_style"].detach(), dim=-1),
                )
            )
            loss_batch_infonce_div = _batch_infonce_diversity_loss(
                fake_zs,
                tgt_idx=tgt_idx,
                temp=float(stage_cfg.batch_infonce_temp),
            )
            loss_l1 = F.l1_loss(fake, real)
            loss_continuity = multi_resolution_stft_loss(
                fake,
                real,
                resolutions=stage_cfg.mrstft_resolutions,
            )
            loss_mrstft = loss_continuity
            loss_lowmid_recon = frequency_weighted_l1_loss(
                fake,
                real,
                split_bin=int(stage_cfg.lowmid_split_bin),
                low_gain=float(stage_cfg.lowmid_gain),
                high_gain=float(stage_cfg.high_gain),
            )
            loss_highpass_anchor = highpass_recon_l1_loss(
                fake,
                real,
                low_cut_bin=int(stage_cfg.highpass_cut_bin),
                gain=float(stage_cfg.highpass_gain),
            )
            loss_flatness = spectral_flatness_penalty_from_db(fake_db)
            loss_timbre_balance = timbre_balance_penalty_from_db(fake_db)
            loss_spectral_tilt = spectral_tilt_penalty_from_db(
                fake_db, hf_to_lf_max_ratio=float(stage_cfg.spectral_tilt_max_ratio)
            )
            loss_zcr_proxy = zcr_proxy_penalty_from_db(
                fake_db, target_max=float(stage_cfg.zcr_proxy_target_max)
            )
            loss_hf_muzzle = hf_excess_penalty_from_db(
                fake_db,
                real_db,
                top_frac=float(stage_cfg.hf_top_frac),
                max_ratio_multiplier=float(stage_cfg.hf_max_ratio_multiplier),
            )
            loss_perceptual = F.l1_loss(enc_out["shared_feat"], enc_real_lp["shared_feat"].detach())
            loss_div = _diversity_loss(
                d_fake_feat,
                tgt_idx=tgt_idx,
                margin=float(stage_cfg.diversity_margin),
                max_pairs=int(stage_cfg.diversity_max_pairs),
            )
            loss_mel_div = _mel_diversity_loss(
                fake,
                tgt_idx=tgt_idx,
                margin=float(stage_cfg.mel_diversity_margin),
                max_pairs=int(stage_cfg.mel_diversity_max_pairs),
            )
            if target_profile_bank is not None:
                p_fake = torch.pow(10.0, fake_db / 10.0).clamp_min(1e-10)
                fake_profile = p_fake.sum(dim=2)
                fake_profile = fake_profile / (fake_profile.sum(dim=1, keepdim=True) + 1e-8)
                tgt_profile = target_profile_bank[tgt_idx]
                loss_target_profile = F.l1_loss(fake_profile, tgt_profile)
            else:
                loss_target_profile = torch.zeros((), device=fake.device)

            if stage_cfg.adaptive_content_weight:
                # Use a blended confidence so adaptive schedule reacts to both critic and prototype alignment.
                cur_conf = float((0.5 * style_conf + 0.5 * contrastive_conf).item())
                if ema_style_conf is None:
                    ema_style_conf = cur_conf
                else:
                    ema_style_conf = 0.9 * ema_style_conf + 0.1 * cur_conf
                if ema_style_conf < float(stage_cfg.adaptive_conf_low):
                    content_w = float(stage_cfg.adaptive_content_low)
                elif ema_style_conf > float(stage_cfg.adaptive_conf_high):
                    content_w = float(stage_cfg.adaptive_content_high)

            w = stage_cfg.weights
            loss_g = (
                w.adv * loss_adv
                + w.recon_l1 * loss_l1
                + w.continuity * loss_continuity
                + float(content_w) * loss_content
                + w.style * loss_style
                + w.style_mid * loss_style_mid
                + w.pfm_style * loss_pfm_style
                + w.style_hinge * loss_style_hinge
                + w.contrastive * loss_contrastive
                + w.batch_infonce_diversity * loss_batch_infonce_div
                + w.mrstft * loss_mrstft
                + w.lowmid_recon * loss_lowmid_recon
                + w.highpass_anchor * loss_highpass_anchor
                + w.flatness * loss_flatness
                + w.timbre_balance * loss_timbre_balance
                + w.spectral_tilt * loss_spectral_tilt
                + w.zcr_proxy * loss_zcr_proxy
                + w.hf_muzzle * loss_hf_muzzle
                + w.feature_match * loss_feature_match
                + w.perceptual * loss_perceptual
                + w.diversity * loss_div
                + w.mel_diversity * loss_mel_div
                + w.target_profile * loss_target_profile
            )
            opt_g.zero_grad(set_to_none=True)
            if thaw_active and opt_style_thaw is not None:
                opt_style_thaw.zero_grad(set_to_none=True)
            loss_g.backward()
            if float(stage_cfg.g_grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=float(stage_cfg.g_grad_clip_norm))
                if thaw_active and style_thaw_params:
                    torch.nn.utils.clip_grad_norm_(style_thaw_params, max_norm=float(stage_cfg.g_grad_clip_norm))
            opt_g.step()
            if thaw_active and opt_style_thaw is not None:
                opt_style_thaw.step()

            nb += 1
            agg["loss_g"] += float(loss_g.item())
            agg["loss_d"] += float(loss_d.item())
            agg["loss_r1"] += float(r1.item())
            agg["loss_sc"] += float(loss_sc.item())
            agg["loss_adv"] += float(loss_adv.item())
            agg["loss_l1"] += float(loss_l1.item())
            agg["loss_continuity"] += float(loss_continuity.item())
            agg["loss_mrstft"] += float(loss_mrstft.item())
            agg["loss_content"] += float(loss_content.item())
            agg["loss_style_ce"] += float(loss_style.item())
            agg["loss_style_mid"] += float(loss_style_mid.item())
            agg["loss_pfm_style"] += float(loss_pfm_style.item())
            agg["loss_contrastive"] += float(loss_contrastive.item())
            agg["loss_batch_infonce_diversity"] += float(loss_batch_infonce_div.item())
            agg["loss_style_hinge"] += float(loss_style_hinge.item())
            agg["loss_lowmid_recon"] += float(loss_lowmid_recon.item())
            agg["loss_highpass_anchor"] += float(loss_highpass_anchor.item())
            agg["loss_flatness"] += float(loss_flatness.item())
            agg["loss_timbre_balance"] += float(loss_timbre_balance.item())
            agg["loss_spectral_tilt"] += float(loss_spectral_tilt.item())
            agg["loss_zcr_proxy"] += float(loss_zcr_proxy.item())
            agg["loss_hf_muzzle"] += float(loss_hf_muzzle.item())
            agg["loss_feature_match"] += float(loss_feature_match.item())
            agg["loss_perceptual"] += float(loss_perceptual.item())
            agg["loss_diversity"] += float(loss_div.item())
            agg["loss_mel_diversity"] += float(loss_mel_div.item())
            agg["loss_target_profile"] += float(loss_target_profile.item())
            agg["content_weight"] += float(content_w)
            agg["lr_g"] += float(cur_lr_g)
            agg["style_conf"] += float(style_conf.item())
            agg["style_thaw_active"] += 1.0 if thaw_active else 0.0
            agg.setdefault("cond_alpha", 0.0)
            agg["cond_alpha"] += float(cond_alpha)
            step_idx += 1

        ep = _epoch_mean(agg, nb)
        ep["stage"] = "stage2"
        ep["epoch"] = int(epoch + 1)
        ep["d_steps_taken"] = float(d_steps_taken)
        ep["d_steps_skipped"] = float(d_steps_skipped)
        ep["d_step_rate"] = float(d_steps_taken / max(1, (d_steps_taken + d_steps_skipped)))
        history.append(ep)
        _save_stage_checkpoint(
            checkpoint_path,
            epoch=int(epoch + 1),
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            meta={"stage": "stage2"},
            style_critic=style_critic,
            opt_sc=opt_sc,
            frozen_encoder_model=frozen_encoder.model,
            opt_style_thaw=opt_style_thaw,
        )
        print(f"[stage2] epoch={epoch + 1} loss_g={ep['loss_g']:.4f} loss_d={ep['loss_d']:.4f}")

    return history
