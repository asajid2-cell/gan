from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from .lab3_codec_bridge import FrozenEncodec
from .lab3_codec_judge import CodecStyleJudge, Lab1StyleProbe, MERTStyleProbe
from .lab3_codec_models import (
    CodecLatentTranslator,
    CodecTrainWeights,
    MultiScaleWaveDiscriminator,
    multiscale_feature_matching_loss,
    multiscale_hinge_d_loss,
    multiscale_hinge_g_loss,
)


@dataclass
class CodecStageTrainConfig:
    stage_name: str
    epochs: int
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.999)
    max_batches_per_epoch: Optional[int] = None
    target_balance: bool = True
    cond_mode: str = "mix"  # centroid | exemplar | mix
    cond_alpha_start: float = 0.8
    cond_alpha_end: float = 0.4
    style_dropout_p: float = 0.0
    style_jitter_std: float = 0.0
    exemplar_noise_std: float = 0.03
    d_step_period: int = 1
    r1_gamma: float = 0.0
    r1_interval: int = 16
    g_grad_clip_norm: float = 0.0
    d_grad_clip_norm: float = 0.0
    weights: CodecTrainWeights = field(default_factory=CodecTrainWeights)
    mode_seeking_noise_scale: float = 1.0
    mode_seeking_target: float = 0.03
    style_push_margin: float = 0.30
    delta_budget: float = 0.12
    style_loss_mode: str = "lab1_cos"  # lab1_cos | codec_judge_ce
    style_embed_align_weight: float = 0.0
    wave_mrstft_resolutions: Tuple[Tuple[int, int, int], ...] = (
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    )


def build_style_centroid_bank(z_style: np.ndarray, genre_idx: np.ndarray, n_genres: int) -> torch.Tensor:
    bank = np.zeros((int(n_genres), z_style.shape[1]), dtype=np.float32)
    for g in range(int(n_genres)):
        mask = genre_idx == g
        if np.any(mask):
            v = z_style[mask].mean(axis=0).astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-8)
            bank[g] = v
    return torch.from_numpy(bank)


def build_style_exemplar_bank(z_style: np.ndarray, genre_idx: np.ndarray, n_genres: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for g in range(int(n_genres)):
        mask = genre_idx == g
        if np.any(mask):
            vals = z_style[mask].astype(np.float32)
            vals = vals / (np.linalg.norm(vals, axis=1, keepdims=True) + 1e-8)
            out[int(g)] = torch.from_numpy(vals)
    return out


def build_q_exemplar_bank(q_emb: np.ndarray, genre_idx: np.ndarray, n_genres: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for g in range(int(n_genres)):
        mask = genre_idx == g
        if np.any(mask):
            out[int(g)] = torch.from_numpy(q_emb[mask].astype(np.float32))
    return out


def _sample_shift_targets(source_idx: torch.Tensor, n_genres: int) -> torch.Tensor:
    n = int(source_idx.shape[0])
    tgt = torch.randint(low=0, high=int(n_genres), size=(n,), device=source_idx.device)
    clash = tgt == source_idx
    if clash.any():
        tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt


def _sample_shift_targets_balanced(source_idx: torch.Tensor, n_genres: int, offset: int = 0) -> torch.Tensor:
    n = int(source_idx.shape[0])
    base = (torch.arange(n, device=source_idx.device, dtype=torch.long) + int(offset)) % int(n_genres)
    tgt = base.clone()
    clash = tgt == source_idx
    if clash.any():
        tgt[clash] = (tgt[clash] + 1) % int(n_genres)
    return tgt


def _sample_bank_rows(bank: Dict[int, torch.Tensor], labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for lab in labels.detach().cpu().tolist():
        src = bank.get(int(lab))
        if src is None or src.shape[0] == 0:
            any_key = sorted(bank.keys())[0]
            src = bank[any_key]
        i = int(torch.randint(low=0, high=int(src.shape[0]), size=(1,)).item())
        rows.append(src[i : i + 1])
    return torch.cat(rows, dim=0).to(device)


def _mix_style_condition(
    target_idx: torch.Tensor,
    cond_mode: str,
    alpha: float,
    centroid_bank: torch.Tensor,
    exemplar_bank: Optional[Dict[int, torch.Tensor]],
    device: torch.device,
    exemplar_noise_std: float,
) -> torch.Tensor:
    z_cent = centroid_bank[target_idx].to(device)
    if exemplar_bank is None or len(exemplar_bank) == 0:
        z_ex = z_cent
    else:
        z_ex = _sample_bank_rows(exemplar_bank, target_idx, device=device)
    if exemplar_noise_std > 0.0:
        z_ex = z_ex + torch.randn_like(z_ex) * float(exemplar_noise_std)
    mode = str(cond_mode).strip().lower()
    if mode == "centroid":
        z = z_cent
    elif mode == "exemplar":
        z = z_ex
    else:
        a = float(alpha)
        z = a * z_cent + (1.0 - a) * z_ex
    return F.normalize(z, dim=-1)


def _waveform_mrstft_loss(
    wav_pred: torch.Tensor,
    wav_true: torch.Tensor,
    resolutions: Iterable[Tuple[int, int, int]],
) -> torch.Tensor:
    if wav_pred.shape != wav_true.shape:
        raise ValueError(f"Shape mismatch: {tuple(wav_pred.shape)} vs {tuple(wav_true.shape)}")
    x = wav_pred.squeeze(1)
    y = wav_true.squeeze(1)
    total = torch.tensor(0.0, device=x.device)
    used = 0
    for n_fft, hop, win in resolutions:
        n_fft = int(n_fft)
        hop = int(hop)
        win = int(win)
        if x.shape[-1] < n_fft:
            continue
        window = torch.hann_window(win, device=x.device)
        sx = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True, center=True)
        sy = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True, center=True)
        mx = torch.abs(sx).clamp_min(1e-8)
        my = torch.abs(sy).clamp_min(1e-8)
        sc = torch.norm(my - mx, p="fro") / (torch.norm(my, p="fro") + 1e-8)
        lm = F.l1_loss(torch.log(mx), torch.log(my))
        total = total + sc + lm
        used += 1
    if used == 0:
        return F.l1_loss(wav_pred, wav_true)
    return total / float(used)


class _Lab1WaveAdapter:
    def __init__(self, codec_sr: int, lab1_sr: int, n_frames: int, device: torch.device):
        self.codec_sr = int(codec_sr)
        self.lab1_sr = int(lab1_sr)
        self.n_frames = int(n_frames)
        self.device = device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(lab1_sr),
            n_fft=1024,
            hop_length=256,
            n_mels=96,
            f_min=20.0,
            f_max=float(lab1_sr) * 0.5,
            power=2.0,
            center=True,
        ).to(device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    def __call__(self, wav_b_1_t: torch.Tensor) -> torch.Tensor:
        y = wav_b_1_t.squeeze(1)
        if self.codec_sr != self.lab1_sr:
            y = torchaudio.functional.resample(
                waveform=y,
                orig_freq=int(self.codec_sr),
                new_freq=int(self.lab1_sr),
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_hann",
            )
        mel = self.mel(y).clamp_min(1e-8)
        mel_db = self.to_db(mel)
        t = int(mel_db.shape[-1])
        if t > self.n_frames:
            start = (t - self.n_frames) // 2
            mel_db = mel_db[:, :, start : start + self.n_frames]
        elif t < self.n_frames:
            pad = torch.full(
                (mel_db.shape[0], mel_db.shape[1], self.n_frames - t),
                fill_value=-80.0,
                device=mel_db.device,
                dtype=mel_db.dtype,
            )
            mel_db = torch.cat([mel_db, pad], dim=2)
        return mel_db


def _save_stage_checkpoint(
    ckpt_path: Path,
    epoch: int,
    generator: CodecLatentTranslator,
    discriminator: MultiScaleWaveDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    meta: Dict,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "meta": dict(meta),
    }
    torch.save(payload, str(ckpt_path))


def _try_load_stage_checkpoint(
    ckpt_path: Path,
    generator: CodecLatentTranslator,
    discriminator: MultiScaleWaveDiscriminator,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not ckpt_path.exists():
        return 0
    try:
        payload = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(str(ckpt_path), map_location=device)
    generator.load_state_dict(payload["generator"], strict=True)
    discriminator.load_state_dict(payload["discriminator"], strict=True)
    opt_g.load_state_dict(payload["opt_g"])
    opt_d.load_state_dict(payload["opt_d"])
    return int(payload.get("epoch", 0))


def train_codec_stage(
    stage_cfg: CodecStageTrainConfig,
    generator: CodecLatentTranslator,
    discriminator: MultiScaleWaveDiscriminator,
    codec: FrozenEncodec,
    lab1_encoder,
    style_judge: Optional[CodecStyleJudge],
    train_loader: DataLoader,
    n_genres: int,
    style_centroid_bank: torch.Tensor,
    style_exemplar_bank: Optional[Dict[int, torch.Tensor]],
    q_exemplar_bank: Dict[int, torch.Tensor],
    out_ckpt_dir: Path,
    device: torch.device,
    resume: bool = False,
    lab1_probe: Optional[Lab1StyleProbe] = None,
    mert_probe: Optional[MERTStyleProbe] = None,
) -> List[Dict]:
    generator.train()
    discriminator.train()
    style_centroid_bank = style_centroid_bank.to(device)
    out_ckpt_dir = Path(out_ckpt_dir)
    if style_judge is not None:
        style_judge.eval()
        for p in style_judge.parameters():
            p.requires_grad = False
    if lab1_probe is not None:
        lab1_probe.eval()
        for p in lab1_probe.parameters():
            p.requires_grad = False
    if mert_probe is not None:
        mert_probe.eval()
        for p in mert_probe.parameters():
            p.requires_grad = False

    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=float(stage_cfg.lr_g),
        betas=(float(stage_cfg.betas[0]), float(stage_cfg.betas[1])),
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=float(stage_cfg.lr_d),
        betas=(float(stage_cfg.betas[0]), float(stage_cfg.betas[1])),
    )

    ckpt_path = out_ckpt_dir / f"{stage_cfg.stage_name}_latest.pt"
    start_epoch = 0
    if resume:
        start_epoch = _try_load_stage_checkpoint(
            ckpt_path=ckpt_path,
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            device=device,
        )

    lab1_wave_adapter = _Lab1WaveAdapter(
        codec_sr=int(codec.cfg.sample_rate),
        lab1_sr=int(lab1_encoder.cfg.sample_rate),
        n_frames=256,
        device=device,
    )

    hist_rows: List[Dict] = []
    global_step = 0
    for epoch in range(start_epoch + 1, int(stage_cfg.epochs) + 1):
        alpha = float(stage_cfg.cond_alpha_start)
        if int(stage_cfg.epochs) > 1:
            ratio = float(epoch - 1) / float(max(1, int(stage_cfg.epochs) - 1))
            alpha = float(stage_cfg.cond_alpha_start) + (
                float(stage_cfg.cond_alpha_end) - float(stage_cfg.cond_alpha_start)
            ) * ratio

        m_loss_g = 0.0
        m_loss_d = 0.0
        m_lat_l1 = 0.0
        m_lat_cont = 0.0
        m_content = 0.0
        m_style = 0.0
        m_style_embed = 0.0
        m_mrstft = 0.0
        m_adv = 0.0
        m_fm = 0.0
        m_ms = 0.0
        m_style_push = 0.0
        m_delta_budget = 0.0
        d_steps_taken = 0
        d_steps_skipped = 0
        n_batches = 0

        for bi, batch in enumerate(train_loader):
            if stage_cfg.max_batches_per_epoch is not None and bi >= int(stage_cfg.max_batches_per_epoch):
                break
            global_step += 1
            n_batches += 1

            q_src = batch["q_emb"].to(device, non_blocking=True).float()
            zc_src = F.normalize(batch["z_content"].to(device, non_blocking=True).float(), dim=-1)
            zs_src = F.normalize(batch["z_style"].to(device, non_blocking=True).float(), dim=-1)
            src_genre_idx = batch["genre_idx"].to(device, non_blocking=True).long()

            if str(stage_cfg.stage_name) == "stage1":
                tgt_genre_idx = src_genre_idx
            else:
                if bool(stage_cfg.target_balance):
                    tgt_genre_idx = _sample_shift_targets_balanced(
                        src_genre_idx, n_genres=int(n_genres), offset=epoch + bi
                    )
                else:
                    tgt_genre_idx = _sample_shift_targets(src_genre_idx, n_genres=int(n_genres))

            z_style_tgt = _mix_style_condition(
                target_idx=tgt_genre_idx,
                cond_mode=str(stage_cfg.cond_mode),
                alpha=float(alpha),
                centroid_bank=style_centroid_bank,
                exemplar_bank=style_exemplar_bank,
                device=device,
                exemplar_noise_std=float(stage_cfg.exemplar_noise_std),
            )
            if str(stage_cfg.stage_name) == "stage1":
                if (
                    str(stage_cfg.style_loss_mode).strip().lower() == "lab1_cos"
                    and int(z_style_tgt.shape[1]) == int(zs_src.shape[1])
                ):
                    z_style_tgt = zs_src
                elif str(stage_cfg.style_loss_mode).strip().lower() == "lab1_probe_ce" and lab1_probe is not None:
                    with torch.no_grad():
                        z_style_tgt = lab1_probe.embed(zs_src).detach()
                elif str(stage_cfg.style_loss_mode).strip().lower() == "mert_probe_ce" and mert_probe is not None:
                    mf = batch.get("mert_feat")
                    if mf is not None:
                        with torch.no_grad():
                            z_style_tgt = mert_probe.embed(mf.to(device).float()).detach()

            noise = generator.sample_noise(batch_size=int(q_src.shape[0]), device=device)
            q_hat = generator(
                q_src=q_src,
                z_content=zc_src,
                z_style_tgt=z_style_tgt,
                noise=noise,
                style_dropout_p=float(stage_cfg.style_dropout_p),
                style_jitter_std=float(stage_cfg.style_jitter_std),
            )
            x_hat = codec.decode_embeddings(q_hat)
            with torch.no_grad():
                x_src = codec.decode_embeddings(q_src)

            # D step
            take_d_step = (global_step % max(1, int(stage_cfg.d_step_period))) == 0
            if take_d_step:
                q_real_tgt = _sample_bank_rows(q_exemplar_bank, tgt_genre_idx, device=device).float()
                with torch.no_grad():
                    x_real = codec.decode_embeddings(q_real_tgt)
                x_real = x_real.detach().requires_grad_(float(stage_cfg.r1_gamma) > 0.0)

                real_outs = discriminator(x_real)
                fake_outs_d = discriminator(x_hat.detach())
                loss_d = multiscale_hinge_d_loss(real_outs=real_outs, fake_outs=fake_outs_d)

                do_r1 = float(stage_cfg.r1_gamma) > 0.0 and (global_step % max(1, int(stage_cfg.r1_interval)) == 0)
                if do_r1:
                    real_sum = torch.tensor(0.0, device=device)
                    for real_logits, _ in real_outs:
                        real_sum = real_sum + real_logits.mean()
                    grad = torch.autograd.grad(
                        outputs=real_sum,
                        inputs=x_real,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    r1 = grad.reshape(grad.shape[0], -1).pow(2).sum(dim=1).mean()
                    loss_d = loss_d + 0.5 * float(stage_cfg.r1_gamma) * r1

                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if float(stage_cfg.d_grad_clip_norm) > 0.0:
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(stage_cfg.d_grad_clip_norm))
                opt_d.step()

                m_loss_d += float(loss_d.detach().cpu())
                d_steps_taken += 1
            else:
                d_steps_skipped += 1

            # G step
            q_real_tgt_g = _sample_bank_rows(q_exemplar_bank, tgt_genre_idx, device=device).float()
            with torch.no_grad():
                x_real_g = codec.decode_embeddings(q_real_tgt_g)
            real_outs_g = discriminator(x_real_g)
            fake_outs = discriminator(x_hat)

            loss_adv = multiscale_hinge_g_loss(fake_outs=fake_outs)
            loss_fm = multiscale_feature_matching_loss(real_outs=real_outs_g, fake_outs=fake_outs)
            loss_lat_l1 = F.l1_loss(q_hat, q_src)
            loss_lat_cont = F.l1_loss(q_hat[:, :, 1:] - q_hat[:, :, :-1], q_src[:, :, 1:] - q_src[:, :, :-1])
            loss_mrstft = _waveform_mrstft_loss(
                wav_pred=x_hat,
                wav_true=x_src,
                resolutions=stage_cfg.wave_mrstft_resolutions,
            )

            mel_hat = lab1_wave_adapter(x_hat)
            out_hat = lab1_encoder.forward_log_mel_tensor(mel_hat)
            zc_hat = F.normalize(out_hat["z_content"], dim=-1)
            loss_content = (1.0 - F.cosine_similarity(zc_hat, zc_src, dim=-1)).mean()
            loss_style = torch.tensor(0.0, device=device)
            loss_style_embed = torch.tensor(0.0, device=device)
            loss_style_push = torch.tensor(0.0, device=device)
            style_mode = str(stage_cfg.style_loss_mode).strip().lower()
            if style_mode == "codec_judge_ce" and style_judge is not None:
                emb_hat = style_judge.embed(q_hat)
                logits = style_judge.head(emb_hat)
                loss_style = F.cross_entropy(logits, tgt_genre_idx)
                if (
                    float(stage_cfg.style_embed_align_weight) > 0.0
                    and int(z_style_tgt.shape[1]) == int(emb_hat.shape[1])
                ):
                    z_ref = F.normalize(z_style_tgt, dim=-1)
                    loss_style_embed = (1.0 - F.cosine_similarity(emb_hat, z_ref, dim=-1)).mean()
                    loss_style = loss_style + float(stage_cfg.style_embed_align_weight) * loss_style_embed
                if str(stage_cfg.stage_name) != "stage1":
                    probs = torch.softmax(logits, dim=1)
                    p_src = probs.gather(1, src_genre_idx.view(-1, 1)).squeeze(1)
                    loss_style_push = F.relu(p_src - float(stage_cfg.style_push_margin)).mean()
            elif style_mode == "lab1_probe_ce" and lab1_probe is not None:
                # Route through Lab1 z_style → probe for CE loss (probe has well-separated embeddings)
                zs_hat = F.normalize(out_hat["z_style"], dim=-1)
                emb_hat = lab1_probe.embed(zs_hat)
                logits = lab1_probe.head(emb_hat)
                loss_style = F.cross_entropy(logits, tgt_genre_idx)
                if (
                    float(stage_cfg.style_embed_align_weight) > 0.0
                    and int(z_style_tgt.shape[1]) == int(emb_hat.shape[1])
                ):
                    z_ref = F.normalize(z_style_tgt, dim=-1)
                    loss_style_embed = (1.0 - F.cosine_similarity(emb_hat, z_ref, dim=-1)).mean()
                    loss_style = loss_style + float(stage_cfg.style_embed_align_weight) * loss_style_embed
                if str(stage_cfg.stage_name) != "stage1":
                    probs = torch.softmax(logits, dim=1)
                    p_src = probs.gather(1, src_genre_idx.view(-1, 1)).squeeze(1)
                    loss_style_push = F.relu(p_src - float(stage_cfg.style_push_margin)).mean()
            elif style_mode == "mert_probe_ce" and mert_probe is not None:
                # MERT probe: use cached mert_feat from batch → frozen probe → CE loss
                mf = batch.get("mert_feat")
                if mf is not None:
                    mf = mf.to(device).float()
                    # We need mert_feat of the *translated* audio, but we only have source mert_feat.
                    # Use the codec judge on q_hat for the actual gradient signal (same as codec_judge_ce),
                    # but also use the MERT probe embeddings as conditioning.
                    # For style loss: re-encode q_hat through codec→waveform→MERT is too expensive.
                    # Instead, use the frozen codec judge on q_hat for CE loss (it's always trained),
                    # and let the MERT probe provide better-separated FiLM conditioning.
                    pass  # fall through to codec_judge_ce path for loss if judge available

                # Use codec judge for the actual differentiable loss on q_hat
                if style_judge is not None:
                    emb_hat = style_judge.embed(q_hat)
                    logits = style_judge.head(emb_hat)
                    loss_style = F.cross_entropy(logits, tgt_genre_idx)
                    if (
                        float(stage_cfg.style_embed_align_weight) > 0.0
                        and int(z_style_tgt.shape[1]) == int(emb_hat.shape[1])
                    ):
                        z_ref = F.normalize(z_style_tgt, dim=-1)
                        loss_style_embed = (1.0 - F.cosine_similarity(emb_hat, z_ref, dim=-1)).mean()
                        loss_style = loss_style + float(stage_cfg.style_embed_align_weight) * loss_style_embed
                    if str(stage_cfg.stage_name) != "stage1":
                        probs = torch.softmax(logits, dim=1)
                        p_src = probs.gather(1, src_genre_idx.view(-1, 1)).squeeze(1)
                        loss_style_push = F.relu(p_src - float(stage_cfg.style_push_margin)).mean()
                else:
                    # Fallback: cosine loss on Lab1 z_style
                    zs_hat = F.normalize(out_hat["z_style"], dim=-1)
                    loss_style = (1.0 - F.cosine_similarity(zs_hat, z_style_tgt, dim=-1)).mean()
            else:
                zs_hat = F.normalize(out_hat["z_style"], dim=-1)
                loss_style = (1.0 - F.cosine_similarity(zs_hat, z_style_tgt, dim=-1)).mean()
                if str(stage_cfg.stage_name) != "stage1":
                    cos_to_src_style = F.cosine_similarity(zs_hat, zs_src, dim=-1)
                    loss_style_push = F.relu(cos_to_src_style - float(stage_cfg.style_push_margin)).mean()

            delta_mean = (q_hat - q_src).abs().mean(dim=(1, 2))
            loss_delta_budget = F.relu(delta_mean - float(stage_cfg.delta_budget)).mean()

            loss_ms = torch.tensor(0.0, device=device)
            if float(stage_cfg.weights.mode_seeking) > 0.0:
                noise2 = generator.sample_noise(batch_size=int(q_src.shape[0]), device=device)
                noise2 = noise2 * float(stage_cfg.mode_seeking_noise_scale)
                q_hat2 = generator(
                    q_src=q_src,
                    z_content=zc_src,
                    z_style_tgt=z_style_tgt,
                    noise=noise2,
                    style_dropout_p=float(stage_cfg.style_dropout_p),
                    style_jitter_std=float(stage_cfg.style_jitter_std),
                )
                # Use latent-space mode seeking to avoid a second expensive waveform decode.
                num = torch.mean(torch.abs(q_hat - q_hat2))
                den = torch.mean(torch.abs(noise - noise2)).clamp_min(1e-5)
                ratio = num / den
                loss_ms = F.relu(torch.tensor(float(stage_cfg.mode_seeking_target), device=device) - ratio)

            w = stage_cfg.weights
            loss_g = (
                float(w.adv) * loss_adv
                + float(w.feature_match) * loss_fm
                + float(w.latent_l1) * loss_lat_l1
                + float(w.latent_continuity) * loss_lat_cont
                + float(w.mrstft) * loss_mrstft
                + float(w.content) * loss_content
                + float(w.style) * loss_style
                + float(w.mode_seeking) * loss_ms
                + float(w.style_push) * loss_style_push
                + float(w.delta_budget) * loss_delta_budget
            )

            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            if float(stage_cfg.g_grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), float(stage_cfg.g_grad_clip_norm))
            opt_g.step()

            m_loss_g += float(loss_g.detach().cpu())
            m_lat_l1 += float(loss_lat_l1.detach().cpu())
            m_lat_cont += float(loss_lat_cont.detach().cpu())
            m_content += float(loss_content.detach().cpu())
            m_style += float(loss_style.detach().cpu())
            m_style_embed += float(loss_style_embed.detach().cpu())
            m_mrstft += float(loss_mrstft.detach().cpu())
            m_adv += float(loss_adv.detach().cpu())
            m_fm += float(loss_fm.detach().cpu())
            m_ms += float(loss_ms.detach().cpu())
            m_style_push += float(loss_style_push.detach().cpu())
            m_delta_budget += float(loss_delta_budget.detach().cpu())

        n = max(1, n_batches)
        row = {
            "stage": str(stage_cfg.stage_name),
            "epoch": int(epoch),
            "loss_g": m_loss_g / n,
            "loss_d": (m_loss_d / max(1, d_steps_taken)) if d_steps_taken > 0 else 0.0,
            "loss_adv": m_adv / n,
            "loss_feature_match": m_fm / n,
            "loss_latent_l1": m_lat_l1 / n,
            "loss_continuity": m_lat_cont / n,
            "loss_content": m_content / n,
            "loss_style": m_style / n,
            "loss_style_embed": m_style_embed / n,
            "loss_style_push": m_style_push / n,
            "loss_delta_budget": m_delta_budget / n,
            "loss_mrstft": m_mrstft / n,
            "loss_mode_seeking": m_ms / n,
            "d_steps_taken": int(d_steps_taken),
            "d_steps_skipped": int(d_steps_skipped),
            "d_step_rate": float(d_steps_taken) / float(max(1, d_steps_taken + d_steps_skipped)),
            "cond_alpha": float(alpha),
        }
        print(
            "[codec-train]"
            f" stage={row['stage']}"
            f" epoch={row['epoch']}"
            f" loss_g={row['loss_g']:.4f}"
            f" loss_d={row['loss_d']:.4f}"
            f" content={row['loss_content']:.4f}"
            f" style={row['loss_style']:.4f}"
            f" d_rate={row['d_step_rate']:.2f}"
        )
        hist_rows.append(row)
        _save_stage_checkpoint(
            ckpt_path=ckpt_path,
            epoch=int(epoch),
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            meta={"stage": str(stage_cfg.stage_name), "config": stage_cfg.__dict__},
        )
    return hist_rows
