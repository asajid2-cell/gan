from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import EncodecModel


REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB33_SCRIPTS = REPO_ROOT / "lab 3.3" / "scripts"
if str(LAB33_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB33_SCRIPTS))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_codec_bridge import FrozenEncodec
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import DIFFUSION_SR, extract_beat_grid, extract_chroma, extract_onset, load_diffusion_cache, pad_or_trim
from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _json_default, _resolve_stems, picked_songs
from train_scratch_retrieval_fusion import (
    _build_track_bank,
    _choose_donor_track,
    _device_from_arg,
    _judge_probs_for_audio,
    _load_or_train_judge,
    _mix_preserved_vocals,
    _slug,
    _write_json,
)
from train_scratch_structure_diffusion import _audio_metrics, _make_balanced_sampler


def _resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if int(sr_from) == int(sr_to):
        return np.asarray(y, dtype=np.float32)
    return librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=int(sr_from), target_sr=int(sr_to), res_type="soxr_hq").astype(np.float32)


def _source_hint_from_qemb(
    q_emb: torch.Tensor,
    *,
    low_keep: float = 0.75,
    mid_keep: float = 0.45,
    high_keep: float = 0.20,
) -> torch.Tensor:
    x = F.avg_pool1d(q_emb.unsqueeze(0), kernel_size=9, stride=1, padding=4)[0]
    x[:32, :] = float(low_keep) * q_emb[:32, :] + (1.0 - float(low_keep)) * x[:32, :]
    x[32:96, :] = float(mid_keep) * q_emb[32:96, :] + (1.0 - float(mid_keep)) * x[32:96, :]
    if x.shape[0] > 96:
        hf_mean = x[96:, :].mean(dim=0, keepdim=True)
        x[96:, :] = float(high_keep) * q_emb[96:, :] + (1.0 - float(high_keep)) * hf_mean
    return x


def _load_encodec_chunk(path: Path, *, start_sec: float, seconds: float, sr: int) -> np.ndarray:
    return load_audio_chunk(Path(path), sample_rate=int(sr), seconds=float(seconds), start_sec=float(start_sec)).astype(np.float32)


def _make_mel_transform(sr: int, device: torch.device) -> nn.Module:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=int(sr),
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        n_mels=80,
        f_min=20.0,
        f_max=11000.0,
        power=2.0,
    ).to(device)


def _wav_to_logmel(y: torch.Tensor, mel_transform: nn.Module) -> torch.Tensor:
    return torch.log(mel_transform(y).clamp_min(1e-5))


class EncodecFusionDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        arrays: Dict[str, np.ndarray],
        indices: Sequence[int],
        *,
        codec: FrozenEncodec,
        cond_frames: int,
        seed: int,
        source_hint_low_keep: float,
        source_hint_mid_keep: float,
        source_hint_high_keep: float,
    ) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.codec = codec
        self.cond_frames = int(cond_frames)
        self.rng = np.random.default_rng(int(seed))
        self.source_hint_low_keep = float(source_hint_low_keep)
        self.source_hint_mid_keep = float(source_hint_mid_keep)
        self.source_hint_high_keep = float(source_hint_high_keep)
        self.genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
        self.track_ids = self.index_df["track_id"].astype(str).to_numpy()
        self.by_genre: Dict[int, np.ndarray] = {}
        for g in np.unique(self.genre_idx[self.indices]).tolist():
            self.by_genre[int(g)] = self.indices[self.genre_idx[self.indices] == int(g)]

    def __len__(self) -> int:
        return int(len(self.indices))

    def genre_indices(self) -> np.ndarray:
        return self.genre_idx[self.indices].astype(np.int64)

    def _pick_donor(self, idx: int, genre: int) -> int:
        pool = self.by_genre[int(genre)]
        if len(pool) <= 1:
            return int(idx)
        src_track = self.track_ids[int(idx)]
        candidates = [int(x) for x in pool.tolist() if self.track_ids[int(x)] != src_track]
        if not candidates:
            candidates = [int(x) for x in pool.tolist() if int(x) != int(idx)]
        if not candidates:
            return int(idx)
        return int(candidates[int(self.rng.integers(0, len(candidates)))])

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[int(item)])
        genre = int(self.genre_idx[idx])
        donor_idx = self._pick_donor(idx, genre)
        row = self.index_df.iloc[idx]
        donor_row = self.index_df.iloc[donor_idx]
        target_y = _load_encodec_chunk(Path(str(row["path"])), start_sec=float(row["start_sec"]), seconds=float(self.codec.cfg.chunk_seconds), sr=int(self.codec.cfg.sample_rate))
        donor_y = _load_encodec_chunk(Path(str(donor_row["path"])), start_sec=float(donor_row["start_sec"]), seconds=float(self.codec.cfg.chunk_seconds), sr=int(self.codec.cfg.sample_rate))
        target_q, _ = self.codec.encode_chunk_embeddings(target_y)
        donor_q, _ = self.codec.encode_chunk_embeddings(donor_y)
        target_q = self.codec.fix_num_frames(target_q).float()
        donor_q = self.codec.fix_num_frames(donor_q).float()
        source_hint = _source_hint_from_qemb(
            target_q,
            low_keep=self.source_hint_low_keep,
            mid_keep=self.source_hint_mid_keep,
            high_keep=self.source_hint_high_keep,
        ).float()
        chroma = pad_or_trim(np.asarray(self.arrays["chroma"][idx], dtype=np.float32), self.cond_frames, axis=1, pad_val=0.0)
        onset = pad_or_trim(np.asarray(self.arrays["onset"][idx], dtype=np.float32), self.cond_frames, axis=0, pad_val=0.0)
        beat = pad_or_trim(np.asarray(self.arrays["beat"][idx], dtype=np.float32), self.cond_frames, axis=0, pad_val=0.0)
        cond = np.concatenate([chroma, onset[None, :], beat[None, :]], axis=0).astype(np.float32)
        return {
            "target_q": target_q,
            "donor_q": donor_q,
            "source_hint": source_hint,
            "cond": torch.from_numpy(cond),
            "genre_idx": torch.tensor(genre, dtype=torch.long),
            "target_audio": torch.from_numpy(target_y),
            "donor_audio": torch.from_numpy(donor_y),
        }


class EncodecLatentFusionNet(nn.Module):
    def __init__(
        self,
        latent_ch: int,
        cond_ch: int,
        num_genres: int,
        base_ch: int = 192,
        proposal_scale: float = 0.65,
        source_skip_mix: float = 1.0,
    ):
        super().__init__()
        self.proposal_scale = float(proposal_scale)
        self.source_skip_mix = float(source_skip_mix)
        self.genre_emb = nn.Embedding(num_genres, 64)
        self.cond_proj = nn.Sequential(
            nn.Conv1d(cond_ch, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.in_proj = nn.Conv1d(latent_ch * 3 + 64, base_ch, 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(base_ch, base_ch, 3, padding=1), nn.GroupNorm(8, base_ch), nn.SiLU())
                for _ in range(4)
            ]
        )
        self.film = nn.Linear(64, base_ch * 2)
        self.out = nn.Conv1d(base_ch, latent_ch * 2, 3, padding=1)

    def forward(self, source_hint: torch.Tensor, donor_q: torch.Tensor, cond: torch.Tensor, genre_idx: torch.Tensor) -> torch.Tensor:
        t_len = int(source_hint.shape[-1])
        cond_feat = self.cond_proj(F.interpolate(cond, size=t_len, mode="linear", align_corners=False))
        h = self.in_proj(torch.cat([source_hint, donor_q, donor_q - source_hint, cond_feat], dim=1))
        gamma, beta = self.film(self.genre_emb(genre_idx)).chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * gamma[:, :, None]) + 0.1 * beta[:, :, None]
        for block in self.blocks:
            h = h + block(h)
        residual, gate_logits = self.out(h).chunk(2, dim=1)
        proposal = torch.tanh(source_hint + self.proposal_scale * torch.tanh(residual))
        gate = torch.sigmoid(gate_logits)
        skip = self.source_skip_mix * source_hint + (1.0 - self.source_skip_mix) * donor_q
        return gate * proposal + (1.0 - gate) * skip


@torch.no_grad()
def _decode_q(codec_model: EncodecModel, q_emb: torch.Tensor) -> torch.Tensor:
    was_training = bool(codec_model.decoder.training)
    if not was_training:
        codec_model.decoder.train(True)
    out = codec_model.decoder(q_emb)
    if not was_training:
        codec_model.decoder.train(False)
    return out


def _build_inference_donor_rows(index_df: pd.DataFrame, donor_track: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in donor_track["rows"]:
        row = index_df.iloc[int(idx)]
        rows.append({"path": str(row["path"]), "start_sec": float(row["start_sec"]), "idx": int(idx)})
    return rows


@torch.no_grad()
def generate_longform_encodec_from_rows(
    model: EncodecLatentFusionNet,
    *,
    codec_model: EncodecModel,
    source_audio_22k: np.ndarray,
    target_genre_idx: int,
    donor_rows: List[Dict[str, Any]],
    codec: FrozenEncodec,
    cond_frames: int,
    device: torch.device,
    source_hint_low_keep: float,
    source_hint_mid_keep: float,
    source_hint_high_keep: float,
) -> np.ndarray:
    from train_scratch_structure_diffusion import assemble_audio_crossfade, split_audio_overlapping

    model.eval()
    source_audio = _resample(source_audio_22k, DIFFUSION_SR, int(codec.cfg.sample_rate))
    chunks = split_audio_overlapping(source_audio, chunk_seconds=3.0, overlap_seconds=0.5, sr=int(codec.cfg.sample_rate))
    total = max(1, len(chunks) - 1)
    out_wavs: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        y = np.asarray(chunk["audio"], dtype=np.float32)
        q_src, _ = codec.encode_chunk_embeddings(y)
        q_src = codec.fix_num_frames(q_src).float()
        source_hint = _source_hint_from_qemb(
            q_src,
            low_keep=source_hint_low_keep,
            mid_keep=source_hint_mid_keep,
            high_keep=source_hint_high_keep,
        ).unsqueeze(0).to(device)
        donor_row = donor_rows[min(len(donor_rows) - 1, round((i / total) * max(0, len(donor_rows) - 1)))]
        donor_y = _load_encodec_chunk(Path(str(donor_row["path"])), start_sec=float(donor_row["start_sec"]), seconds=float(codec.cfg.chunk_seconds), sr=int(codec.cfg.sample_rate))
        donor_q, _ = codec.encode_chunk_embeddings(donor_y)
        donor_q = codec.fix_num_frames(donor_q).float().unsqueeze(0).to(device)
        chroma = pad_or_trim(extract_chroma(y, sr=int(codec.cfg.sample_rate)), cond_frames, axis=1, pad_val=0.0)
        onset = pad_or_trim(extract_onset(y, sr=int(codec.cfg.sample_rate)), cond_frames, axis=0, pad_val=0.0)
        beat = pad_or_trim(extract_beat_grid(y, sr=int(codec.cfg.sample_rate), n_frames=cond_frames), cond_frames, axis=0, pad_val=0.0)
        cond = torch.from_numpy(np.concatenate([chroma, onset[None, :], beat[None, :]], axis=0).astype(np.float32)).unsqueeze(0).to(device)
        pred_q = model(source_hint, donor_q, cond, torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device))
        wav = _decode_q(codec_model, pred_q)[0, 0].detach().cpu().numpy().astype(np.float32)
        out_wavs.append(wav)
    audio_24k = assemble_audio_crossfade(out_wavs, overlap_seconds=0.5, sr=int(codec.cfg.sample_rate))
    return _resample(audio_24k, int(codec.cfg.sample_rate), DIFFUSION_SR)


def benchmark_checkpoint(
    *,
    model: EncodecLatentFusionNet,
    codec_model: EncodecModel,
    codec: FrozenEncodec,
    judge: Any,
    genre_to_idx: Dict[str, int],
    track_bank: Dict[int, List[Dict[str, Any]]],
    index_df: pd.DataFrame,
    cond_frames: int,
    device: torch.device,
    seconds: float,
    out_dir: Path,
    single_genre_target: str = "",
    source_hint_low_keep: float = 0.75,
    source_hint_mid_keep: float = 0.45,
    source_hint_high_keep: float = 0.20,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_cfg = HybridPushConfig()
    manifest_rows: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    separation_vals: List[float] = []
    for song in picked_songs():
        stems = _resolve_stems(hybrid_cfg, song)
        source_key = _slug(Path(song["path"]).stem)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
        per_song: List[Dict[str, Any]] = []
        targets = [str(single_genre_target)] if str(single_genre_target).strip() else list(TARGET_GENRES)
        for target in targets:
            target_idx = int(genre_to_idx[target])
            donor_track = _choose_donor_track(source_acc, track_bank, target_idx)
            donor_rows = _build_inference_donor_rows(index_df, donor_track)
            render_dir = out_dir / "renders" / source_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp = generate_longform_encodec_from_rows(
                model,
                codec_model=codec_model,
                source_audio_22k=source_acc,
                target_genre_idx=target_idx,
                donor_rows=donor_rows,
                codec=codec,
                cond_frames=cond_frames,
                device=device,
                source_hint_low_keep=source_hint_low_keep,
                source_hint_mid_keep=source_hint_mid_keep,
                source_hint_high_keep=source_hint_high_keep,
            )
            accomp = accomp[: len(source_acc)].astype(np.float32)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), accomp, DIFFUSION_SR)
            final_mix = _mix_preserved_vocals(stems["vocals"], accomp, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = _judge_probs_for_audio(accomp, judge, device, 256)
            metrics = _audio_metrics(source_acc, accomp, DIFFUSION_SR)
            tgt_conf = float(probs[target_idx])
            tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
            row = {
                "song": source_key,
                "target": target,
                "target_conf": tgt_conf,
                "target_margin": tgt_margin,
                "warble": float(metrics["warble"]),
                "fullness": float(metrics["fullness"]),
                "structure": float(metrics["structure"]),
                "hybrid_wav": str(final_mix),
                "accompaniment_wav": str(accomp_path),
                "donor_track_id": donor_track["track_id"],
                "judge_probs": probs.tolist(),
            }
            rows.append(row)
            per_song.append(row)
            manifest_rows.append(
                {
                    "target": target,
                    "source_song": source_key,
                    "source_target_dir": str(render_dir),
                    "hybrid_wav": str(final_mix),
                    "accompaniment_wav": str(accomp_path),
                }
            )
        for i in range(len(per_song)):
            for j in range(i + 1, len(per_song)):
                pa = np.asarray(per_song[i]["judge_probs"], dtype=np.float32)
                pb = np.asarray(per_song[j]["judge_probs"], dtype=np.float32)
                separation_vals.append(float(np.mean(np.abs(pa - pb))))
    mean_sep = float(np.mean(separation_vals)) if separation_vals else 0.0
    for row in rows:
        row["separation"] = mean_sep
        row["overall"] = float(0.45 * row["target_margin"] + 0.20 * row["target_conf"] + 0.18 * row["fullness"] + 0.18 * row["structure"] + 0.20 * row["separation"] - 0.18 * row["warble"])
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "mean_overall": float(np.mean([r["overall"] for r in rows])) if rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in rows])) if rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in rows])) if rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in rows])) if rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in rows])) if rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in rows])) if rows else 0.0,
        "mean_separation": mean_sep,
        "rows": rows,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


@dataclass
class TrainConfig:
    cache_dir: Path = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    judge_ckpt: Path = Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt"
    out_root: Path = Path.home() / "Desktop" / "dggr_pretrained_encodec_fusion_runs"
    model_id: str = "facebook/encodec_24khz"
    bandwidth: float = 6.0
    chunk_seconds: float = 5.0
    epochs: int = 2
    batch_size: int = 4
    max_batches_per_epoch: int = 160
    val_batches: int = 8
    eval_every_steps: int = 60
    lr_model: float = 2e-4
    lr_decoder: float = 5e-6
    weight_decay: float = 1e-4
    base_ch: int = 192
    source_hint_low_keep: float = 0.75
    source_hint_mid_keep: float = 0.45
    source_hint_high_keep: float = 0.20
    proposal_scale: float = 0.65
    source_skip_mix: float = 1.0
    loss_lat_weight: float = 1.0
    loss_dt_weight: float = 0.25
    loss_mel_weight: float = 0.30
    loss_low_weight: float = 0.25
    loss_timbre_weight: float = 0.12
    loss_donor_band_weight: float = 0.0
    seed: int = 328
    benchmark_seconds: float = 30.0
    final_seconds: float = 60.0
    device: str = "auto"
    single_genre_target: str = ""


def train(cfg: TrainConfig) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = _device_from_arg(str(cfg.device))
    out_dir = Path(cfg.out_root) / f"pretrained_encodec_fusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(cfg))

    codec = FrozenEncodec(model_id=str(cfg.model_id), bandwidth=float(cfg.bandwidth), chunk_seconds=float(cfg.chunk_seconds), device=str(cfg.device))
    codec_model = EncodecModel.from_pretrained(str(cfg.model_id), local_files_only=True).to(device)
    codec_model.train()
    for p in codec_model.encoder.parameters():
        p.requires_grad = False
    for p in codec_model.quantizer.parameters():
        p.requires_grad = False

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=int(cfg.seed))
    if str(cfg.single_genre_target).strip():
        keep = int(genre_to_idx[str(cfg.single_genre_target)])
        train_idx = train_idx[genre_idx[train_idx] == keep]
        val_idx = val_idx[genre_idx[val_idx] == keep]
        if len(val_idx) == 0:
            val_idx = train_idx[: min(len(train_idx), 64)]

    cond_frames = int(round(float(cfg.chunk_seconds) * 75.0))
    train_ds = EncodecFusionDataset(
        index_df,
        arrays,
        train_idx,
        codec=codec,
        cond_frames=cond_frames,
        seed=int(cfg.seed),
        source_hint_low_keep=float(cfg.source_hint_low_keep),
        source_hint_mid_keep=float(cfg.source_hint_mid_keep),
        source_hint_high_keep=float(cfg.source_hint_high_keep),
    )
    val_ds = EncodecFusionDataset(
        index_df,
        arrays,
        val_idx,
        codec=codec,
        cond_frames=cond_frames,
        seed=int(cfg.seed + 1),
        source_hint_low_keep=float(cfg.source_hint_low_keep),
        source_hint_mid_keep=float(cfg.source_hint_mid_keep),
        source_hint_high_keep=float(cfg.source_hint_high_keep),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=_make_balanced_sampler(train_ds, int(cfg.seed)), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    judge, judge_genre_to_idx = _load_or_train_judge(Path(cfg.judge_ckpt), Path(cfg.cache_dir), out_dir, device, 256)
    if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
        raise RuntimeError("Judge genre mismatch")

    model = EncodecLatentFusionNet(
        latent_ch=int(codec.cfg.latent_channels),
        cond_ch=14,
        num_genres=len(genre_to_idx),
        base_ch=int(cfg.base_ch),
        proposal_scale=float(cfg.proposal_scale),
        source_skip_mix=float(cfg.source_skip_mix),
    ).to(device)
    mel_transform = _make_mel_transform(int(codec.cfg.sample_rate), device)
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": float(cfg.lr_model)},
            {"params": codec_model.decoder.parameters(), "lr": float(cfg.lr_decoder)},
        ],
        weight_decay=float(cfg.weight_decay),
    )
    track_bank = _build_track_bank(index_df, arrays, train_idx)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1e18
    best_ckpt = ckpt_dir / "best_by_judge.pt"
    history: List[Dict[str, Any]] = []
    global_step = 0

    def _save_ckpt(path: Path, epoch: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "codec_decoder": codec_model.decoder.state_dict(),
                "cfg": asdict(cfg),
                "genre_to_idx": genre_to_idx,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "codec_cfg": asdict(codec.cfg),
            },
            str(path),
        )

    def _run_eval(tag: str, epoch: int) -> Dict[str, Any]:
        nonlocal best_score
        bench = benchmark_checkpoint(
            model=model,
            codec_model=codec_model,
            codec=codec,
            judge=judge,
            genre_to_idx=genre_to_idx,
            track_bank=track_bank,
            index_df=index_df,
            cond_frames=cond_frames,
            device=device,
            seconds=float(cfg.benchmark_seconds),
            out_dir=out_dir / "benchmark" / tag,
            single_genre_target=str(cfg.single_genre_target),
            source_hint_low_keep=float(cfg.source_hint_low_keep),
            source_hint_mid_keep=float(cfg.source_hint_mid_keep),
            source_hint_high_keep=float(cfg.source_hint_high_keep),
        )
        if float(bench["mean_overall"]) > float(best_score):
            best_score = float(bench["mean_overall"])
            _save_ckpt(best_ckpt, epoch)
            _write_json(out_dir / "winner_map.json", {"best_tag": tag, "best_checkpoint": str(best_ckpt), "best_score": float(best_score)})
        return bench

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        codec_model.decoder.train(True)
        start_t = time.time()
        train_lat: List[float] = []
        train_mel: List[float] = []
        train_low: List[float] = []
        train_timbre: List[float] = []
        for batch_idx, batch in enumerate(train_loader):
            if int(cfg.max_batches_per_epoch) > 0 and batch_idx >= int(cfg.max_batches_per_epoch):
                break
            target_q = batch["target_q"].to(device)
            donor_q = batch["donor_q"].to(device)
            source_hint = batch["source_hint"].to(device)
            cond = batch["cond"].to(device)
            genre = batch["genre_idx"].to(device)
            target_audio = batch["target_audio"].to(device).unsqueeze(1)
            donor_audio = batch["donor_audio"].to(device).unsqueeze(1)
            pred_q = model(source_hint, donor_q, cond, genre)
            pred_audio = codec_model.decoder(pred_q)
            pred_mel = _wav_to_logmel(pred_audio.squeeze(1), mel_transform)
            target_mel = _wav_to_logmel(target_audio.squeeze(1), mel_transform)
            donor_mel = _wav_to_logmel(donor_audio.squeeze(1), mel_transform)
            loss_lat = F.l1_loss(pred_q, target_q)
            loss_dt = F.l1_loss(pred_q[:, :, 1:] - pred_q[:, :, :-1], target_q[:, :, 1:] - target_q[:, :, :-1])
            loss_mel = F.l1_loss(pred_mel, target_mel)
            loss_low = F.l1_loss(pred_mel[:, :28, :], target_mel[:, :28, :])
            loss_timbre = F.l1_loss(torch.cat([pred_mel.mean(dim=-1), pred_mel.std(dim=-1)], dim=1), torch.cat([donor_mel.mean(dim=-1), donor_mel.std(dim=-1)], dim=1))
            loss_donor_band = F.l1_loss(pred_mel[:, 28:, :].mean(dim=-1), donor_mel[:, 28:, :].mean(dim=-1))
            loss = (
                float(cfg.loss_lat_weight) * loss_lat
                + float(cfg.loss_dt_weight) * loss_dt
                + float(cfg.loss_mel_weight) * loss_mel
                + float(cfg.loss_low_weight) * loss_low
                + float(cfg.loss_timbre_weight) * loss_timbre
                + float(cfg.loss_donor_band_weight) * loss_donor_band
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(codec_model.decoder.parameters()), 1.0)
            opt.step()
            train_lat.append(float(loss_lat.item()))
            train_mel.append(float(loss_mel.item()))
            train_low.append(float(loss_low.item()))
            train_timbre.append(float(loss_timbre.item()))
            global_step += 1
            if int(cfg.eval_every_steps) > 0 and global_step % int(cfg.eval_every_steps) == 0:
                _save_ckpt(ckpt_dir / "latest.pt", epoch)
                bench = _run_eval(f"step_{global_step:05d}", epoch)
                history.append(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "tag": f"step_{global_step:05d}",
                        "train_lat": float(np.mean(train_lat)) if train_lat else 0.0,
                        "train_mel": float(np.mean(train_mel)) if train_mel else 0.0,
                        "train_low": float(np.mean(train_low)) if train_low else 0.0,
                        "train_timbre": float(np.mean(train_timbre)) if train_timbre else 0.0,
                        "benchmark_overall": float(bench["mean_overall"]),
                        "benchmark_target_conf": float(bench["mean_target_conf"]),
                        "benchmark_target_margin": float(bench["mean_target_margin"]),
                        "benchmark_warble": float(bench["mean_warble"]),
                        "benchmark_fullness": float(bench["mean_fullness"]),
                        "benchmark_structure": float(bench["mean_structure"]),
                        "benchmark_separation": float(bench["mean_separation"]),
                    }
                )
                _write_json(out_dir / "history.json", {"rows": history})

        model.eval()
        codec_model.decoder.eval()
        val_lat: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= int(cfg.val_batches):
                    break
                pred_q = model(batch["source_hint"].to(device), batch["donor_q"].to(device), batch["cond"].to(device), batch["genre_idx"].to(device))
                val_lat.append(float(F.l1_loss(pred_q, batch["target_q"].to(device)).item()))
        _save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch)
        _save_ckpt(ckpt_dir / "latest.pt", epoch)
        bench = _run_eval(f"epoch_{epoch:03d}", epoch)
        history.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "tag": f"epoch_{epoch:03d}",
                "train_lat": float(np.mean(train_lat)) if train_lat else 0.0,
                "train_mel": float(np.mean(train_mel)) if train_mel else 0.0,
                "train_low": float(np.mean(train_low)) if train_low else 0.0,
                "train_timbre": float(np.mean(train_timbre)) if train_timbre else 0.0,
                "val_lat": float(np.mean(val_lat)) if val_lat else 0.0,
                "benchmark_overall": float(bench["mean_overall"]),
                "benchmark_target_conf": float(bench["mean_target_conf"]),
                "benchmark_target_margin": float(bench["mean_target_margin"]),
                "benchmark_warble": float(bench["mean_warble"]),
                "benchmark_fullness": float(bench["mean_fullness"]),
                "benchmark_structure": float(bench["mean_structure"]),
                "benchmark_separation": float(bench["mean_separation"]),
                "epoch_seconds": float(time.time() - start_t),
            }
        )
        _write_json(out_dir / "history.json", {"rows": history})

    payload = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    codec_model.decoder.load_state_dict(payload["codec_decoder"])
    final_pack = out_dir / "final_pack"
    final_summary = benchmark_checkpoint(
        model=model,
        codec_model=codec_model,
        codec=codec,
        judge=judge,
        genre_to_idx=genre_to_idx,
        track_bank=track_bank,
        index_df=index_df,
        cond_frames=cond_frames,
        device=device,
        seconds=float(cfg.final_seconds),
        out_dir=final_pack,
        single_genre_target=str(cfg.single_genre_target),
        source_hint_low_keep=float(cfg.source_hint_low_keep),
        source_hint_mid_keep=float(cfg.source_hint_mid_keep),
        source_hint_high_keep=float(cfg.source_hint_high_keep),
    )
    (out_dir / "diagnosis_report.md").write_text(
        "\n".join(
            [
                "# Pretrained Encodec Fusion Diagnosis",
                "",
                "- New strategy: pretrained Encodec decoder fine-tuning plus a new latent fusion model.",
                f"- Best checkpoint: {best_ckpt}",
                f"- Final mean overall: {final_summary['mean_overall']:.4f}",
                f"- Final mean target confidence: {final_summary['mean_target_conf']:.4f}",
                f"- Final mean target margin: {final_summary['mean_target_margin']:.4f}",
                f"- Final mean fullness: {final_summary['mean_fullness']:.4f}",
                f"- Final mean structure: {final_summary['mean_structure']:.4f}",
                f"- Final mean warble: {final_summary['mean_warble']:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    summary = {"out_dir": str(out_dir), "best_checkpoint": str(best_ckpt), "final_pack_dir": str(final_pack), "history_rows": history, "final_summary": final_summary}
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a pretrained-Encodec fusion accompaniment generator and benchmark it.")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_pretrained_encodec_fusion_runs")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--judge-ckpt", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-batches-per-epoch", type=int, default=160)
    ap.add_argument("--eval-every-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--single-genre-target", type=str, default="")
    ap.add_argument("--benchmark-seconds", type=float, default=30.0)
    ap.add_argument("--final-seconds", type=float, default=60.0)
    ap.add_argument("--lr-model", type=float, default=2e-4)
    ap.add_argument("--lr-decoder", type=float, default=5e-6)
    ap.add_argument("--source-hint-low-keep", type=float, default=0.75)
    ap.add_argument("--source-hint-mid-keep", type=float, default=0.45)
    ap.add_argument("--source-hint-high-keep", type=float, default=0.20)
    ap.add_argument("--proposal-scale", type=float, default=0.65)
    ap.add_argument("--source-skip-mix", type=float, default=1.0)
    ap.add_argument("--loss-lat-weight", type=float, default=1.0)
    ap.add_argument("--loss-dt-weight", type=float, default=0.25)
    ap.add_argument("--loss-mel-weight", type=float, default=0.30)
    ap.add_argument("--loss-low-weight", type=float, default=0.25)
    ap.add_argument("--loss-timbre-weight", type=float, default=0.12)
    ap.add_argument("--loss-donor-band-weight", type=float, default=0.0)
    args = ap.parse_args()
    cfg = TrainConfig(
        out_root=Path(args.out_root),
        cache_dir=Path(args.cache_dir),
        judge_ckpt=Path(args.judge_ckpt),
        epochs=int(args.epochs),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        eval_every_steps=int(args.eval_every_steps),
        batch_size=int(args.batch_size),
        lr_model=float(args.lr_model),
        lr_decoder=float(args.lr_decoder),
        single_genre_target=str(args.single_genre_target),
        benchmark_seconds=float(args.benchmark_seconds),
        final_seconds=float(args.final_seconds),
        source_hint_low_keep=float(args.source_hint_low_keep),
        source_hint_mid_keep=float(args.source_hint_mid_keep),
        source_hint_high_keep=float(args.source_hint_high_keep),
        proposal_scale=float(args.proposal_scale),
        source_skip_mix=float(args.source_skip_mix),
        loss_lat_weight=float(args.loss_lat_weight),
        loss_dt_weight=float(args.loss_dt_weight),
        loss_mel_weight=float(args.loss_mel_weight),
        loss_low_weight=float(args.loss_low_weight),
        loss_timbre_weight=float(args.loss_timbre_weight),
        loss_donor_band_weight=float(args.loss_donor_band_weight),
    )
    summary = train(cfg)
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
