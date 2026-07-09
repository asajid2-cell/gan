from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .lab3_bridge import load_audio_chunk
from .lab3_data import stratified_group_split_indices
from .lab3_diffusion_data import (
    DIFFUSION_SR,
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from .lab3_diffusion_train import load_bigvgan_robust, vocode_bigvgan


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _device_from_arg(raw: str) -> torch.device:
    if str(raw).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(raw))


def normalize_mel_np(mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    span = float(mel_max) - float(mel_min)
    if span < 1e-6:
        span = 1.0
    out = 2.0 * (np.asarray(mel, dtype=np.float32) - float(mel_min)) / span - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def smooth_mel_tensor(mel: torch.Tensor, time_kernel: int = 15, freq_kernel: int = 9) -> torch.Tensor:
    x = mel
    if int(time_kernel) > 1:
        pad = int(time_kernel) // 2
        x = F.avg_pool2d(F.pad(x, (pad, pad, 0, 0), mode="replicate"), (1, int(time_kernel)), stride=1)
    if int(freq_kernel) > 1:
        pad = int(freq_kernel) // 2
        x = F.avg_pool2d(F.pad(x, (0, 0, pad, pad), mode="replicate"), (int(freq_kernel), 1), stride=1)
    return x


def structure_proxy_from_mel(mel_norm: np.ndarray) -> np.ndarray:
    mel_t = torch.from_numpy(mel_norm[None, None, :, :]).float()
    coarse = smooth_mel_tensor(mel_t, time_kernel=19, freq_kernel=15)[0, 0].cpu().numpy()
    out = coarse.astype(np.float32)
    out[:20, :] = 0.75 * mel_norm[:20, :] + 0.25 * out[:20, :]
    if out.shape[0] > 56:
        hf_mean = out[56:, :].mean(axis=0, keepdims=True)
        out[56:, :] = np.repeat(hf_mean, out.shape[0] - 56, axis=0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def cond_feat_from_audio(audio: np.ndarray, max_frames: int, height: int) -> np.ndarray:
    chroma = pad_or_trim(extract_chroma(audio, sr=DIFFUSION_SR), max_frames, axis=1)
    onset = pad_or_trim(extract_onset(audio, sr=DIFFUSION_SR), max_frames, axis=0)
    beat = pad_or_trim(extract_beat_grid(audio, sr=DIFFUSION_SR, n_frames=max_frames), max_frames, axis=0)
    return np.concatenate(
        [
            np.repeat(chroma[:, None, :], int(height), axis=1),
            np.repeat(onset[None, None, :], int(height), axis=1),
            np.repeat(beat[None, None, :], int(height), axis=1),
        ],
        axis=0,
    ).astype(np.float32)


def cond_feat_from_arrays(arrays: Dict[str, np.ndarray], idx: int, max_frames: int, height: int) -> np.ndarray:
    chroma = np.asarray(arrays["chroma"][idx], dtype=np.float32)[:, :max_frames]
    onset = np.asarray(arrays["onset"][idx], dtype=np.float32)[:max_frames]
    beat = np.asarray(arrays["beat"][idx], dtype=np.float32)[:max_frames]
    return np.concatenate(
        [
            np.repeat(chroma[:, None, :], int(height), axis=1),
            np.repeat(onset[None, None, :], int(height), axis=1),
            np.repeat(beat[None, None, :], int(height), axis=1),
        ],
        axis=0,
    ).astype(np.float32)


class RealRetrievalDataset(Dataset):
    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        index_df: pd.DataFrame,
        indices: Sequence[int],
        *,
        mel_min: float,
        mel_max: float,
        max_frames: int,
        seed: int,
    ) -> None:
        self.arrays = arrays
        self.index_df = index_df.reset_index(drop=True)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.max_frames = int(max_frames)
        self.rng = random.Random(int(seed))
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
        source_track = self.track_ids[int(idx)]
        if len(pool) <= 1:
            return int(idx)
        for _ in range(16):
            cand = int(pool[self.rng.randrange(len(pool))])
            if cand != int(idx) and self.track_ids[cand] != source_track:
                return cand
        for cand_raw in pool:
            cand = int(cand_raw)
            if cand != int(idx) and self.track_ids[cand] != source_track:
                return cand
        for cand_raw in pool:
            cand = int(cand_raw)
            if cand != int(idx):
                return cand
        return int(idx)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[int(item)])
        genre = int(self.genre_idx[idx])
        donor_idx = self._pick_donor(idx, genre)
        target_mel = np.asarray(self.arrays["mel"][idx], dtype=np.float32)[:, : self.max_frames]
        donor_mel = np.asarray(self.arrays["mel"][donor_idx], dtype=np.float32)[:, : self.max_frames]
        target_norm = normalize_mel_np(target_mel, self.mel_min, self.mel_max)
        donor_norm = normalize_mel_np(donor_mel, self.mel_min, self.mel_max)
        struct = structure_proxy_from_mel(target_norm)
        cond_feat = cond_feat_from_arrays(self.arrays, idx, self.max_frames, target_norm.shape[0])
        return {
            "target_mel": torch.from_numpy(target_norm[None, :, :]),
            "donor_mel": torch.from_numpy(donor_norm[None, :, :]),
            "struct_mel": torch.from_numpy(struct[None, :, :]),
            "cond_feat": torch.from_numpy(cond_feat),
            "genre_idx": torch.tensor(genre, dtype=torch.long),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = max(1, out_ch // 8)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RetrievalFusionUNet(nn.Module):
    def __init__(self, in_ch: int, num_genres: int, base_ch: int = 48) -> None:
        super().__init__()
        self.genre_emb = nn.Embedding(int(num_genres), 64)
        self.donor_head = nn.Sequential(
            nn.Conv2d(1, base_ch, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)
        self.mid = ConvBlock(base_ch * 4, base_ch * 4)
        self.film = nn.Linear(base_ch * 2 + 64, base_ch * 8)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(
        self,
        struct_mel: torch.Tensor,
        donor_mel: torch.Tensor,
        cond_feat: torch.Tensor,
        genre_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([struct_mel, donor_mel, cond_feat], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        h = self.mid(self.down2(e2))
        donor_vec = self.donor_head(donor_mel).flatten(1)
        film = self.film(torch.cat([donor_vec, self.genre_emb(genre_idx)], dim=-1))
        scale, bias = film.chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * scale[:, :, None, None]) + 0.1 * bias[:, :, None, None]
        h = self.up1(h)
        h = self.dec1(torch.cat([h, e2], dim=1))
        h = self.up2(h)
        h = self.dec2(torch.cat([h, e1], dim=1))
        return torch.tanh(self.out(h))


class MelGenreJudge(nn.Module):
    def __init__(self, num_genres: int, base_ch: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_ch, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(base_ch * 4, int(num_genres))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(mel).flatten(1))


def _mel_timbre_stats(mel: torch.Tensor) -> torch.Tensor:
    mean_f = mel.mean(dim=-1)
    std_f = mel.std(dim=-1)
    return torch.cat([mean_f, std_f], dim=1)


def _make_balanced_sampler(ds: RealRetrievalDataset, seed: int) -> WeightedRandomSampler:
    genre = ds.genre_indices()
    uniq, counts = np.unique(genre, return_counts=True)
    inv = {int(g): 1.0 / max(1, int(c)) for g, c in zip(uniq.tolist(), counts.tolist())}
    weights = np.asarray([inv[int(g)] for g in genre.tolist()], dtype=np.float64)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True, generator=generator)


def _train_judge(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_genres: int,
    device: torch.device,
    steps: int,
    val_batches: int = 20,
) -> Tuple[Optional[MelGenreJudge], Dict[str, float]]:
    if int(num_genres) < 2 or int(steps) <= 0:
        return None, {"enabled": 0.0, "val_acc": 0.0}
    judge = MelGenreJudge(num_genres=num_genres).to(device)
    opt = torch.optim.AdamW(judge.parameters(), lr=3e-4, weight_decay=1e-4)
    train_iter = iter(train_loader)
    for _ in range(int(steps)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        mel = batch["target_mel"].to(device)
        genre = batch["genre_idx"].to(device)
        loss = F.cross_entropy(judge(mel), genre)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (_ + 1) % 100 == 0 or (_ + 1) == int(steps):
            print(json.dumps({"event": "judge_progress", "step": int(_ + 1), "steps": int(steps), "loss": float(loss.item())}), flush=True)
    judge.eval()
    accs: List[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if int(val_batches) > 0 and batch_idx >= int(val_batches):
                break
            pred = judge(batch["target_mel"].to(device)).argmax(dim=-1)
            accs.append(float((pred == batch["genre_idx"].to(device)).float().mean().item()))
    for p in judge.parameters():
        p.requires_grad = False
    return judge, {"enabled": 1.0, "val_acc": float(np.mean(accs)) if accs else 0.0}


@dataclass
class RealTransferTrainConfig:
    cache_dir: Path
    out_root: Path = Path("saves2") / "real_music_transfer"
    resume_checkpoint: Optional[Path] = None
    resume_out_dir: Optional[Path] = None
    epochs: int = 8
    batch_size: int = 4
    max_batches_per_epoch: int = 0
    val_batches: int = 20
    max_frames: int = 320
    base_ch: int = 48
    lr: float = 2e-4
    weight_decay: float = 1e-4
    judge_steps: int = 400
    judge_loss_weight: float = 0.25
    donor_timbre_weight: float = 0.15
    checkpoint_every_batches: int = 1000
    seed: int = 328
    device: str = "auto"
    epoch_sample_plan: Path = Path("saves2") / "real_music_transfer" / "validation_plan.json"
    epoch_sample_count: int = 2
    epoch_sample_every: int = 1
    epoch_sample_seconds: float = 12.0
    epoch_longform_seconds: float = 60.0
    epoch_sample_chunk_seconds: float = 3.0
    epoch_sample_overlap_seconds: float = 0.5
    epoch_sample_style_strength: float = 1.0
    epoch_sample_envelope_strength: float = 0.75


def _load_epoch_sample_cases(plan_path: Path, count: int, genre_to_idx: Dict[str, int]) -> List[Dict[str, Any]]:
    if int(count) <= 0:
        return []
    path = Path(plan_path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("rows", []))
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for row in rows:
        source_audio = str(row.get("source_audio", ""))
        target_genre = str(row.get("target_genre", ""))
        case_id = str(row.get("case_id", f"case_{len(out):04d}"))
        if not source_audio or target_genre not in genre_to_idx:
            continue
        key = (source_audio, target_genre, case_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(row))
        if len(out) >= int(count):
            break
    return out


@torch.no_grad()
def _render_epoch_samples(
    *,
    cfg: RealTransferTrainConfig,
    epoch: int,
    model: RetrievalFusionUNet,
    arrays: Dict[str, np.ndarray],
    index_df: pd.DataFrame,
    genre_to_idx: Dict[str, int],
    mel_min: float,
    mel_max: float,
    max_frames: int,
    out_dir: Path,
    device: torch.device,
    sample_state: Dict[str, Any],
) -> Dict[str, Any]:
    every = max(1, int(cfg.epoch_sample_every))
    if int(cfg.epoch_sample_count) <= 0 or int(epoch) % every != 0:
        return {"enabled": False, "reason": "disabled_or_not_sample_epoch", "epoch": int(epoch)}
    cases = _load_epoch_sample_cases(Path(cfg.epoch_sample_plan), int(cfg.epoch_sample_count), genre_to_idx)
    if not cases:
        return {"enabled": False, "reason": "no_sample_cases", "epoch": int(epoch), "plan": str(cfg.epoch_sample_plan)}

    sample_dir = Path(out_dir) / "epoch_samples" / f"epoch_{int(epoch):03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    if "bank" not in sample_state:
        sample_state["bank"] = build_track_bank(index_df, arrays)
    if "vocoder" not in sample_state:
        sample_state["vocoder"] = load_bigvgan_robust(device=device)

    was_training = bool(model.training)
    model.eval()
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for case in cases:
        case_id = str(case.get("case_id", f"case_{len(rows):04d}"))
        source_audio = Path(str(case["source_audio"]))
        target_genre = str(case["target_genre"])
        target_idx = int(genre_to_idx[target_genre])
        try:
            for form, seconds in (
                ("shortform", float(cfg.epoch_sample_seconds)),
                ("longform", float(cfg.epoch_longform_seconds)),
            ):
                source = load_audio_chunk(source_audio, sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
                donor = choose_donor_track(source, sample_state["bank"], target_idx)
                generated = generate_longform(
                    model,
                    source_audio=source,
                    target_genre_idx=target_idx,
                    donor_track=donor,
                    arrays=arrays,
                    mel_min=float(mel_min),
                    mel_max=float(mel_max),
                    max_frames=int(max_frames),
                    chunk_seconds=float(cfg.epoch_sample_chunk_seconds),
                    overlap_seconds=float(cfg.epoch_sample_overlap_seconds),
                    vocoder=sample_state["vocoder"],
                    device=device,
                    style_strength=float(cfg.epoch_sample_style_strength),
                    envelope_strength=float(cfg.epoch_sample_envelope_strength),
                )
                out_wav = sample_dir / form / f"{case_id}__to__{target_genre}.wav"
                out_wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_wav), generated, DIFFUSION_SR)
                rows.append(
                    {
                        "epoch": int(epoch),
                        "form": form,
                        "case_id": case_id,
                        "source_audio": str(source_audio),
                        "target_genre": target_genre,
                        "target_genre_idx": int(target_idx),
                        "donor_track_id": str(donor.get("track_id", "")),
                        "seconds": float(seconds),
                        "out_wav": str(out_wav),
                    }
                )
        except RuntimeError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            errors.append({"case_id": case_id, "target_genre": target_genre, "error": str(exc)})
        except Exception as exc:
            errors.append({"case_id": case_id, "target_genre": target_genre, "error": str(exc)})
    if was_training:
        model.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "enabled": True,
        "epoch": int(epoch),
        "plan": str(cfg.epoch_sample_plan),
        "sample_dir": str(sample_dir),
        "sample_count_requested": int(cfg.epoch_sample_count),
        "shortform_seconds": float(cfg.epoch_sample_seconds),
        "longform_seconds": float(cfg.epoch_longform_seconds),
        "chunk_seconds": float(cfg.epoch_sample_chunk_seconds),
        "overlap_seconds": float(cfg.epoch_sample_overlap_seconds),
        "style_strength": float(cfg.epoch_sample_style_strength),
        "envelope_strength": float(cfg.epoch_sample_envelope_strength),
        "rows": rows,
        "errors": errors,
    }
    _write_json(sample_dir / "summary.json", summary)
    return summary


def train_real_transfer(cfg: RealTransferTrainConfig) -> Dict[str, Any]:
    device = _device_from_arg(str(cfg.device))
    resume_checkpoint = Path(cfg.resume_checkpoint) if cfg.resume_checkpoint else None
    if cfg.resume_out_dir:
        out_dir = Path(cfg.resume_out_dir)
    elif resume_checkpoint is not None:
        out_dir = resume_checkpoint.resolve().parents[1]
    else:
        out_dir = Path(cfg.out_root) / f"real_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(cfg))

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    track_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, track_ids, val_ratio=0.15, seed=int(cfg.seed))
    if len(val_idx) == 0:
        val_idx = train_idx[: min(64, len(train_idx))]

    train_ds = RealRetrievalDataset(
        arrays,
        index_df,
        train_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        seed=int(cfg.seed),
    )
    val_ds = RealRetrievalDataset(
        arrays,
        index_df,
        val_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        seed=int(cfg.seed + 1),
    )
    base_train_sampler = _make_balanced_sampler(train_ds, int(cfg.seed))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        sampler=base_train_sampler,
        num_workers=0,
        drop_last=len(train_ds) >= int(cfg.batch_size),
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    model = RetrievalFusionUNet(in_ch=16, num_genres=len(genre_to_idx), base_ch=int(cfg.base_ch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    judge, judge_summary = _train_judge(train_loader, val_loader, len(genre_to_idx), device, int(cfg.judge_steps), int(cfg.val_batches))
    _write_json(out_dir / "judge_summary.json", judge_summary)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    global_step = 0
    start_epoch = 1
    resume_batch_start = 0
    resume_epoch_loss_sum = 0.0
    resume_epoch_loss_count = 0

    history_path = out_dir / "history.json"
    if history_path.exists():
        try:
            prior = json.loads(history_path.read_text(encoding="utf-8"))
            history = list(prior.get("rows", []))
            if history:
                best_val = min(float(r.get("val_l1", float("inf"))) for r in history)
        except Exception:
            history = []

    if resume_checkpoint is not None:
        payload = torch.load(str(resume_checkpoint), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if payload.get("optimizer") is not None:
            opt.load_state_dict(payload["optimizer"])
        global_step = int(payload.get("global_step", 0))
        resume_batch_start = int(payload.get("next_batch_idx", 0))
        if resume_batch_start > 0:
            start_epoch = int(payload.get("epoch", 0))
            resume_epoch_loss_sum = float(payload.get("epoch_loss_sum", 0.0))
            resume_epoch_loss_count = int(payload.get("epoch_loss_count", 0))
        else:
            start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = min(best_val, float(payload.get("val_l1", float("inf"))))
        print(
            json.dumps(
                {
                    "event": "resume_training",
                    "checkpoint": str(resume_checkpoint),
                    "out_dir": str(out_dir),
                    "start_epoch": int(start_epoch),
                    "resume_batch_start": int(resume_batch_start),
                    "global_step": int(global_step),
                    "history_rows": int(len(history)),
                    "optimizer_state": "loaded" if payload.get("optimizer") is not None else "fresh",
                }
            ),
            flush=True,
        )

    def save_ckpt(path: Path, epoch: int, val_l1: float) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "genre_to_idx": genre_to_idx,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "val_l1": float(val_l1),
                "optimizer": opt.state_dict(),
                "meta": {"mel_min": float(meta.mel_min), "mel_max": float(meta.mel_max), "max_frames": int(cfg.max_frames)},
            },
            str(path),
        )

    def save_partial_ckpt(path: Path, epoch: int, next_batch_idx: int, epoch_loss_sum: float, epoch_loss_count: int) -> None:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": asdict(cfg),
                "genre_to_idx": genre_to_idx,
                "epoch": int(epoch),
                "next_batch_idx": int(next_batch_idx),
                "global_step": int(global_step),
                "val_l1": float(best_val),
                "epoch_loss_sum": float(epoch_loss_sum),
                "epoch_loss_count": int(epoch_loss_count),
                "meta": {"mel_min": float(meta.mel_min), "mel_max": float(meta.mel_max), "max_frames": int(cfg.max_frames)},
            },
            str(path),
        )

    if start_epoch > int(cfg.epochs):
        summary = {
            "out_dir": str(out_dir),
            "best_checkpoint": str(ckpt_dir / "best_by_val.pt"),
            "latest_checkpoint": str(ckpt_dir / "latest.pt"),
            "genres": genre_to_idx,
            "history": history,
            "judge_summary": judge_summary,
        }
        _write_json(out_dir / "summary.json", summary)
        return summary

    sample_state: Dict[str, Any] = {}
    for epoch in range(int(start_epoch), int(cfg.epochs) + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        train_loader_epoch = train_loader
        resume_batch_offset = 0
        if epoch == int(start_epoch) and int(resume_batch_start) > 0:
            epoch_loss_sum = float(resume_epoch_loss_sum)
            epoch_loss_count = int(resume_epoch_loss_count)
            if epoch_loss_count > 0:
                epoch_losses.append(float(epoch_loss_sum / max(1, epoch_loss_count)))
            resume_batch_offset = int(resume_batch_start)
            sampled_indices = list(iter(_make_balanced_sampler(train_ds, int(cfg.seed))))
            sample_offset = int(resume_batch_offset) * int(cfg.batch_size)
            train_loader_epoch = DataLoader(
                train_ds,
                batch_size=int(cfg.batch_size),
                sampler=sampled_indices[sample_offset:],
                num_workers=0,
                drop_last=len(train_ds) >= int(cfg.batch_size),
            )
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader_epoch):
            absolute_batch_idx = int(batch_idx) + int(resume_batch_offset)
            if int(cfg.max_batches_per_epoch) > 0 and absolute_batch_idx >= int(cfg.max_batches_per_epoch):
                break
            target = batch["target_mel"].to(device)
            donor = batch["donor_mel"].to(device)
            struct = batch["struct_mel"].to(device)
            cond_feat = batch["cond_feat"].to(device)
            genre = batch["genre_idx"].to(device)
            pred = model(struct, donor, cond_feat, genre)
            loss_l1 = F.l1_loss(pred, target)
            loss_dt = F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
            loss_df = F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :])
            loss_timbre = F.l1_loss(_mel_timbre_stats(pred), _mel_timbre_stats(donor))
            loss = loss_l1 + 0.25 * loss_dt + 0.20 * loss_df + float(cfg.donor_timbre_weight) * loss_timbre
            if judge is not None and float(cfg.judge_loss_weight) > 0:
                loss = loss + float(cfg.judge_loss_weight) * F.cross_entropy(judge(pred), genre)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            global_step += 1
            batch_number = int(absolute_batch_idx + 1)
            if batch_number == 1 or batch_number % 500 == 0:
                print(
                    json.dumps(
                        {
                            "event": "train_progress",
                            "epoch": int(epoch),
                            "batch": int(batch_number),
                            "global_step": int(global_step),
                            "loss": float(loss.item()),
                            "elapsed_seconds": float(time.time() - t0),
                        }
                    ),
                    flush=True,
                )
            if int(cfg.checkpoint_every_batches) > 0 and batch_number % int(cfg.checkpoint_every_batches) == 0:
                save_partial_ckpt(ckpt_dir / "partial.pt", epoch, batch_number, epoch_loss_sum, epoch_loss_count)
                print(
                    json.dumps(
                        {
                            "event": "partial_checkpoint",
                            "epoch": int(epoch),
                            "next_batch": int(batch_number),
                            "global_step": int(global_step),
                            "path": str(ckpt_dir / "partial.pt"),
                        }
                    ),
                    flush=True,
                )

        model.eval()
        val_l1: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if int(cfg.val_batches) > 0 and batch_idx >= int(cfg.val_batches):
                    break
                pred = model(
                    batch["struct_mel"].to(device),
                    batch["donor_mel"].to(device),
                    batch["cond_feat"].to(device),
                    batch["genre_idx"].to(device),
                )
                val_l1.append(float(F.l1_loss(pred, batch["target_mel"].to(device)).item()))
        val_mean = float(np.mean(val_l1)) if val_l1 else float("inf")
        save_ckpt(ckpt_dir / "latest.pt", epoch, val_mean)
        save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, val_mean)
        if val_mean < best_val:
            best_val = val_mean
            save_ckpt(ckpt_dir / "best_by_val.pt", epoch, val_mean)
        partial_path = ckpt_dir / "partial.pt"
        if partial_path.exists():
            partial_path.unlink()
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "val_l1": val_mean,
            "epoch_seconds": float(time.time() - t0),
        }
        sample_summary = _render_epoch_samples(
            cfg=cfg,
            epoch=int(epoch),
            model=model,
            arrays=arrays,
            index_df=index_df,
            genre_to_idx=genre_to_idx,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            out_dir=out_dir,
            device=device,
            sample_state=sample_state,
        )
        row["epoch_samples"] = {
            "enabled": bool(sample_summary.get("enabled", False)),
            "sample_dir": sample_summary.get("sample_dir"),
            "n_files": int(len(sample_summary.get("rows", []))),
            "n_errors": int(len(sample_summary.get("errors", []))),
            "summary": str(Path(sample_summary["sample_dir"]) / "summary.json") if sample_summary.get("sample_dir") else "",
            "reason": sample_summary.get("reason", ""),
        }
        history.append(row)
        _write_json(out_dir / "history.json", {"rows": history})
        print(json.dumps(row))

    summary = {
        "out_dir": str(out_dir),
        "best_checkpoint": str(ckpt_dir / "best_by_val.pt"),
        "latest_checkpoint": str(ckpt_dir / "latest.pt"),
        "genres": genre_to_idx,
        "history": history,
        "judge_summary": judge_summary,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def split_audio_overlapping(audio: np.ndarray, chunk_seconds: float, overlap_seconds: float, sr: int) -> List[Dict[str, Any]]:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    chunk = max(1, int(round(float(chunk_seconds) * int(sr))))
    hop = max(1, int(round((float(chunk_seconds) - float(overlap_seconds)) * int(sr))))
    if len(y) <= chunk:
        return [{"audio": np.pad(y, (0, max(0, chunk - len(y))), mode="constant"), "start": 0}]
    out: List[Dict[str, Any]] = []
    for start in range(0, max(1, len(y) - chunk + 1), hop):
        out.append({"audio": y[start : start + chunk], "start": int(start)})
    last_start = len(y) - chunk
    if out[-1]["start"] != last_start:
        out.append({"audio": y[last_start : last_start + chunk], "start": int(last_start)})
    return out


def assemble_audio_crossfade(chunks: Sequence[np.ndarray], overlap_seconds: float, sr: int) -> np.ndarray:
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    overlap = max(0, int(round(float(overlap_seconds) * int(sr))))
    out = np.asarray(chunks[0], dtype=np.float32).copy()
    for raw in chunks[1:]:
        y = np.asarray(raw, dtype=np.float32).reshape(-1)
        n = min(overlap, len(out), len(y))
        if n > 0:
            tail = out[-n:]
            head = y[:n]
            tail_rms = float(np.sqrt(np.mean(np.square(tail), dtype=np.float64)) + 1e-8)
            head_rms = float(np.sqrt(np.mean(np.square(head), dtype=np.float64)) + 1e-8)
            gain = float(np.clip(tail_rms / head_rms, 0.70, 1.45))
            if abs(gain - 1.0) > 1e-3:
                y = y.copy()
                ramp_len = min(len(y), max(n, 3 * n))
                gain_ramp = np.linspace(gain, 1.0, ramp_len, dtype=np.float32)
                y[:ramp_len] *= gain_ramp
            phase = np.linspace(0.0, 0.5 * np.pi, n, dtype=np.float32)
            fade_out = np.cos(phase)
            fade_in = np.sin(phase)
            mixed = out[-n:] * fade_out + y[:n] * fade_in
            out = np.concatenate([out[:-n], mixed, y[n:]], axis=0)
        else:
            out = np.concatenate([out, y], axis=0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def smooth_longform_boundary_rms(
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    sr: int,
    window_seconds: float = 0.25,
    release_seconds: float = 1.0,
    max_adjust_db: float = 4.5,
) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
    if len(y) == 0:
        return y
    hop_seconds = max(1e-3, float(chunk_seconds) - float(overlap_seconds))
    window = max(1, int(round(float(window_seconds) * int(sr))))
    release = max(window, int(round(float(release_seconds) * int(sr))))
    max_gain = float(10.0 ** (abs(float(max_adjust_db)) / 20.0))
    min_gain = float(1.0 / max_gain)
    boundary_count = int(math.floor(max(0.0, float(len(y) / int(sr)) - float(overlap_seconds)) / hop_seconds))
    for i in range(1, boundary_count + 1):
        center = int(round(i * hop_seconds * int(sr)))
        if center - window < 0 or center + window > len(y):
            continue
        left = y[center - window : center]
        right = y[center : center + window]
        left_rms = float(np.sqrt(np.mean(np.square(left), dtype=np.float64)) + 1e-8)
        right_rms = float(np.sqrt(np.mean(np.square(right), dtype=np.float64)) + 1e-8)
        gain = float(np.clip(left_rms / max(right_rms, 1e-8), min_gain, max_gain))
        if abs(gain - 1.0) < 0.03:
            continue
        end = min(len(y), center + release)
        ramp = np.linspace(gain, 1.0, end - center, dtype=np.float32)
        y[center:end] *= ramp
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def _sample_rms_envelope(audio: np.ndarray, *, sr: int, n_samples: int) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(y) == 0:
        return np.ones((int(n_samples),), dtype=np.float32)
    if len(y) < int(n_samples):
        y = np.pad(y, (0, int(n_samples) - len(y)), mode="constant")
    else:
        y = y[: int(n_samples)]
    hop = 256
    frame = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=True)[0].astype(np.float32)
    if len(rms) <= 1:
        return np.full((int(n_samples),), float(rms[0]) if len(rms) else 1.0, dtype=np.float32)
    frame_pos = np.linspace(0, int(n_samples) - 1, num=len(rms), dtype=np.float32)
    sample_pos = np.arange(int(n_samples), dtype=np.float32)
    env = np.interp(sample_pos, frame_pos, rms).astype(np.float32)
    return np.maximum(env, 1e-4).astype(np.float32)


def apply_source_envelope_anchor(
    generated: np.ndarray,
    source: np.ndarray,
    *,
    sr: int,
    strength: float = 0.35,
) -> np.ndarray:
    y = np.asarray(generated, dtype=np.float32).reshape(-1)
    if len(y) == 0 or float(strength) <= 0.0:
        return y.astype(np.float32)
    src_env = _sample_rms_envelope(source, sr=int(sr), n_samples=len(y))
    gen_env = _sample_rms_envelope(y, sr=int(sr), n_samples=len(y))
    ratio = np.clip(src_env / np.maximum(gen_env, 1e-4), 0.55, 1.80).astype(np.float32)
    gain = 1.0 + float(np.clip(strength, 0.0, 1.0)) * (ratio - 1.0)
    anchored = y * gain
    old_rms = float(np.sqrt(np.mean(np.square(y), dtype=np.float64)) + 1e-8)
    new_rms = float(np.sqrt(np.mean(np.square(anchored), dtype=np.float64)) + 1e-8)
    anchored = anchored * float(np.clip(old_rms / new_rms, 0.80, 1.25))
    return np.clip(anchored, -1.0, 1.0).astype(np.float32)


def build_track_bank(index_df: pd.DataFrame, arrays: Dict[str, np.ndarray]) -> Dict[int, List[Dict[str, Any]]]:
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    grouped: Dict[Tuple[int, str], List[int]] = {}
    for idx in range(len(index_df)):
        g = int(genre_idx[int(idx)])
        tid = str(index_df.iloc[int(idx)]["track_id"])
        grouped.setdefault((g, tid), []).append(int(idx))
    bank: Dict[int, List[Dict[str, Any]]] = {}
    for (g, tid), rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(index_df.iloc[int(r)].get("start_sec", 0.0)))
        chroma_mean = np.mean([np.asarray(arrays["chroma"][int(r)], dtype=np.float32).mean(axis=1) for r in rows], axis=0)
        onset_mean = float(np.mean([float(np.asarray(arrays["onset"][int(r)], dtype=np.float32).mean()) for r in rows]))
        beat_mean = float(np.mean([float(np.asarray(arrays["beat"][int(r)], dtype=np.float32).mean()) for r in rows]))
        bank.setdefault(int(g), []).append(
            {
                "track_id": tid,
                "rows": rows,
                "chroma_mean": chroma_mean.astype(np.float32),
                "onset_mean": onset_mean,
                "beat_mean": beat_mean,
            }
        )
    return bank


def choose_donor_track(source_audio: np.ndarray, bank: Dict[int, List[Dict[str, Any]]], target_genre_idx: int) -> Dict[str, Any]:
    chroma = np.mean(librosa.feature.chroma_stft(y=source_audio, sr=DIFFUSION_SR), axis=1).astype(np.float32)
    onset_mean = float(np.mean(librosa.onset.onset_strength(y=source_audio, sr=DIFFUSION_SR)))
    best: Optional[Dict[str, Any]] = None
    best_score = -1e18
    for row in bank.get(int(target_genre_idx), []):
        denom = (np.linalg.norm(chroma) + 1e-6) * (np.linalg.norm(row["chroma_mean"]) + 1e-6)
        chroma_score = float(np.dot(chroma, row["chroma_mean"]) / denom)
        onset_score = -abs(onset_mean - float(row["onset_mean"]))
        score = chroma_score + 0.05 * onset_score
        if score > best_score:
            best_score = score
            best = dict(row)
    if best is None:
        raise RuntimeError(f"No donor track is available for target genre idx={target_genre_idx}")
    return best


@torch.no_grad()
def generate_longform(
    model: RetrievalFusionUNet,
    *,
    source_audio: np.ndarray,
    target_genre_idx: int,
    donor_track: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    mel_min: float,
    mel_max: float,
    max_frames: int,
    chunk_seconds: float,
    overlap_seconds: float,
    vocoder: Any,
    device: torch.device,
    style_strength: float = 1.00,
    envelope_strength: float = 0.35,
) -> np.ndarray:
    model.eval()
    chunks = split_audio_overlapping(source_audio, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds, sr=DIFFUSION_SR)
    donor_rows = donor_track["rows"]
    total = max(1, len(chunks) - 1)
    out_wavs: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        mel = pad_or_trim(extract_bigvgan_mel_np(chunk["audio"], sr=DIFFUSION_SR), int(max_frames), axis=1, pad_val=float(mel_min))
        mel_norm = normalize_mel_np(mel, mel_min, mel_max)
        struct = structure_proxy_from_mel(mel_norm)
        donor_idx = int(donor_rows[min(len(donor_rows) - 1, round((i / total) * max(0, len(donor_rows) - 1)))])
        donor_mel = np.asarray(arrays["mel"][donor_idx], dtype=np.float32)[:, : int(max_frames)]
        donor_norm = normalize_mel_np(donor_mel, mel_min, mel_max)
        cond_feat = cond_feat_from_audio(chunk["audio"], int(max_frames), mel_norm.shape[0])
        pred = model(
            torch.from_numpy(struct[None, None, :, :]).to(device),
            torch.from_numpy(donor_norm[None, None, :, :]).to(device),
            torch.from_numpy(cond_feat[None, :, :, :]).to(device),
            torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device),
        )
        pred_mel = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        # The trained generator is intentionally conservative; the final transfer
        # stage makes the selected target donor the dominant timbral reference
        # while keeping the source's lowest-band contour for song identity.
        style_mask = np.full((pred_mel.shape[0], 1), 0.985, dtype=np.float32)
        style_mask[:6, :] = 0.90
        blend = np.clip(float(style_strength) * style_mask, 0.0, 0.995).astype(np.float32)
        pred_mel = (1.0 - blend) * pred_mel + blend * donor_norm
        pred_mel[:6, :] = 0.15 * mel_norm[:6, :] + 0.85 * pred_mel[:6, :]
        if i > 0:
            pred_mel = smooth_mel_tensor(torch.from_numpy(pred_mel[None, None, :, :]), time_kernel=5, freq_kernel=3)[0, 0].cpu().numpy()
        wav = vocode_bigvgan(torch.from_numpy(pred_mel[None, None, :, :]).to(device), float(mel_min), float(mel_max), vocoder, device)
        wav = apply_source_envelope_anchor(
            np.asarray(wav, dtype=np.float32).reshape(-1),
            chunk["audio"],
            sr=DIFFUSION_SR,
            strength=float(envelope_strength),
        )
        out_wavs.append(np.asarray(wav, dtype=np.float32).reshape(-1))
    assembled = assemble_audio_crossfade(out_wavs, overlap_seconds=float(overlap_seconds), sr=DIFFUSION_SR)
    return smooth_longform_boundary_rms(
        assembled,
        chunk_seconds=float(chunk_seconds),
        overlap_seconds=float(overlap_seconds),
        sr=DIFFUSION_SR,
    )


def infer_real_transfer(
    *,
    checkpoint: Path,
    cache_dir: Path,
    source_audio: Path,
    target_genre: str,
    out_wav: Path,
    seconds: float = 30.0,
    chunk_seconds: float = 3.0,
    overlap_seconds: float = 0.5,
    style_strength: float = 1.0,
    envelope_strength: float = 0.35,
    device_arg: str = "auto",
) -> Dict[str, Any]:
    device = _device_from_arg(device_arg)
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cache_dir), mmap=True)
    if str(target_genre) not in genre_to_idx:
        raise ValueError(f"Unknown target genre '{target_genre}'. Available: {sorted(genre_to_idx)}")
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    ckpt_cfg = payload.get("cfg", {})
    max_frames = int(payload.get("meta", {}).get("max_frames", ckpt_cfg.get("max_frames", 320)))
    base_ch = int(ckpt_cfg.get("base_ch", 48))
    model = RetrievalFusionUNet(in_ch=16, num_genres=len(genre_to_idx), base_ch=base_ch).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    source = load_audio_chunk(Path(source_audio), sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
    bank = build_track_bank(index_df, arrays)
    target_idx = int(genre_to_idx[str(target_genre)])
    donor = choose_donor_track(source, bank, target_idx)
    vocoder = load_bigvgan_robust(device=device)
    generated = generate_longform(
        model,
        source_audio=source,
        target_genre_idx=target_idx,
        donor_track=donor,
        arrays=arrays,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=max_frames,
        chunk_seconds=float(chunk_seconds),
        overlap_seconds=float(overlap_seconds),
        vocoder=vocoder,
        device=device,
        style_strength=float(style_strength),
        envelope_strength=float(envelope_strength),
    )
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), generated, DIFFUSION_SR)
    meta_out = {
        "out_wav": str(out_wav),
        "checkpoint": str(checkpoint),
        "cache_dir": str(cache_dir),
        "source_audio": str(source_audio),
        "target_genre": str(target_genre),
        "donor_track_id": str(donor["track_id"]),
        "seconds": float(seconds),
        "style_strength": float(style_strength),
        "envelope_strength": float(envelope_strength),
    }
    _write_json(out_wav.with_suffix(".json"), meta_out)
    return meta_out
