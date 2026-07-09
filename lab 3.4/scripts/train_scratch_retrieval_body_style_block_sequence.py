from __future__ import annotations

import argparse
import csv
import json
import random
import sys
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
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB33_SCRIPTS = REPO_ROOT / "lab 3.3" / "scripts"
if str(LAB33_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB33_SCRIPTS))

from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import (
    DIFFUSION_SR,
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from dggr.lab3_diffusion_train import load_bigvgan_robust, vocode_bigvgan
from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _resolve_stems, picked_songs
from train_scratch_structure_diffusion import (
    MelGenreJudge,
    _audio_metrics,
    _device_from_arg,
    _make_balanced_sampler,
    _normalize_mel_np,
    _slug,
    _write_json,
    assemble_audio_crossfade,
    smooth_mel_tensor,
    split_audio_overlapping,
)


def _pad_audio(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) >= n:
        return y[:n].astype(np.float32)
    return np.pad(y.astype(np.float32), (0, n - len(y)))


def _structure_proxy_from_mel(mel_norm: np.ndarray) -> np.ndarray:
    mel_t = torch.from_numpy(mel_norm[None, None, :, :]).float()
    coarse = smooth_mel_tensor(mel_t, time_kernel=19, freq_kernel=15)[0, 0].cpu().numpy()
    out = coarse.astype(np.float32)
    out[:20, :] = 0.75 * mel_norm[:20, :] + 0.25 * out[:20, :]
    hf_mean = out[56:, :].mean(axis=0, keepdims=True)
    out[56:, :] = np.repeat(hf_mean, out.shape[0] - 56, axis=0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _expanded_cond_feat(arrays: Dict[str, np.ndarray], idx: int, max_frames: int, height: int) -> np.ndarray:
    chroma = pad_or_trim(np.asarray(arrays["chroma"][idx], dtype=np.float32), max_frames, axis=1, pad_val=0.0)
    onset = pad_or_trim(np.asarray(arrays["onset"][idx], dtype=np.float32), max_frames, axis=0, pad_val=0.0)
    beat = pad_or_trim(np.asarray(arrays["beat"][idx], dtype=np.float32), max_frames, axis=0, pad_val=0.0)
    return np.concatenate(
        [
            np.repeat(chroma[:, None, :], height, axis=1),
            np.repeat(onset[None, None, :], height, axis=1),
            np.repeat(beat[None, None, :], height, axis=1),
        ],
        axis=0,
    ).astype(np.float32)


@torch.no_grad()
def _judge_probs_for_audio(audio: np.ndarray, judge: MelGenreJudge, device: torch.device, max_frames: int) -> np.ndarray:
    win = int(round(5.0 * DIFFUSION_SR))
    starts = [0] if len(audio) <= win else np.linspace(0, max(0, len(audio) - win), 3, dtype=np.int64).tolist()
    probs: List[np.ndarray] = []
    for st in starts:
        seg = audio[int(st) : int(st) + win].astype(np.float32)
        mel = extract_bigvgan_mel_np(seg, sr=DIFFUSION_SR)
        mel = pad_or_trim(mel, max_frames, axis=1, pad_val=-11.5)
        mel_norm = _normalize_mel_np(mel, -11.5, 2.0)
        mel_t = torch.from_numpy(mel_norm[None, None, :, :]).to(device)
        probs.append(torch.softmax(judge(mel_t), dim=-1)[0].cpu().numpy().astype(np.float32))
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def _cond_feat_from_audio(audio: np.ndarray, max_frames: int, height: int) -> np.ndarray:
    chroma = extract_chroma(audio, sr=DIFFUSION_SR)
    chroma = pad_or_trim(chroma, max_frames, axis=1, pad_val=0.0)
    onset = extract_onset(audio, sr=DIFFUSION_SR)
    onset = pad_or_trim(onset, max_frames, axis=0, pad_val=0.0)
    beat = extract_beat_grid(audio, sr=DIFFUSION_SR, n_frames=max_frames)
    beat = pad_or_trim(beat, max_frames, axis=0, pad_val=0.0)
    return np.concatenate(
        [
            np.repeat(chroma[:, None, :], height, axis=1),
            np.repeat(onset[None, None, :], height, axis=1),
            np.repeat(beat[None, None, :], height, axis=1),
        ],
        axis=0,
    ).astype(np.float32)


def _load_or_train_judge(
    judge_path: Optional[Path],
    cache_dir: Path,
    out_dir: Path,
    device: torch.device,
    max_frames: int,
) -> Tuple[MelGenreJudge, Dict[str, int]]:
    if judge_path and judge_path.exists():
        payload = torch.load(str(judge_path), map_location=device, weights_only=False)
        genre_to_idx = {str(k): int(v) for k, v in payload["genre_to_idx"].items()}
        judge = MelGenreJudge(num_genres=len(genre_to_idx)).to(device)
        judge.load_state_dict(payload["judge"])
        judge.eval()
        for p in judge.parameters():
            p.requires_grad = False
        return judge, genre_to_idx

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=328)

    class JudgeDataset(Dataset):
        def __init__(self, indices: Sequence[int]) -> None:
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            idx = int(self.indices[int(i)])
            mel = np.asarray(arrays["mel"][idx], dtype=np.float32)[:, :max_frames]
            mel_norm = _normalize_mel_np(mel, float(meta.mel_min), float(meta.mel_max))
            return {
                "mel": torch.from_numpy(mel_norm[None, :, :]),
                "genre_idx": torch.tensor(int(genre_idx[idx]), dtype=torch.long),
            }

        def genre_indices(self) -> np.ndarray:
            return genre_idx[self.indices]

    train_ds = JudgeDataset(train_idx)
    val_ds = JudgeDataset(val_idx)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=_make_balanced_sampler(train_ds, 328), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    judge = MelGenreJudge(num_genres=len(genre_to_idx)).to(device)
    opt = torch.optim.AdamW(judge.parameters(), lr=3e-4, weight_decay=1e-4)
    train_iter = iter(train_loader)
    for _ in range(300):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        mel = batch["mel"].to(device)
        genre = batch["genre_idx"].to(device)
        loss = F.cross_entropy(judge(mel), genre)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    judge.eval()
    accs: List[float] = []
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            genre = batch["genre_idx"].to(device)
            accs.append(float((judge(mel).argmax(dim=-1) == genre).float().mean().item()))
    save_path = out_dir / "genre_judge.pt"
    torch.save({"judge": judge.state_dict(), "genre_to_idx": genre_to_idx}, str(save_path))
    _write_json(out_dir / "judge_summary.json", {"val_acc": float(np.mean(accs)) if accs else 0.0, "judge_path": str(save_path)})
    for p in judge.parameters():
        p.requires_grad = False
    return judge, genre_to_idx


class RetrievalFusionDataset(Dataset):
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
        self.prev_idx_map: Dict[int, Optional[int]] = {}
        self.prev2_idx_map: Dict[int, Optional[int]] = {}
        for g in np.unique(self.genre_idx[self.indices]).tolist():
            self.by_genre[int(g)] = self.indices[self.genre_idx[self.indices] == int(g)]
        grouped: Dict[str, List[int]] = {}
        for idx in self.indices.tolist():
            grouped.setdefault(str(self.track_ids[int(idx)]), []).append(int(idx))
        for _, rows in grouped.items():
            rows = sorted(rows, key=lambda r: float(self.index_df.iloc[int(r)]["start_sec"]))
            prev: Optional[int] = None
            prev2: Optional[int] = None
            for row_idx in rows:
                self.prev_idx_map[int(row_idx)] = prev
                self.prev2_idx_map[int(row_idx)] = prev2
                prev2 = prev
                prev = int(row_idx)

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
        return int(candidates[self.rng.randrange(len(candidates))])

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[int(item)])
        genre = int(self.genre_idx[idx])
        donor_idx = self._pick_donor(idx, genre)
        prev_idx = self.prev_idx_map.get(int(idx))
        prev2_idx = self.prev2_idx_map.get(int(idx))
        target_mel = pad_or_trim(
            np.asarray(self.arrays["mel"][idx], dtype=np.float32),
            self.max_frames,
            axis=1,
            pad_val=float(self.mel_min),
        )
        donor_mel = pad_or_trim(
            np.asarray(self.arrays["mel"][donor_idx], dtype=np.float32),
            self.max_frames,
            axis=1,
            pad_val=float(self.mel_min),
        )
        if prev_idx is None:
            prev_mel = np.zeros_like(target_mel, dtype=np.float32)
        else:
            prev_mel = pad_or_trim(
                np.asarray(self.arrays["mel"][int(prev_idx)], dtype=np.float32),
                self.max_frames,
                axis=1,
                pad_val=float(self.mel_min),
            )
        if prev2_idx is None:
            prev2_mel = np.zeros_like(target_mel, dtype=np.float32)
        else:
            prev2_mel = pad_or_trim(
                np.asarray(self.arrays["mel"][int(prev2_idx)], dtype=np.float32),
                self.max_frames,
                axis=1,
                pad_val=float(self.mel_min),
            )
        target_norm = _normalize_mel_np(target_mel, self.mel_min, self.mel_max)
        donor_norm = _normalize_mel_np(donor_mel, self.mel_min, self.mel_max)
        prev_norm = _normalize_mel_np(prev_mel, self.mel_min, self.mel_max)
        prev2_norm = _normalize_mel_np(prev2_mel, self.mel_min, self.mel_max)
        struct = _structure_proxy_from_mel(target_norm)
        cond_feat = _expanded_cond_feat(self.arrays, idx, self.max_frames, target_norm.shape[0])
        context_count = 0.0
        if prev_idx is not None:
            context_count += 1.0
        if prev2_idx is not None:
            context_count += 1.0
        return {
            "target_mel": torch.from_numpy(target_norm[None, :, :]),
            "donor_mel": torch.from_numpy(donor_norm[None, :, :]),
            "prev_mel": torch.from_numpy(prev_norm[None, :, :]),
            "prev2_mel": torch.from_numpy(prev2_norm[None, :, :]),
            "struct_mel": torch.from_numpy(struct[None, :, :]),
            "cond_feat": torch.from_numpy(cond_feat),
            "genre_idx": torch.tensor(genre, dtype=torch.long),
            "context_count": torch.tensor(context_count / 2.0, dtype=torch.float32),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
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
    def __init__(self, in_ch: int, num_genres: int, base_ch: int = 48):
        super().__init__()
        self.genre_emb = nn.Embedding(num_genres, 64)
        self.donor_head = nn.Sequential(
            nn.Conv2d(1, base_ch, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.prev_head = nn.Sequential(
            nn.Conv2d(1, base_ch, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.prev2_head = nn.Sequential(
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
        self.film = nn.Linear(base_ch * 6 + 64 + 1, base_ch * 8)
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(
        self,
        struct_mel: torch.Tensor,
        donor_mel: torch.Tensor,
        prev_mel: torch.Tensor,
        prev2_mel: torch.Tensor,
        cond_feat: torch.Tensor,
        genre_idx: torch.Tensor,
        context_count: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([struct_mel, donor_mel, prev_mel, prev2_mel, cond_feat], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        h = self.mid(self.down2(e2))
        donor_vec = self.donor_head(donor_mel).flatten(1)
        prev_vec = self.prev_head(prev_mel).flatten(1)
        prev2_vec = self.prev2_head(prev2_mel).flatten(1)
        film = self.film(torch.cat([donor_vec, prev_vec, prev2_vec, self.genre_emb(genre_idx), context_count[:, None]], dim=-1))
        scale, bias = film.chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * scale[:, :, None, None]) + 0.1 * bias[:, :, None, None]
        h = self.up1(h)
        h = self.dec1(torch.cat([h, e2], dim=1))
        h = self.up2(h)
        h = self.dec2(torch.cat([h, e1], dim=1))
        return torch.tanh(self.out(h))


def _mel_timbre_stats(mel: torch.Tensor) -> torch.Tensor:
    mean_f = mel.mean(dim=-1)
    std_f = mel.std(dim=-1)
    return torch.cat([mean_f, std_f], dim=1)


def _band_profile(mel: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    band = mel[:, :, lo:hi, :]
    mean_t = band.mean(dim=-1)
    std_t = band.std(dim=-1)
    mean_f = band.mean(dim=-2)
    return torch.cat(
        [
            mean_t.reshape(mean_t.shape[0], -1),
            std_t.reshape(std_t.shape[0], -1),
            mean_f.reshape(mean_f.shape[0], -1),
        ],
        dim=1,
    )


def _band_envelope(mel: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    band = mel[:, :, lo:hi, :]
    env = band.mean(dim=-2)
    diff = env[:, :, 1:] - env[:, :, :-1]
    diff = F.pad(diff, (1, 0))
    return torch.cat([env.reshape(env.shape[0], -1), diff.reshape(diff.shape[0], -1)], dim=1)


def _build_track_bank(index_df: pd.DataFrame, arrays: Dict[str, np.ndarray], indices: np.ndarray) -> Dict[int, List[Dict[str, Any]]]:
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    chroma_arr = arrays["chroma"]
    onset_arr = arrays["onset"]
    beat_arr = arrays["beat"]
    grouped: Dict[Tuple[int, str], List[int]] = {}
    for idx in indices.tolist():
        g = int(genre_idx[int(idx)])
        tid = str(index_df.iloc[int(idx)]["track_id"])
        grouped.setdefault((g, tid), []).append(int(idx))
    bank: Dict[int, List[Dict[str, Any]]] = {}
    for (g, tid), rows in grouped.items():
        rows = sorted(rows, key=lambda r: float(index_df.iloc[int(r)]["start_sec"]))
        chroma_mean = np.mean([np.asarray(chroma_arr[int(r)], dtype=np.float32).mean(axis=1) for r in rows], axis=0).astype(np.float32)
        onset_mean = float(np.mean([float(np.asarray(onset_arr[int(r)], dtype=np.float32).mean()) for r in rows]))
        beat_mean = float(np.mean([float(np.asarray(beat_arr[int(r)], dtype=np.float32).mean()) for r in rows]))
        bank.setdefault(int(g), []).append(
            {
                "track_id": tid,
                "rows": rows,
                "chroma_mean": chroma_mean,
                "onset_mean": onset_mean,
                "beat_mean": beat_mean,
            }
        )
    return bank


def _choose_donor_track(source_audio: np.ndarray, bank: Dict[int, List[Dict[str, Any]]], target_genre_idx: int) -> Dict[str, Any]:
    chroma = np.mean(librosa.feature.chroma_stft(y=source_audio, sr=DIFFUSION_SR), axis=1).astype(np.float32)
    onset_mean = float(np.mean(librosa.onset.onset_strength(y=source_audio, sr=DIFFUSION_SR)))
    beat_strength = float(np.mean(librosa.onset.onset_strength(y=source_audio, sr=DIFFUSION_SR, lag=2)))
    best: Optional[Dict[str, Any]] = None
    best_score = -1e9
    for row in bank.get(int(target_genre_idx), []):
        chroma_score = float(np.dot(chroma, row["chroma_mean"]) / ((np.linalg.norm(chroma) + 1e-6) * (np.linalg.norm(row["chroma_mean"]) + 1e-6)))
        onset_score = -abs(onset_mean - float(row["onset_mean"]))
        beat_score = -abs(beat_strength - float(row["beat_mean"]))
        score = chroma_score + 0.05 * onset_score + 0.05 * beat_score
        if score > best_score:
            best_score = score
            best = row
    if best is None:
        raise RuntimeError(f"No donor track available for genre idx={target_genre_idx}")
    return dict(best)


def _assemble_mel_context_trim(mels: Sequence[np.ndarray], trim_frames: Sequence[int]) -> np.ndarray:
    if not mels:
        return np.zeros((80, 1), dtype=np.float32)
    kept: List[np.ndarray] = []
    for i, mel in enumerate(mels):
        cur = np.asarray(mel, dtype=np.float32)
        if i == 0:
            kept.append(cur)
            continue
        trim = int(max(0, min(trim_frames[i - 1], cur.shape[1] - 1)))
        kept.append(cur[:, trim:])
    if not kept:
        return np.zeros((80, 1), dtype=np.float32)
    return np.concatenate(kept, axis=1).astype(np.float32)


def _best_seam_shift(prev_tail: np.ndarray, next_head: np.ndarray, max_shift: int) -> int:
    best_shift = 0
    best_score = -1e18
    for shift in range(-int(max_shift), int(max_shift) + 1):
        if shift >= 0:
            a = prev_tail[:, shift:]
            b = next_head[:, : a.shape[1]]
        else:
            a = prev_tail[:, : prev_tail.shape[1] + shift]
            b = next_head[:, -shift : -shift + a.shape[1]]
        if a.shape[1] < 8 or b.shape[1] < 8:
            continue
        band_a = a[8:64, :]
        band_b = b[8:64, :]
        l1 = float(np.mean(np.abs(band_a - band_b)))
        va = band_a.reshape(-1) - float(np.mean(band_a))
        vb = band_b.reshape(-1) - float(np.mean(band_b))
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
        cos = float(np.dot(va, vb) / denom)
        score = -l1 + 0.10 * cos
        if score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def _assemble_mel_seamblend(
    mels: Sequence[np.ndarray],
    trim_frames: Sequence[int],
    *,
    seam_blend_frames: int = 48,
    max_shift_frames: int = 24,
) -> np.ndarray:
    if not mels:
        return np.zeros((80, 1), dtype=np.float32)
    assembled = np.asarray(mels[0], dtype=np.float32).copy()
    for i, mel in enumerate(mels[1:], start=1):
        cur = np.asarray(mel, dtype=np.float32)
        trim = int(max(0, min(trim_frames[i - 1], cur.shape[1] - 1)))
        kept = cur[:, trim:].astype(np.float32)
        if kept.shape[1] < 4:
            continue
        overlap = int(min(seam_blend_frames, max(1, assembled.shape[1] - 1), max(1, kept.shape[1] - 1)))
        if overlap < 8:
            assembled = np.concatenate([assembled, kept], axis=1).astype(np.float32)
            continue
        tail_window = assembled[:, -(overlap + max_shift_frames) :].astype(np.float32)
        head_window = kept[:, : (overlap + max_shift_frames)].astype(np.float32)
        shift = _best_seam_shift(tail_window, head_window, max_shift_frames)
        if shift > 0:
            kept = kept[:, shift:]
        elif shift < 0 and assembled.shape[1] + shift > overlap:
            assembled = assembled[:, : assembled.shape[1] + shift]
        overlap = int(min(seam_blend_frames, max(1, assembled.shape[1] - 1), max(1, kept.shape[1] - 1)))
        if overlap < 8:
            assembled = np.concatenate([assembled, kept], axis=1).astype(np.float32)
            continue
        tail = assembled[:, -overlap:].astype(np.float32)
        head = kept[:, :overlap].astype(np.float32)
        mean_delta = (np.mean(tail, axis=1, keepdims=True) - np.mean(head, axis=1, keepdims=True)).astype(np.float32)
        head = head + 0.35 * mean_delta
        fade = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :]
        blended = ((1.0 - fade) * tail + fade * head).astype(np.float32)
        assembled = np.concatenate([assembled[:, :-overlap], blended, kept[:, overlap:]], axis=1).astype(np.float32)
    return assembled.astype(np.float32)


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
) -> np.ndarray:
    model.eval()
    chunks = split_audio_overlapping(source_audio, chunk_seconds=float(chunk_seconds), overlap_seconds=float(overlap_seconds), sr=DIFFUSION_SR)
    total = max(1, len(chunks) - 1)
    donor_rows = donor_track["rows"]
    out_mels: List[np.ndarray] = []
    trim_frames: List[int] = []
    prev_pred_mel: Optional[np.ndarray] = None
    prev2_pred_mel: Optional[np.ndarray] = None
    for i, chunk in enumerate(chunks):
        mel_raw = extract_bigvgan_mel_np(chunk["audio"], sr=DIFFUSION_SR)
        mel_len = int(min(max_frames, mel_raw.shape[1]))
        mel = pad_or_trim(mel_raw, max_frames, axis=1, pad_val=float(mel_min))
        mel_norm = _normalize_mel_np(mel, mel_min, mel_max)
        struct = _structure_proxy_from_mel(mel_norm)
        donor_idx = int(donor_rows[min(len(donor_rows) - 1, round((i / total) * max(0, len(donor_rows) - 1)))])
        donor_mel = pad_or_trim(
            np.asarray(arrays["mel"][donor_idx], dtype=np.float32),
            max_frames,
            axis=1,
            pad_val=float(mel_min),
        )
        donor_norm = _normalize_mel_np(donor_mel, mel_min, mel_max)
        cond_feat = _cond_feat_from_audio(chunk["audio"], max_frames, mel_norm.shape[0])
        if prev_pred_mel is None:
            prev_norm = np.zeros_like(mel_norm, dtype=np.float32)
            context_count = 0.0
        else:
            prev_norm = prev_pred_mel.astype(np.float32)
            context_count = 0.5 if prev2_pred_mel is None else 1.0
        if prev2_pred_mel is None:
            prev2_norm = np.zeros_like(mel_norm, dtype=np.float32)
        else:
            prev2_norm = prev2_pred_mel.astype(np.float32)
        pred = model(
            torch.from_numpy(struct[None, None, :, :]).to(device),
            torch.from_numpy(donor_norm[None, None, :, :]).to(device),
            torch.from_numpy(prev_norm[None, None, :, :]).to(device),
            torch.from_numpy(prev2_norm[None, None, :, :]).to(device),
            torch.from_numpy(cond_feat[None, :, :, :]).to(device),
            torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device),
            torch.tensor([float(context_count)], dtype=torch.float32, device=device),
        )
        pred_mel = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_mel[:12, :] = 0.55 * mel_norm[:12, :] + 0.45 * pred_mel[:12, :]
        if i > 0:
            pred_mel = smooth_mel_tensor(torch.from_numpy(pred_mel[None, None, :, :]), time_kernel=7, freq_kernel=3)[0, 0].cpu().numpy().astype(np.float32)
            warm_cols = min(96, pred_mel.shape[1], prev_norm.shape[1])
            pred_mel[:, :warm_cols] = 0.82 * prev_norm[:, -warm_cols:] + 0.18 * pred_mel[:, :warm_cols]
        prev2_pred_mel = None if prev_pred_mel is None else prev_pred_mel.copy()
        prev_pred_mel = pred_mel.copy()
        out_mels.append(pred_mel[:, :mel_len].astype(np.float32))
        frames_per_sample = float(mel_len) / float(max(1, len(chunk["audio"])))
        trim_frames.append(int(round(float(overlap_seconds) * float(DIFFUSION_SR) * frames_per_sample)))
    full_mel = _assemble_mel_seamblend(out_mels, trim_frames, seam_blend_frames=56, max_shift_frames=28)
    full_t = torch.from_numpy(full_mel[None, None, :, :]).to(device)
    audio = np.asarray(vocode_bigvgan(full_t, float(mel_min), float(mel_max), vocoder, device), dtype=np.float32).reshape(-1)
    return audio


def _mix_preserved_vocals(vocal_path: Path, accompaniment: np.ndarray, out_dir: Path, vocal_gain: float, accomp_gain: float) -> Path:
    vocals = load_audio_chunk(vocal_path, sample_rate=DIFFUSION_SR, seconds=len(accompaniment) / float(DIFFUSION_SR), start_sec=0.0)
    vocals = _pad_audio(vocals, len(accompaniment))
    mix = np.clip(vocal_gain * vocals + accomp_gain * accompaniment, -1.0, 1.0).astype(np.float32)
    mix_path = out_dir / "hybrid_longform_coherent.wav"
    sf.write(str(mix_path), mix, DIFFUSION_SR)
    sf.write(str(out_dir / "backing_fixed.wav"), accompaniment, DIFFUSION_SR)
    sf.write(str(out_dir / "longform_coherent.wav"), accompaniment, DIFFUSION_SR)
    return mix_path


def benchmark_checkpoint(
    *,
    model: RetrievalFusionUNet,
    judge: MelGenreJudge,
    genre_to_idx: Dict[str, int],
    track_bank: Dict[int, List[Dict[str, Any]]],
    arrays: Dict[str, np.ndarray],
    mel_min: float,
    mel_max: float,
    max_frames: int,
    vocoder: Any,
    device: torch.device,
    seconds: float,
    out_dir: Path,
    single_genre_target: str = "",
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_cfg = HybridPushConfig()
    manifest_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    separation_vals: List[float] = []
    for song in picked_songs():
        stems = _resolve_stems(hybrid_cfg, song)
        source_key = _slug(Path(song["path"]).stem)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
        target_rows: List[Dict[str, Any]] = []
        targets = [str(single_genre_target)] if str(single_genre_target).strip() else list(TARGET_GENRES)
        for target in targets:
            target_idx = int(genre_to_idx[target])
            donor_track = _choose_donor_track(source_acc, track_bank, target_idx)
            render_dir = out_dir / "renders" / source_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp = generate_longform(
                model,
                source_audio=source_acc,
                target_genre_idx=target_idx,
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(mel_min),
                mel_max=float(mel_max),
                max_frames=int(max_frames),
                chunk_seconds=float(max(6.0, (float(max_frames) / 320.0) * 3.0)),
                overlap_seconds=float(max(3.0, ((float(max_frames) / 320.0) - 1.0) * 3.0)),
                vocoder=vocoder,
                device=device,
            )
            accomp = _pad_audio(accomp, len(source_acc))
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), accomp, DIFFUSION_SR)
            final_mix = _mix_preserved_vocals(stems["vocals"], accomp, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = _judge_probs_for_audio(accomp, judge, device, max_frames)
            tgt_conf = float(probs[target_idx])
            tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
            metrics = _audio_metrics(source_acc, accomp, DIFFUSION_SR)
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
            target_rows.append(row)
            all_rows.append(row)
            manifest_rows.append(
                {
                    "target": target,
                    "source_song": source_key,
                    "source_target_dir": str(render_dir),
                    "hybrid_wav": str(final_mix),
                    "accompaniment_wav": str(accomp_path),
                }
            )
        for i in range(len(target_rows)):
            for j in range(i + 1, len(target_rows)):
                pa = np.asarray(target_rows[i]["judge_probs"], dtype=np.float32)
                pb = np.asarray(target_rows[j]["judge_probs"], dtype=np.float32)
                separation_vals.append(float(np.mean(np.abs(pa - pb))))
    mean_sep = float(np.mean(separation_vals)) if separation_vals else 0.0
    for row in all_rows:
        row["separation"] = mean_sep
        row["overall"] = float(
            0.32 * row["target_margin"] +
            0.18 * row["target_conf"] +
            0.28 * row["fullness"] +
            0.22 * row["structure"] +
            0.20 * row["separation"] -
            0.20 * row["warble"]
        )
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    summary = {
        "mean_overall": float(np.mean([r["overall"] for r in all_rows])) if all_rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in all_rows])) if all_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in all_rows])) if all_rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in all_rows])) if all_rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in all_rows])) if all_rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in all_rows])) if all_rows else 0.0,
        "mean_separation": mean_sep,
        "rows": all_rows,
        "manifest": str(manifest_path),
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


@dataclass
class TrainConfig:
    cache_dir: Path = REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    baseline_pack: Path = REPO_ROOT / "Desktop Outputs" / "dggr_new_model_rounds" / "round_20260331_173856" / "compare_pack"
    judge_ckpt: Path = REPO_ROOT / "Desktop Outputs" / "dggr_per_genre_retrieval_suite" / "suite_20260331_214339" / "judge_compare" / "genre_judge.pt"
    out_root: Path = REPO_ROOT / "Desktop Outputs" / "dggr_retrieval_body_style_sequence_runs"
    epochs: int = 4
    batch_size: int = 4
    max_batches_per_epoch: int = 480
    val_batches: int = 10
    eval_every_steps: int = 120
    max_frames: int = 320
    base_ch: int = 64
    lr: float = 2.25e-4
    weight_decay: float = 1e-4
    seed: int = 328
    benchmark_seconds: float = 30.0
    final_seconds: float = 60.0
    device: str = "auto"
    single_genre_target: str = ""


def train(cfg: TrainConfig) -> Dict[str, Any]:
    device = _device_from_arg(str(cfg.device))
    out_dir = Path(cfg.out_root) / f"retrieval_body_style_seq_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(cfg))

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=int(cfg.seed))
    if str(cfg.single_genre_target).strip():
        if str(cfg.single_genre_target) not in genre_to_idx:
            raise ValueError(f"Unknown single_genre_target={cfg.single_genre_target}")
        keep = int(genre_to_idx[str(cfg.single_genre_target)])
        train_idx = train_idx[genre_idx[train_idx] == keep]
        val_idx = val_idx[genre_idx[val_idx] == keep]
        if len(train_idx) == 0:
            raise RuntimeError(f"No train samples for target genre {cfg.single_genre_target}")
        if len(val_idx) == 0:
            val_idx = train_idx[: min(len(train_idx), 64)]
    train_ds = RetrievalFusionDataset(
        arrays,
        index_df,
        train_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        seed=int(cfg.seed),
    )
    val_ds = RetrievalFusionDataset(
        arrays,
        index_df,
        val_idx,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        seed=int(cfg.seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=_make_balanced_sampler(train_ds, int(cfg.seed)), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    judge, judge_genre_to_idx = _load_or_train_judge(Path(cfg.judge_ckpt), Path(cfg.cache_dir), out_dir, device, int(cfg.max_frames))
    if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
        raise RuntimeError(f"Judge genre mismatch: {judge_genre_to_idx} vs {genre_to_idx}")

    model = RetrievalFusionUNet(in_ch=18, num_genres=len(genre_to_idx), base_ch=int(cfg.base_ch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    track_bank = _build_track_bank(index_df, arrays, train_idx)
    vocoder = load_bigvgan_robust(device=device)

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
                "cfg": asdict(cfg),
                "genre_to_idx": genre_to_idx,
                "epoch": int(epoch),
                "global_step": int(global_step),
                "meta": {"mel_min": float(meta.mel_min), "mel_max": float(meta.mel_max)},
            },
            str(path),
        )

    def _run_eval(tag: str, epoch: int) -> Dict[str, Any]:
        nonlocal best_score
        summary = benchmark_checkpoint(
            model=model,
            judge=judge,
            genre_to_idx=genre_to_idx,
            track_bank=track_bank,
            arrays=arrays,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(cfg.max_frames),
            vocoder=vocoder,
            device=device,
            seconds=float(cfg.benchmark_seconds),
            out_dir=out_dir / "benchmark" / tag,
            single_genre_target=str(cfg.single_genre_target),
        )
        if float(summary["mean_overall"]) > float(best_score):
            best_score = float(summary["mean_overall"])
            _save_ckpt(best_ckpt, epoch)
            _write_json(out_dir / "winner_map.json", {"best_tag": tag, "best_checkpoint": str(best_ckpt), "best_score": float(best_score)})
        return summary

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_l1: List[float] = []
        train_judge: List[float] = []
        train_timbre: List[float] = []
        train_body: List[float] = []
        train_groove: List[float] = []
        train_cont: List[float] = []
        train_cont2: List[float] = []
        start_t = time.time()
        for batch_idx, batch in enumerate(train_loader):
            if int(cfg.max_batches_per_epoch) > 0 and batch_idx >= int(cfg.max_batches_per_epoch):
                break
            target = batch["target_mel"].to(device)
            donor = batch["donor_mel"].to(device)
            prev_mel = batch["prev_mel"].to(device)
            prev2_mel = batch["prev2_mel"].to(device)
            struct = batch["struct_mel"].to(device)
            cond_feat = batch["cond_feat"].to(device)
            genre = batch["genre_idx"].to(device)
            context_count = batch["context_count"].to(device)
            pred = model(struct, donor, prev_mel, prev2_mel, cond_feat, genre, context_count)
            loss_l1 = F.l1_loss(pred, target)
            loss_dt = F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
            loss_df = F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :])
            loss_judge = F.cross_entropy(judge(pred), genre)
            loss_timbre = F.l1_loss(_mel_timbre_stats(pred[:, :, 24:, :]), _mel_timbre_stats(donor[:, :, 24:, :]))
            loss_body = F.l1_loss(_band_profile(pred, 6, 48), _band_profile(target, 6, 48))
            loss_groove = F.l1_loss(_band_envelope(pred, 6, 64), _band_envelope(target, 6, 64))
            tail_cols = min(32, target.shape[-1], prev_mel.shape[-1])
            pred_lead = pred[:, :, :, :tail_cols]
            prev_tail = prev_mel[:, :, :, -tail_cols:]
            prev2_tail = prev2_mel[:, :, :, -tail_cols:]
            target_lead = target[:, :, :, :tail_cols]
            target_tail = target[:, :, :, -tail_cols:]
            loss_cont = (
                context_count.mean() * F.l1_loss(pred_lead, prev_tail) +
                0.5 * F.l1_loss(pred_lead, target_lead) +
                0.25 * F.l1_loss(pred[:, :, :, -tail_cols:], target_tail)
            )
            loss_cont2 = 0.5 * context_count.mean() * F.l1_loss(prev_tail - prev2_tail, pred_lead - prev_tail)
            loss = (
                0.90 * loss_l1 +
                0.28 * loss_dt +
                0.20 * loss_df +
                0.24 * loss_judge +
                0.14 * loss_timbre +
                0.28 * loss_body +
                0.16 * loss_groove +
                0.22 * loss_cont +
                0.10 * loss_cont2
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_l1.append(float(loss_l1.item()))
            train_judge.append(float(loss_judge.item()))
            train_timbre.append(float(loss_timbre.item()))
            train_body.append(float(loss_body.item()))
            train_groove.append(float(loss_groove.item()))
            train_cont.append(float(loss_cont.item()))
            train_cont2.append(float(loss_cont2.item()))
            global_step += 1
            if int(cfg.eval_every_steps) > 0 and global_step % int(cfg.eval_every_steps) == 0:
                _save_ckpt(ckpt_dir / "latest.pt", epoch)
                bench = _run_eval(f"step_{global_step:05d}", epoch)
                history.append(
                    {
                        "epoch": int(epoch),
                        "global_step": int(global_step),
                        "tag": f"step_{global_step:05d}",
                        "train_l1": float(np.mean(train_l1)) if train_l1 else 0.0,
                        "train_judge": float(np.mean(train_judge)) if train_judge else 0.0,
                        "train_timbre": float(np.mean(train_timbre)) if train_timbre else 0.0,
                        "train_body": float(np.mean(train_body)) if train_body else 0.0,
                        "train_groove": float(np.mean(train_groove)) if train_groove else 0.0,
                        "train_cont": float(np.mean(train_cont)) if train_cont else 0.0,
                        "train_cont2": float(np.mean(train_cont2)) if train_cont2 else 0.0,
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
        val_l1: List[float] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= int(cfg.val_batches):
                    break
                pred = model(
                    batch["struct_mel"].to(device),
                    batch["donor_mel"].to(device),
                    batch["prev_mel"].to(device),
                    batch["prev2_mel"].to(device),
                    batch["cond_feat"].to(device),
                    batch["genre_idx"].to(device),
                    batch["context_count"].to(device),
                )
                val_l1.append(float(F.l1_loss(pred, batch["target_mel"].to(device)).item()))

        _save_ckpt(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch)
        _save_ckpt(ckpt_dir / "latest.pt", epoch)
        bench = _run_eval(f"epoch_{epoch:03d}", epoch)
        history.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "tag": f"epoch_{epoch:03d}",
                "train_l1": float(np.mean(train_l1)) if train_l1 else 0.0,
                "train_judge": float(np.mean(train_judge)) if train_judge else 0.0,
                "train_timbre": float(np.mean(train_timbre)) if train_timbre else 0.0,
                "train_body": float(np.mean(train_body)) if train_body else 0.0,
                "train_groove": float(np.mean(train_groove)) if train_groove else 0.0,
                "train_cont": float(np.mean(train_cont)) if train_cont else 0.0,
                "train_cont2": float(np.mean(train_cont2)) if train_cont2 else 0.0,
                "val_l1": float(np.mean(val_l1)) if val_l1 else 0.0,
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

    if not best_ckpt.exists():
        raise RuntimeError("No best checkpoint produced.")

    payload = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    final_pack = out_dir / "final_pack"
    final_summary = benchmark_checkpoint(
        model=model,
        judge=judge,
        genre_to_idx=genre_to_idx,
        track_bank=track_bank,
        arrays=arrays,
        mel_min=float(meta.mel_min),
        mel_max=float(meta.mel_max),
        max_frames=int(cfg.max_frames),
        vocoder=vocoder,
        device=device,
        seconds=float(cfg.final_seconds),
        out_dir=final_pack,
        single_genre_target=str(cfg.single_genre_target),
    )
    report_lines = [
        "# Scratch Retrieval Body-Style Sequence Diagnosis",
        "",
        "- Old limitation: source-conditioned translation stayed too close to source timbre and produced filtered, samey instrumentation.",
        "- Prior retrieval family moved style but still came out too thin and body-poor.",
        "- New family: retrieval-conditioned accompaniment generator from scratch using source structure, donor texture, explicit body/groove losses, and blockwise sequence conditioning over previous windows.",
        f"- Best checkpoint: {best_ckpt}",
        f"- Final mean overall: {final_summary['mean_overall']:.4f}",
        f"- Final mean target confidence: {final_summary['mean_target_conf']:.4f}",
        f"- Final mean target margin: {final_summary['mean_target_margin']:.4f}",
        f"- Final mean separation: {final_summary['mean_separation']:.4f}",
        f"- Final mean fullness: {final_summary['mean_fullness']:.4f}",
        f"- Final mean warble: {final_summary['mean_warble']:.4f}",
    ]
    (out_dir / "diagnosis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {
        "out_dir": str(out_dir),
        "best_checkpoint": str(best_ckpt),
        "final_pack_dir": str(final_pack),
        "history_rows": history,
        "final_summary": final_summary,
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a scratch retrieval-conditioned accompaniment generator with blockwise sequence conditioning and benchmark it.")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "Desktop Outputs" / "dggr_retrieval_body_style_block_runs")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--baseline-pack", type=Path, default=Path.home() / "Desktop" / "dggr_new_model_rounds" / "round_20260331_173856" / "compare_pack")
    ap.add_argument("--judge-ckpt", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-batches-per-epoch", type=int, default=480)
    ap.add_argument("--eval-every-steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=960)
    ap.add_argument("--benchmark-seconds", type=float, default=30.0)
    ap.add_argument("--final-seconds", type=float, default=60.0)
    ap.add_argument("--single-genre-target", type=str, default="")
    args = ap.parse_args()

    cfg = TrainConfig(
        out_root=Path(args.out_root),
        cache_dir=Path(args.cache_dir),
        baseline_pack=Path(args.baseline_pack),
        judge_ckpt=Path(args.judge_ckpt),
        epochs=int(args.epochs),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        eval_every_steps=int(args.eval_every_steps),
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
        benchmark_seconds=float(args.benchmark_seconds),
        final_seconds=float(args.final_seconds),
        single_genre_target=str(args.single_genre_target),
    )
    summary = train(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
