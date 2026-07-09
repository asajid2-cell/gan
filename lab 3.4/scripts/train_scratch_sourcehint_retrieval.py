from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_scratch_retrieval_fusion import (
    REPO_ROOT,
    RetrievalFusionUNet,
    _build_track_bank,
    _choose_donor_track,
    _device_from_arg,
    _judge_probs_for_audio,
    _load_or_train_judge,
    _mel_timbre_stats,
    _mix_preserved_vocals,
    _normalize_mel_np,
    _slug,
    _structure_proxy_from_mel,
    _write_json,
    benchmark_checkpoint,
)
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import load_diffusion_cache
from dggr.lab3_diffusion_train import load_bigvgan_robust


def _source_hint_from_mel(mel_norm: np.ndarray) -> np.ndarray:
    mel_t = torch.from_numpy(mel_norm[None, None, :, :]).float()
    coarse = F.avg_pool2d(mel_t, kernel_size=(9, 9), stride=1, padding=(4, 4))[0, 0].cpu().numpy().astype(np.float32)
    out = coarse.copy()
    out[:24, :] = 0.75 * mel_norm[:24, :] + 0.25 * out[:24, :]
    out[24:56, :] = 0.45 * mel_norm[24:56, :] + 0.55 * out[24:56, :]
    hf_mean = out[56:, :].mean(axis=0, keepdims=True)
    out[56:, :] = 0.20 * mel_norm[56:, :] + 0.80 * np.repeat(hf_mean, out.shape[0] - 56, axis=0)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


class SourceHintRetrievalDataset(Dataset):
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
        self.rng = np.random.default_rng(int(seed))
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
        target_mel = np.asarray(self.arrays["mel"][idx], dtype=np.float32)[:, : self.max_frames]
        donor_mel = np.asarray(self.arrays["mel"][donor_idx], dtype=np.float32)[:, : self.max_frames]
        target_norm = _normalize_mel_np(target_mel, self.mel_min, self.mel_max)
        donor_norm = _normalize_mel_np(donor_mel, self.mel_min, self.mel_max)
        source_hint = _source_hint_from_mel(target_norm)
        struct = _structure_proxy_from_mel(target_norm)
        chroma = np.asarray(self.arrays["chroma"][idx], dtype=np.float32)[:, : self.max_frames]
        onset = np.asarray(self.arrays["onset"][idx], dtype=np.float32)[: self.max_frames]
        beat = np.asarray(self.arrays["beat"][idx], dtype=np.float32)[: self.max_frames]
        h = target_norm.shape[0]
        cond_feat = np.concatenate(
            [
                np.repeat(chroma[:, None, :], h, axis=1),
                np.repeat(onset[None, None, :], h, axis=1),
                np.repeat(beat[None, None, :], h, axis=1),
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "target_mel": torch.from_numpy(target_norm[None, :, :]),
            "donor_mel": torch.from_numpy(donor_norm[None, :, :]),
            "source_hint": torch.from_numpy(source_hint[None, :, :]),
            "struct_mel": torch.from_numpy(struct[None, :, :]),
            "cond_feat": torch.from_numpy(cond_feat),
            "genre_idx": torch.tensor(genre, dtype=torch.long),
        }


class SourceHintRetrievalUNet(nn.Module):
    def __init__(self, num_genres: int, base_ch: int = 48) -> None:
        super().__init__()
        self.core = RetrievalFusionUNet(in_ch=17, num_genres=num_genres, base_ch=base_ch)
        self.out_head = nn.Conv2d(1, 2, 3, padding=1)

    def forward(
        self,
        source_hint: torch.Tensor,
        struct_mel: torch.Tensor,
        donor_mel: torch.Tensor,
        cond_feat: torch.Tensor,
        genre_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([source_hint, struct_mel, donor_mel, cond_feat], dim=1)
        base = self.core.enc1(x)
        e2 = self.core.enc2(self.core.down1(base))
        h = self.core.mid(self.core.down2(e2))
        donor_vec = self.core.donor_head(donor_mel).flatten(1)
        film = self.core.film(torch.cat([donor_vec, self.core.genre_emb(genre_idx)], dim=-1))
        scale, bias = film.chunk(2, dim=-1)
        h = h * (1.0 + 0.1 * scale[:, :, None, None]) + 0.1 * bias[:, :, None, None]
        h = self.core.up1(h)
        h = self.core.dec1(torch.cat([h, e2], dim=1))
        h = self.core.up2(h)
        h = self.core.dec2(torch.cat([h, base], dim=1))
        raw = self.core.out(h)
        head = self.out_head(raw)
        residual, gate_logits = head.chunk(2, dim=1)
        proposal = torch.tanh(source_hint + 0.70 * torch.tanh(residual))
        gate = torch.sigmoid(gate_logits)
        return gate * proposal + (1.0 - gate) * source_hint


@torch.no_grad()
def generate_longform_sourcehint(
    model: SourceHintRetrievalUNet,
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
    from train_scratch_retrieval_fusion import _cond_feat_from_audio, assemble_audio_crossfade, split_audio_overlapping
    from dggr.lab3_diffusion_data import extract_bigvgan_mel_np, pad_or_trim
    from dggr.lab3_diffusion_train import vocode_bigvgan
    from train_scratch_structure_diffusion import smooth_mel_tensor

    model.eval()
    chunks = split_audio_overlapping(source_audio, chunk_seconds=float(chunk_seconds), overlap_seconds=float(overlap_seconds), sr=22050)
    donor_rows = donor_track["rows"]
    total = max(1, len(chunks) - 1)
    out_wavs = []
    for i, chunk in enumerate(chunks):
        mel = extract_bigvgan_mel_np(chunk["audio"], sr=22050)
        mel = pad_or_trim(mel, max_frames, axis=1, pad_val=float(mel_min))
        mel_norm = _normalize_mel_np(mel, mel_min, mel_max)
        source_hint = _source_hint_from_mel(mel_norm)
        struct = _structure_proxy_from_mel(mel_norm)
        donor_idx = int(donor_rows[min(len(donor_rows) - 1, round((i / total) * max(0, len(donor_rows) - 1)))])
        donor_mel = np.asarray(arrays["mel"][donor_idx], dtype=np.float32)[:, :max_frames]
        donor_norm = _normalize_mel_np(donor_mel, mel_min, mel_max)
        cond_feat = _cond_feat_from_audio(chunk["audio"], max_frames, mel_norm.shape[0])
        pred = model(
            torch.from_numpy(source_hint[None, None, :, :]).to(device),
            torch.from_numpy(struct[None, None, :, :]).to(device),
            torch.from_numpy(donor_norm[None, None, :, :]).to(device),
            torch.from_numpy(cond_feat[None, :, :, :]).to(device),
            torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device),
        )
        pred_mel = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_mel[:10, :] = 0.65 * mel_norm[:10, :] + 0.35 * pred_mel[:10, :]
        if i > 0:
            pred_mel = smooth_mel_tensor(torch.from_numpy(pred_mel[None, None, :, :]), time_kernel=5, freq_kernel=3)[0, 0].cpu().numpy().astype(np.float32)
        audio = np.asarray(vocode_bigvgan(torch.from_numpy(pred_mel[None, None, :, :]).to(device), float(mel_min), float(mel_max), vocoder, device), dtype=np.float32).reshape(-1)
        out_wavs.append(audio)
    return assemble_audio_crossfade(out_wavs, overlap_seconds=float(overlap_seconds), sr=22050)


def benchmark_checkpoint_sourcehint(
    *,
    model: SourceHintRetrievalUNet,
    judge: Any,
    genre_to_idx: Dict[str, int],
    track_bank: Dict[int, list[Dict[str, Any]]],
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
    from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _resolve_stems, picked_songs
    from dggr.lab3_bridge import load_audio_chunk
    from dggr.lab3_diffusion_data import DIFFUSION_SR
    from train_scratch_structure_diffusion import _audio_metrics

    out_dir.mkdir(parents=True, exist_ok=True)
    hybrid_cfg = HybridPushConfig()
    manifest_rows = []
    rows = []
    separation_vals = []
    for song in picked_songs():
        stems = _resolve_stems(hybrid_cfg, song)
        source_key = _slug(Path(song["path"]).stem)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=float(seconds), start_sec=0.0)
        per_song = []
        targets = [str(single_genre_target)] if str(single_genre_target).strip() else list(TARGET_GENRES)
        for target in targets:
            target_idx = int(genre_to_idx[target])
            donor_track = _choose_donor_track(source_acc, track_bank, target_idx)
            render_dir = out_dir / "renders" / source_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp = generate_longform_sourcehint(
                model,
                source_audio=source_acc,
                target_genre_idx=target_idx,
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(mel_min),
                mel_max=float(mel_max),
                max_frames=int(max_frames),
                chunk_seconds=3.0,
                overlap_seconds=0.5,
                vocoder=vocoder,
                device=device,
            )
            accomp = accomp[: len(source_acc)]
            accomp_path = render_dir / "accompaniment_generated.wav"
            import soundfile as sf
            sf.write(str(accomp_path), accomp, DIFFUSION_SR)
            final_mix = _mix_preserved_vocals(stems["vocals"], accomp, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = _judge_probs_for_audio(accomp, judge, device, max_frames)
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
    import csv
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    for row in rows:
        row["separation"] = mean_sep
        row["overall"] = float(0.45 * row["target_margin"] + 0.20 * row["target_conf"] + 0.18 * row["fullness"] + 0.20 * row["structure"] + 0.22 * row["separation"] - 0.20 * row["warble"])
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
    out_root: Path = Path.home() / "Desktop" / "dggr_sourcehint_retrieval_runs"
    epochs: int = 2
    batch_size: int = 4
    max_batches_per_epoch: int = 180
    val_batches: int = 10
    eval_every_steps: int = 60
    max_frames: int = 320
    base_ch: int = 48
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 328
    benchmark_seconds: float = 30.0
    final_seconds: float = 60.0
    device: str = "auto"
    single_genre_target: str = ""


def train(cfg: TrainConfig) -> Dict[str, Any]:
    device = _device_from_arg(str(cfg.device))
    out_dir = Path(cfg.out_root) / f"sourcehint_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "config.json", asdict(cfg))

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

    train_ds = SourceHintRetrievalDataset(arrays, index_df, train_idx, mel_min=float(meta.mel_min), mel_max=float(meta.mel_max), max_frames=int(cfg.max_frames), seed=int(cfg.seed))
    val_ds = SourceHintRetrievalDataset(arrays, index_df, val_idx, mel_min=float(meta.mel_min), mel_max=float(meta.mel_max), max_frames=int(cfg.max_frames), seed=int(cfg.seed + 1))
    from train_scratch_structure_diffusion import _make_balanced_sampler
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), sampler=_make_balanced_sampler(train_ds, int(cfg.seed)), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=0)

    judge, judge_genre_to_idx = _load_or_train_judge(Path(cfg.judge_ckpt), Path(cfg.cache_dir), out_dir, device, int(cfg.max_frames))
    if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
        raise RuntimeError("Judge genre mismatch")

    model = SourceHintRetrievalUNet(num_genres=len(genre_to_idx), base_ch=int(cfg.base_ch)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    track_bank = _build_track_bank(index_df, arrays, train_idx)
    vocoder = load_bigvgan_robust(device=device)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1e18
    global_step = 0
    best_ckpt = ckpt_dir / "best_by_judge.pt"

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
        bench = benchmark_checkpoint_sourcehint(
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
        if float(bench["mean_overall"]) > float(best_score):
            best_score = float(bench["mean_overall"])
            _save_ckpt(best_ckpt, epoch)
            _write_json(out_dir / "winner_map.json", {"best_tag": tag, "best_checkpoint": str(best_ckpt), "best_score": float(best_score)})
        return bench

    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_l1 = []
        train_judge = []
        train_timbre = []
        train_low = []
        start_t = time.time()
        for batch_idx, batch in enumerate(train_loader):
            if int(cfg.max_batches_per_epoch) > 0 and batch_idx >= int(cfg.max_batches_per_epoch):
                break
            target = batch["target_mel"].to(device)
            donor = batch["donor_mel"].to(device)
            source_hint = batch["source_hint"].to(device)
            struct = batch["struct_mel"].to(device)
            cond_feat = batch["cond_feat"].to(device)
            genre = batch["genre_idx"].to(device)
            pred = model(source_hint, struct, donor, cond_feat, genre)
            loss_l1 = F.l1_loss(pred, target)
            loss_dt = F.l1_loss(pred[:, :, :, 1:] - pred[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1])
            loss_df = F.l1_loss(pred[:, :, 1:, :] - pred[:, :, :-1, :], target[:, :, 1:, :] - target[:, :, :-1, :])
            loss_low = F.l1_loss(pred[:, :, :36, :], target[:, :, :36, :])
            loss_judge = F.cross_entropy(judge(pred), genre)
            loss_timbre = F.l1_loss(_mel_timbre_stats(pred), _mel_timbre_stats(donor))
            loss = loss_l1 + 0.22 * loss_dt + 0.16 * loss_df + 0.30 * loss_judge + 0.14 * loss_timbre + 0.28 * loss_low
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_l1.append(float(loss_l1.item()))
            train_judge.append(float(loss_judge.item()))
            train_timbre.append(float(loss_timbre.item()))
            train_low.append(float(loss_low.item()))
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
                        "train_low": float(np.mean(train_low)) if train_low else 0.0,
                        "train_judge": float(np.mean(train_judge)) if train_judge else 0.0,
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
        val_l1 = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= int(cfg.val_batches):
                    break
                pred = model(batch["source_hint"].to(device), batch["struct_mel"].to(device), batch["donor_mel"].to(device), batch["cond_feat"].to(device), batch["genre_idx"].to(device))
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
                "train_low": float(np.mean(train_low)) if train_low else 0.0,
                "train_judge": float(np.mean(train_judge)) if train_judge else 0.0,
                "train_timbre": float(np.mean(train_timbre)) if train_timbre else 0.0,
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

    payload = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    final_pack = out_dir / "final_pack"
    final_summary = benchmark_checkpoint_sourcehint(
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
        "# Scratch SourceHint Retrieval Diagnosis",
        "",
        "- Hypothesis: the pure retrieval model wins on genre separation but loses too much fullness because structure conditioning is too coarse.",
        "- New family: source-hint + donor retrieval generator trained from scratch.",
        f"- Best checkpoint: {best_ckpt}",
        f"- Final mean overall: {final_summary['mean_overall']:.4f}",
        f"- Final mean target confidence: {final_summary['mean_target_conf']:.4f}",
        f"- Final mean target margin: {final_summary['mean_target_margin']:.4f}",
        f"- Final mean fullness: {final_summary['mean_fullness']:.4f}",
        f"- Final mean structure: {final_summary['mean_structure']:.4f}",
        f"- Final mean warble: {final_summary['mean_warble']:.4f}",
    ]
    (out_dir / "diagnosis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    summary = {"out_dir": str(out_dir), "best_checkpoint": str(best_ckpt), "final_pack_dir": str(final_pack), "history_rows": history, "final_summary": final_summary}
    _write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a scratch source-hint retrieval accompaniment generator and benchmark it.")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_sourcehint_retrieval_runs")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--judge-ckpt", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-batches-per-epoch", type=int, default=180)
    ap.add_argument("--eval-every-steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=320)
    ap.add_argument("--benchmark-seconds", type=float, default=30.0)
    ap.add_argument("--final-seconds", type=float, default=60.0)
    ap.add_argument("--single-genre-target", type=str, default="")
    args = ap.parse_args()
    cfg = TrainConfig(
        out_root=Path(args.out_root),
        cache_dir=Path(args.cache_dir),
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
