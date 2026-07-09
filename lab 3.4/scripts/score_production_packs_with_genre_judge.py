from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.lab3_diffusion_data import extract_bigvgan_mel_np, load_diffusion_cache, pad_or_trim
from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_data import stratified_group_split_indices
from dggr.lab3_diffusion_data import DIFFUSION_SR
from train_scratch_structure_diffusion import MelGenreJudge, _audio_metrics, _normalize_mel_np, _slug


class MelOnlyDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray, mel_min: float, mel_max: float, max_frames: int):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.max_frames = int(max_frames)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        idx = int(self.indices[int(i)])
        mel = np.asarray(self.arrays["mel"][idx], dtype=np.float32)[:, : self.max_frames]
        mel_norm = _normalize_mel_np(mel, self.mel_min, self.mel_max)
        genre = int(np.asarray(self.arrays["genre_idx"], dtype=np.int64)[idx])
        return {"mel": torch.from_numpy(mel_norm[None, :, :]), "genre_idx": torch.tensor(genre, dtype=torch.long)}

    def genre_indices(self) -> np.ndarray:
        return np.asarray(self.arrays["genre_idx"], dtype=np.int64)[self.indices]


def _make_balanced_sampler(ds: MelOnlyDataset, seed: int) -> WeightedRandomSampler:
    genre = ds.genre_indices()
    uniq, counts = np.unique(genre, return_counts=True)
    inv = {int(g): 1.0 / float(c) for g, c in zip(uniq.tolist(), counts.tolist())}
    weights = np.asarray([inv[int(g)] for g in genre.tolist()], dtype=np.float64)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True, generator=gen)


def _train_judge(cache_dir: Path, device: torch.device, out_dir: Path, max_frames: int = 256, steps: int = 300) -> Tuple[MelGenreJudge, Dict[str, int]]:
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(cache_dir, mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    train_idx, val_idx = stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=328)
    train_ds = MelOnlyDataset(arrays, train_idx, float(meta.mel_min), float(meta.mel_max), max_frames=max_frames)
    val_ds = MelOnlyDataset(arrays, val_idx, float(meta.mel_min), float(meta.mel_max), max_frames=max_frames)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=_make_balanced_sampler(train_ds, 328), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    judge = MelGenreJudge(num_genres=len(genre_to_idx)).to(device)
    opt = torch.optim.AdamW(judge.parameters(), lr=3e-4, weight_decay=1e-4)
    train_iter = iter(train_loader)
    for step in range(int(steps)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        mel = batch["mel"].to(device)
        genre = batch["genre_idx"].to(device)
        logits = judge(mel)
        loss = F.cross_entropy(logits, genre)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    judge.eval()
    val_acc = []
    with torch.no_grad():
        for batch in val_loader:
            mel = batch["mel"].to(device)
            genre = batch["genre_idx"].to(device)
            pred = judge(mel).argmax(dim=-1)
            val_acc.append(float((pred == genre).float().mean().item()))
    summary = {"val_acc": float(np.mean(val_acc)) if val_acc else 0.0, "steps": int(steps)}
    (out_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"judge": judge.state_dict(), "genre_to_idx": genre_to_idx}, str(out_dir / "genre_judge.pt"))
    return judge, genre_to_idx


@torch.no_grad()
def _probs_for_audio(audio_path: Path, judge: MelGenreJudge, device: torch.device, max_frames: int = 256) -> np.ndarray:
    audio = load_audio_chunk(audio_path, sample_rate=DIFFUSION_SR, seconds=60.0, start_sec=0.0)
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


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = [part.strip() for part in lines[0].split(",")]
    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == len(header):
            rows.append(dict(zip(header, parts)))
            continue
        # Some legacy manifests were written without CSV quoting, so commas in the
        # source song name expand the row width. Reconstruct them from the fixed schema.
        if header == [
            "job_idx",
            "setting_label",
            "source_audio",
            "target_genre",
            "output_dir",
            "generated_wav",
            "final_mix_wav",
        ] and len(parts) >= 7:
            rows.append(
                {
                    "job_idx": parts[0],
                    "setting_label": parts[1],
                    "source_audio": ",".join(parts[2:-4]).strip(),
                    "target_genre": parts[-4],
                    "output_dir": parts[-3],
                    "generated_wav": parts[-2],
                    "final_mix_wav": parts[-1],
                }
            )
            continue
        raise ValueError(f"Could not parse manifest row in {manifest_path}: {line}")
    return rows


def _score_pack(rows: List[Dict[str, str]], judge: MelGenreJudge, genre_to_idx: Dict[str, int], device: torch.device, out_path: Path) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    all_rows: List[Dict[str, Any]] = []
    for row in rows:
        target = row.get("target") or row.get("target_genre")
        song = row.get("source_song") or _slug(Path(str(row.get("source_audio", ""))).stem)
        hybrid = Path(str(row.get("hybrid_wav") or row.get("final_mix_wav")))
        accomp = Path(str(row.get("accompaniment_wav") or row.get("generated_wav") or ""))
        probs = _probs_for_audio(hybrid if hybrid.exists() else accomp, judge, device)
        tgt_idx = int(genre_to_idx[str(target)])
        target_conf = float(probs[tgt_idx])
        others = np.delete(probs, tgt_idx)
        target_margin = float(target_conf - float(np.max(others)))
        item = {
            "song": song,
            "target": str(target),
            "hybrid_wav": str(hybrid),
            "accompaniment_wav": str(accomp),
            "target_conf": target_conf,
            "target_margin": target_margin,
            "judge_probs": probs.tolist(),
        }
        grouped.setdefault(song, []).append(item)
        all_rows.append(item)

    seps: List[float] = []
    for song, items in grouped.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pa = np.asarray(items[i]["judge_probs"], dtype=np.float32)
                pb = np.asarray(items[j]["judge_probs"], dtype=np.float32)
                seps.append(float(np.mean(np.abs(pa - pb))))
    sep = float(np.mean(seps)) if seps else 0.0
    summary = {
        "n_rows": len(all_rows),
        "mean_target_conf": float(np.mean([r["target_conf"] for r in all_rows])) if all_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in all_rows])) if all_rows else 0.0,
        "mean_separation": sep,
        "rows": all_rows,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a unified genre judge and score two production packs.")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--suite-manifest", type=Path, required=True)
    ap.add_argument("--baseline-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge, genre_to_idx = _train_judge(Path(args.cache_dir), device, out_dir, max_frames=256, steps=300)
    suite_rows = _load_manifest_rows(Path(args.suite_manifest))
    base_rows = _load_manifest_rows(Path(args.baseline_manifest))
    suite_summary = _score_pack(suite_rows, judge, genre_to_idx, device, out_dir / "suite_score.json")
    base_summary = _score_pack(base_rows, judge, genre_to_idx, device, out_dir / "baseline_score.json")
    summary = {"suite": suite_summary, "baseline": base_summary}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
