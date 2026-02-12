from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch

from .lab3_bridge import (
    FrozenLab1Encoder,
    denormalize_log_mel,
    extract_log_mel,
    fix_log_mel_frames,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import soundfile as sf

    HAS_SF = True
except Exception:
    HAS_SF = False


RUN_NAME_RE = re.compile(r"^run(\d+)$")


def resolve_next_run_name(out_root: Path) -> str:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for d in out_root.iterdir():
        if not d.is_dir():
            continue
        m = RUN_NAME_RE.match(d.name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"run{max_id + 1}"


def validate_run_name(
    run_name: str,
    strict_run_naming: bool = True,
    force_custom_run_name: bool = False,
) -> str:
    name = str(run_name).strip()
    if not name:
        return name
    if not strict_run_naming:
        return name
    if RUN_NAME_RE.match(name):
        return name
    if force_custom_run_name:
        return name
    raise ValueError(
        f"Invalid run name '{name}' with strict naming enabled. "
        "Use runN (e.g., run1) or pass --force-custom-run-name."
    )


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))


def _mel_norm_to_db_np(mel_norm: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(mel_norm).unsqueeze(0).float()
    return denormalize_log_mel(t).squeeze(0).cpu().numpy().astype(np.float32)


def _mel_db_to_audio(mel_db: np.ndarray, sr: int, griffin_lim_iters: int) -> np.ndarray:
    mel_power = librosa.db_to_power(mel_db)
    y = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=int(sr),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fmin=20,
        fmax=int(sr // 2),
        n_iter=int(griffin_lim_iters),
    )
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-8)
    return y.astype(np.float32)


def _audio_to_mel_db(y: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    mel = extract_log_mel(y.astype(np.float32), sr=int(sr))
    mel = fix_log_mel_frames(mel, n_frames=int(n_frames))
    return mel.astype(np.float32)


def build_balanced_generation_index(
    genre_idx: np.ndarray,
    val_idx: np.ndarray,
    genre_to_idx: Dict[str, int],
    n_samples: int,
    seed: int = 328,
    target_mode: str = "balanced_random",
) -> pd.DataFrame:
    if len(val_idx) == 0:
        raise ValueError("val_idx is empty; cannot build generation index.")
    rng = np.random.default_rng(int(seed))
    idx_to_genre = {int(v): str(k) for k, v in genre_to_idx.items()}
    genres = sorted(idx_to_genre.keys())
    by_src: Dict[int, List[int]] = {}
    for g in genres:
        g_idx = val_idx[genre_idx[val_idx] == g].astype(np.int64).tolist()
        if g_idx:
            rng.shuffle(g_idx)
            by_src[g] = g_idx
    src_genres = sorted(by_src.keys())
    if not src_genres:
        raise ValueError("No validation samples found for any source genre.")

    ptr = {g: 0 for g in src_genres}

    pairs: List[tuple[int, int]] = []
    if str(target_mode).lower() == "balanced_random":
        base_pairs = [(s, t) for s in src_genres for t in genres if t != s]
        if not base_pairs:
            base_pairs = [(s, s) for s in src_genres]
        reps = int(math.ceil(float(n_samples) / float(len(base_pairs))))
        pairs = list(base_pairs) * reps
        rng.shuffle(pairs)
        pairs = pairs[: int(n_samples)]
    elif str(target_mode).lower() == "round_robin":
        t_all = genres[:]
        for i in range(int(n_samples)):
            s = src_genres[i % len(src_genres)]
            t = t_all[i % len(t_all)]
            if t == s and len(t_all) > 1:
                t = t_all[(i + 1) % len(t_all)]
            pairs.append((s, t))
    else:
        # random
        for i in range(int(n_samples)):
            s = src_genres[i % len(src_genres)]
            t = int(rng.choice(genres))
            if t == s and len(genres) > 1:
                others = [x for x in genres if x != s]
                t = int(rng.choice(others))
            pairs.append((s, t))

    rows: List[Dict] = []
    for i, (s, t) in enumerate(pairs):
        src_pool = by_src[s]
        ridx = int(src_pool[ptr[s] % len(src_pool)])
        ptr[s] += 1
        rows.append(
            {
                "sample_id": int(i),
                "cache_row": ridx,
                "source_genre_idx": int(s),
                "target_genre_idx": int(t),
                "source_genre": idx_to_genre[int(s)],
                "target_genre": idx_to_genre[int(t)],
            }
        )
    return pd.DataFrame(rows)


def export_posttrain_samples(
    generator,
    frozen_encoder: FrozenLab1Encoder,
    arrays: Dict[str, np.ndarray],
    index_df: pd.DataFrame,
    genre_to_idx: Dict[str, int],
    cond_bank: torch.Tensor,
    out_dir: Path,
    val_idx: np.ndarray,
    n_samples: int = 100,
    target_mode: str = "balanced_random",
    griffin_lim_iters: int = 48,
    seed: int = 328,
    device: str = "cpu",
    genre_to_source_idx: Optional[Dict[str, int]] = None,
    write_real_audio: bool = True,
) -> Dict[str, str]:
    if not HAS_SF:
        raise RuntimeError("soundfile is required for post-train sample export.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_df = build_balanced_generation_index(
        genre_idx=arrays["genre_idx"],
        val_idx=val_idx,
        genre_to_idx=genre_to_idx,
        n_samples=int(n_samples),
        seed=int(seed),
        target_mode=target_mode,
    )
    idx_to_genre = {int(v): str(k) for k, v in genre_to_idx.items()}
    device_t = str(device)
    if device_t == "auto":
        device_t = "cuda" if torch.cuda.is_available() else "cpu"

    generator.eval()
    generator = generator.to(device_t)
    cond_bank = cond_bank.to(device_t)

    sr = int(frozen_encoder.cfg.sample_rate)
    recs: List[Dict] = []

    for row in plan_df.itertuples(index=False):
        sample_id = int(row.sample_id)
        ridx = int(row.cache_row)
        src_g = str(row.source_genre)
        tgt_g = str(row.target_genre)
        tgt_idx = int(row.target_genre_idx)

        zc_np = arrays["z_content"][ridx].astype(np.float32)
        mel_real_norm = arrays["mel_norm"][ridx].astype(np.float32)
        zc = torch.from_numpy(zc_np).unsqueeze(0).to(device_t)
        cond = cond_bank[tgt_idx].unsqueeze(0).float()

        with torch.no_grad():
            mel_fake_norm = generator(zc, cond).squeeze(0).detach().cpu().numpy().astype(np.float32)

        mel_real_db = _mel_norm_to_db_np(mel_real_norm)
        mel_fake_db = _mel_norm_to_db_np(mel_fake_norm)
        y_fake = _mel_db_to_audio(mel_fake_db, sr=sr, griffin_lim_iters=int(griffin_lim_iters))
        y_real = _mel_db_to_audio(mel_real_db, sr=sr, griffin_lim_iters=int(griffin_lim_iters)) if write_real_audio else None

        fake_gl_db = _audio_to_mel_db(y_fake, sr=sr, n_frames=mel_fake_db.shape[1])
        gl_fake_floor_l1 = float(np.mean(np.abs(fake_gl_db - mel_fake_db)))
        gl_real_floor_l1 = float("nan")
        if write_real_audio and y_real is not None:
            real_gl_db = _audio_to_mel_db(y_real, sr=sr, n_frames=mel_real_db.shape[1])
            gl_real_floor_l1 = float(np.mean(np.abs(real_gl_db - mel_real_db)))

        fake_wav = out_dir / f"sample_{sample_id:03d}_fake_to_{tgt_g}.wav"
        real_wav = out_dir / f"sample_{sample_id:03d}_real.wav"
        mel_plot = out_dir / f"sample_{sample_id:03d}_mel.png"
        sf.write(str(fake_wav), y_fake, sr)
        if write_real_audio and y_real is not None:
            sf.write(str(real_wav), y_real, sr)

        with torch.no_grad():
            enc_out = frozen_encoder.forward_log_mel_tensor(torch.from_numpy(mel_fake_db).unsqueeze(0).to(frozen_encoder.device))
            zc_pred = enc_out["z_content"][0].detach().cpu().numpy().astype(np.float32)
        mps = _cosine_np(zc_pred, zc_np)

        style_pred_source = ""
        style_conf_target = float("nan")
        if genre_to_source_idx is not None and tgt_g in genre_to_source_idx:
            source_idx = int(genre_to_source_idx.get(tgt_g, -1))
            with torch.no_grad():
                logits = frozen_encoder.model(torch.from_numpy(mel_fake_db).unsqueeze(0).to(frozen_encoder.device), grl_lambda=0.0)[
                    "style_logits"
                ][0]
                probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
            idx_to_source = {int(v): str(k) for k, v in frozen_encoder.source_to_idx.items()}
            style_pred_source = idx_to_source.get(int(np.argmax(probs)), str(int(np.argmax(probs))))
            if 0 <= source_idx < len(probs):
                style_conf_target = float(probs[source_idx])

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        im0 = ax[0].imshow(mel_real_db, origin="lower", aspect="auto", cmap="magma")
        ax[0].set_title(f"Real | {src_g}")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        im1 = ax[1].imshow(mel_fake_db, origin="lower", aspect="auto", cmap="magma")
        ax[1].set_title(f"Fake | {src_g} -> {tgt_g}")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(mel_plot, dpi=140)
        plt.close(fig)

        src_path = ""
        if "path" in index_df.columns and ridx < len(index_df):
            src_path = str(index_df.iloc[ridx]["path"])

        recs.append(
            {
                "sample_id": sample_id,
                "cache_row": ridx,
                "source_path": src_path,
                "source_genre": src_g,
                "target_genre": tgt_g,
                "mps_cosine": float(mps),
                "style_pred_source": style_pred_source,
                "style_conf_target": style_conf_target,
                "gl_fake_floor_l1": gl_fake_floor_l1,
                "gl_real_floor_l1": gl_real_floor_l1,
                "mel_plot": str(mel_plot),
                "real_wav": str(real_wav) if write_real_audio else "",
                "fake_wav": str(fake_wav),
            }
        )

    summary_csv = out_dir / "generation_summary.csv"
    pd.DataFrame(recs).to_csv(summary_csv, index=False)
    meta = {
        "n_samples": int(len(recs)),
        "target_mode": str(target_mode),
        "griffin_lim_iters": int(griffin_lim_iters),
        "seed": int(seed),
        "write_real_audio": bool(write_real_audio),
        "mean_gl_fake_floor_l1": float(np.nanmean([r.get("gl_fake_floor_l1", np.nan) for r in recs])),
        "mean_gl_real_floor_l1": float(np.nanmean([r.get("gl_real_floor_l1", np.nan) for r in recs])),
        "output_dir": str(out_dir),
        "summary_csv": str(summary_csv),
    }
    meta_path = out_dir / "sample_export_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {
        "output_dir": str(out_dir),
        "summary_csv": str(summary_csv),
        "meta_json": str(meta_path),
    }
