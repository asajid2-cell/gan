#!/usr/bin/env python3
"""Lab 4 longform coherence runner.

Generates longform style transfer with diffusion continuity constraints:
  - SDEdit anchor from source mel per chunk
  - Prefix overlap locking at every DDIM step
  - Waveform crossfade assembly
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_REPO_ROOT = _SCRIPT_DIR.parent

from src.lab3_bridge import (
    FrozenLab1Encoder,
    extract_log_mel,
    fix_log_mel_frames,
    load_audio_chunk,
)
from src.lab3_diffusion_data import (
    DIFFUSION_HOP,
    DIFFUSION_N_FRAMES,
    DIFFUSION_SR,
    extract_beat_grid,
    extract_bigvgan_mel_np,
    extract_chroma,
    extract_onset,
    load_diffusion_cache,
    pad_or_trim,
)
from src.lab3_diffusion_model import DiffusionUNetV2, EMA, NoiseSchedule
from src.lab3_diffusion_train import ddim_sample_v2_constrained, vocode_bigvgan


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def normalize_mel_np(mel: np.ndarray, mel_min: float, mel_max: float) -> np.ndarray:
    span = float(mel_max - mel_min)
    if span < 1e-6:
        span = 1.0
    mel_norm = (mel - float(mel_min)) / span
    mel_norm = mel_norm * 2.0 - 1.0
    return np.clip(mel_norm, -1.0, 1.0).astype(np.float32)


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x))
    return (x / (n + 1e-8)).astype(np.float32)


def smooth_mel_tensor(
    mel: torch.Tensor,
    time_kernel: int = 0,
    freq_kernel: int = 0,
) -> torch.Tensor:
    """Apply optional box smoothing to mel [B,1,80,T]."""
    out = mel
    if int(freq_kernel) > 1:
        kf = int(freq_kernel)
        if kf % 2 == 0:
            kf += 1
        k = torch.ones((1, 1, kf, 1), device=out.device, dtype=out.dtype) / float(kf)
        out = F.conv2d(out, k, padding=(kf // 2, 0))
    if int(time_kernel) > 1:
        kt = int(time_kernel)
        if kt % 2 == 0:
            kt += 1
        k = torch.ones((1, 1, 1, kt), device=out.device, dtype=out.dtype) / float(kt)
        out = F.conv2d(out, k, padding=(0, kt // 2))
    return out


def split_audio_overlapping(
    audio: np.ndarray,
    chunk_seconds: float,
    overlap_seconds: float,
    sr: int,
) -> List[Dict]:
    chunk_samples = int(round(float(chunk_seconds) * float(sr)))
    overlap_samples = int(round(float(overlap_seconds) * float(sr)))
    hop_samples = max(1, chunk_samples - overlap_samples)
    chunks: List[Dict] = []

    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        chunk = audio[pos:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append({
            "audio": chunk.astype(np.float32),
            "start_sample": int(pos),
            "end_sample": int(end),
        })
        if end >= len(audio):
            break
        pos += hop_samples

    if len(chunks) == 0:
        chunks.append({
            "audio": np.zeros(chunk_samples, dtype=np.float32),
            "start_sample": 0,
            "end_sample": 0,
        })
    return chunks


def cosine_crossfade_weights(overlap_samples: int) -> np.ndarray:
    t = np.linspace(0.0, np.pi / 2.0, int(overlap_samples), dtype=np.float32)
    return np.cos(t).astype(np.float32)


def assemble_audio_crossfade(chunk_wavs: List[np.ndarray], overlap_seconds: float, sr: int) -> np.ndarray:
    if len(chunk_wavs) == 0:
        return np.zeros(1, dtype=np.float32)
    if len(chunk_wavs) == 1:
        return chunk_wavs[0].astype(np.float32)

    overlap_samples = int(round(float(overlap_seconds) * float(sr)))
    overlap_samples = max(1, overlap_samples)
    fade = cosine_crossfade_weights(overlap_samples)

    out = chunk_wavs[0].astype(np.float32).copy()
    for i in range(1, len(chunk_wavs)):
        cur = chunk_wavs[i].astype(np.float32)
        real_ov = min(overlap_samples, len(out), len(cur))
        if real_ov > 0:
            f = fade[:real_ov]
            out[-real_ov:] = out[-real_ov:] * f + cur[:real_ov] * (1.0 - f)
            out = np.concatenate([out, cur[real_ov:]], axis=0)
        else:
            out = np.concatenate([out, cur], axis=0)
    return out.astype(np.float32)


def assemble_mel_crossfade(chunk_mels: List[np.ndarray], overlap_frames: int) -> np.ndarray:
    """Assemble normalized mel chunks [80, T] with cosine crossfade in time."""
    if len(chunk_mels) == 0:
        return np.zeros((80, 1), dtype=np.float32)
    if len(chunk_mels) == 1:
        return chunk_mels[0].astype(np.float32)

    ov = max(1, int(overlap_frames))
    fade = np.cos(np.linspace(0.0, np.pi / 2.0, ov, dtype=np.float32))[None, :]  # [1, ov]
    out = chunk_mels[0].astype(np.float32).copy()
    for i in range(1, len(chunk_mels)):
        cur = chunk_mels[i].astype(np.float32)
        real_ov = min(ov, out.shape[1], cur.shape[1])
        if real_ov > 0:
            f = fade[:, :real_ov]
            out[:, -real_ov:] = out[:, -real_ov:] * f + cur[:, :real_ov] * (1.0 - f)
            out = np.concatenate([out, cur[:, real_ov:]], axis=1)
        else:
            out = np.concatenate([out, cur], axis=1)
    return out.astype(np.float32)


def extract_chunk_features(
    y: np.ndarray,
    *,
    n_frames: int,
    mel_min: float,
    mel_max: float,
    lab1_encoder: FrozenLab1Encoder,
) -> Dict[str, np.ndarray]:
    mel = extract_bigvgan_mel_np(y, sr=DIFFUSION_SR)
    mel = pad_or_trim(mel, n_frames, axis=1, pad_val=float(mel_min))
    mel_norm = normalize_mel_np(mel, mel_min, mel_max)

    chroma = extract_chroma(y, sr=DIFFUSION_SR)
    chroma = pad_or_trim(chroma, n_frames, axis=1, pad_val=0.0)
    onset = extract_onset(y, sr=DIFFUSION_SR)
    onset = pad_or_trim(onset, n_frames, axis=0, pad_val=0.0)
    beat = extract_beat_grid(y, sr=DIFFUSION_SR, n_frames=n_frames)
    beat = pad_or_trim(beat, n_frames, axis=0, pad_val=0.0)
    H = mel_norm.shape[0]  # 80 mel bins
    chroma_exp = np.repeat(chroma[:, None, :], H, axis=1)          # [12, 80, T]
    onset_exp = np.repeat(onset[None, None, :], H, axis=1)         # [1, 80, T]
    beat_exp = np.repeat(beat[None, None, :], H, axis=1)           # [1, 80, T]
    cond_feat = np.concatenate([chroma_exp, onset_exp, beat_exp], axis=0).astype(np.float32)

    log_mel = extract_log_mel(y, sr=lab1_encoder.cfg.sample_rate)
    log_mel = fix_log_mel_frames(log_mel, n_frames=n_frames)
    lat = lab1_encoder.infer_log_mel(log_mel)

    return {
        "mel_norm": mel_norm.astype(np.float32),
        "cond_feat": cond_feat.astype(np.float32),
        "z_content": lat["z_content"].astype(np.float32),
        "z_style": lat["z_style"].astype(np.float32),
    }


def chunk_boundary_discontinuity(
    audio: np.ndarray,
    *,
    chunk_seconds: float,
    overlap_seconds: float,
    sr: int,
    window_ms: float = 50.0,
) -> List[float]:
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, hop_length=DIFFUSION_HOP),
        ref=np.max,
    )
    hop_samples = int(round((float(chunk_seconds) - float(overlap_seconds)) * float(sr)))
    if hop_samples <= 0:
        return []

    boundaries = np.arange(hop_samples, len(audio), hop_samples, dtype=np.int64)
    window_frames = max(1, int(round(float(window_ms) / 1000.0 * float(sr) / float(DIFFUSION_HOP))))
    vals: List[float] = []
    for b in boundaries:
        frame = int(b // DIFFUSION_HOP)
        if frame - window_frames < 0 or frame + window_frames >= mel_db.shape[1]:
            continue
        left = mel_db[:, frame - window_frames:frame].mean(axis=1)
        right = mel_db[:, frame:frame + window_frames].mean(axis=1)
        vals.append(float(np.mean(np.abs(left - right))))
    return vals


def load_bigvgan_robust(device: torch.device):
    import bigvgan as bvg

    model_id = "nvidia/bigvgan_v2_22khz_80band_256x"
    try:
        vocoder = bvg.BigVGAN.from_pretrained(model_id, use_cuda_kernel=False)
        vocoder.remove_weight_norm()
        vocoder.eval().to(device)
        return vocoder
    except Exception:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model_id, "config.json")
        ckpt_path = hf_hub_download(model_id, "bigvgan_generator.pt")

        with open(config_path, "r", encoding="utf-8") as f:
            hparams = bvg.AttrDict(json.load(f))
        vocoder = bvg.BigVGAN(hparams)

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload["generator"] if isinstance(payload, dict) and "generator" in payload else payload
        vocoder.load_state_dict(state_dict, strict=False)
        vocoder.remove_weight_norm()
        vocoder.eval().to(device)
        return vocoder


def pick_source_path(index_df, source_genre: str) -> Path:
    rows = index_df[index_df["genre"] == str(source_genre)]
    if len(rows) == 0:
        rows = index_df
    path = Path(str(rows.iloc[0]["path"]))
    if not path.exists():
        raise FileNotFoundError(f"Source audio not found: {path}")
    return path


def build_style_centroid(arrays: Dict[str, np.ndarray], target_genre_idx: int) -> np.ndarray:
    g = np.asarray(arrays["genre_idx"]).astype(np.int64)
    mask = g == int(target_genre_idx)
    if not np.any(mask):
        raise ValueError(f"No style vectors for target genre idx={target_genre_idx}")
    zs = np.asarray(arrays["z_style"][mask], dtype=np.float32)
    centroid = zs.mean(axis=0).astype(np.float32)
    return l2_normalize_np(centroid)


def main() -> None:
    p = argparse.ArgumentParser(description="Lab 4 coherence-first longform diffusion")
    p.add_argument("--cache-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"))
    p.add_argument("--checkpoint", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt"))
    p.add_argument("--lab1-checkpoint", type=str,
                   default=str(_REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"))
    p.add_argument("--source-audio", type=str, default="")
    p.add_argument("--source-genre", type=str, default="hiphop_xtc")
    p.add_argument("--target-genre", type=str, default="baroque_classical")
    p.add_argument("--source-start-sec", type=float, default=0.0)
    p.add_argument("--source-seconds", type=float, default=60.0,
                   help="If <=0, use full file duration from source-start-sec")
    p.add_argument("--out-dir", type=str,
                   default=str(_REPO_ROOT / "saves2" / "lab4_longform_coherence" / "run_c001"))
    p.add_argument("--chunk-seconds", type=float, default=3.0)
    p.add_argument("--overlap-seconds", type=float, default=0.5)
    p.add_argument("--n-frames", type=int, default=256)
    p.add_argument("--t-start", type=int, default=350)
    p.add_argument("--t-start-end", type=int, default=-1,
                   help="If >=0, linearly anneal t-start across chunks toward this value.")
    p.add_argument("--reanchor-every", type=int, default=12,
                   help="Every N chunks, disable prefix lock and use reanchor t-start. 0 disables.")
    p.add_argument("--reanchor-t-start", type=int, default=220,
                   help="SDEdit t-start for reanchor chunks.")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=2.0)
    p.add_argument("--style-strength", type=float, default=0.75,
                   help="0=keep source style, 1=full target style.")
    p.add_argument("--prefix-blend", type=float, default=1.0)
    p.add_argument("--source-prefix-blend", type=float, default=0.25,
                   help="Blend current source overlap into locked prefix to prevent drift.")
    p.add_argument("--source-mel-blend", type=float, default=0.10,
                   help="Blend full generated mel with source mel to reduce warble/static.")
    p.add_argument("--hf-source-blend", type=float, default=0.20,
                   help="Extra source blend on high mel bins to reduce hiss/warble.")
    p.add_argument("--hf-start-bin", type=int, default=56,
                   help="Start mel bin for high-frequency source blend.")
    p.add_argument("--mel-time-smooth", type=int, default=5,
                   help="Temporal smoothing kernel on mel (odd recommended).")
    p.add_argument("--mel-freq-smooth", type=int, default=0,
                   help="Frequency smoothing kernel on mel (0 disables).")
    p.add_argument("--assemble-domain", type=str, default="mel", choices=["mel", "wave"],
                   help="Assemble in mel domain then vocode once (recommended) or in waveform domain.")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=328)
    args = p.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coherence_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load cache for mel bounds + genre indices + style centroid.
    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(args.cache_dir), mmap=True)
    if args.target_genre not in genre_to_idx:
        raise ValueError(f"Unknown target genre '{args.target_genre}'. choices={list(genre_to_idx.keys())}")
    target_idx = int(genre_to_idx[args.target_genre])
    z_style_tgt = build_style_centroid(arrays, target_idx)
    z_style_tgt_t = torch.from_numpy(z_style_tgt).unsqueeze(0).to(device)
    style_strength = float(np.clip(float(args.style_strength), 0.0, 1.0))
    print(f"Target style centroid: {args.target_genre} (idx={target_idx})")

    # Source audio.
    if args.source_audio.strip():
        source_path = Path(args.source_audio)
        if not source_path.exists():
            raise FileNotFoundError(f"Source audio not found: {source_path}")
    else:
        source_path = pick_source_path(index_df, args.source_genre)
    print(f"Source path: {source_path}")

    source_seconds = float(args.source_seconds)
    if source_seconds <= 0.0:
        full_dur = float(librosa.get_duration(path=str(source_path)))
        source_seconds = max(1.0, full_dur - float(args.source_start_sec))
        print(f"Auto source duration: {source_seconds:.2f}s")

    source_audio = load_audio_chunk(
        path=source_path,
        sample_rate=DIFFUSION_SR,
        seconds=source_seconds,
        start_sec=float(args.source_start_sec),
    )
    sf.write(str(out_dir / "source.wav"), source_audio, DIFFUSION_SR)

    # Models.
    lab1 = FrozenLab1Encoder(Path(args.lab1_checkpoint), device=str(device))
    model = DiffusionUNetV2(
        in_channels=15,
        out_channels=1,
        base_ch=64,
        ch_mults=(1, 2, 4, 4),
        n_res=2,
        attn_levels=(2, 3),
        z_content_dim=128,
        z_style_dim=128,
        dropout=0.1,
    ).to(device)
    ema = EMA(model, decay=0.9999)
    schedule = NoiseSchedule(T=1000).to(device)

    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        ema = EMA(model, decay=0.0)
    else:
        raise ValueError("Checkpoint missing both 'ema' and 'model'")
    ema.shadow.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    vocoder = load_bigvgan_robust(device)
    print("BigVGAN loaded")

    # Longform generation.
    chunks = split_audio_overlapping(
        source_audio, float(args.chunk_seconds), float(args.overlap_seconds), DIFFUSION_SR)
    overlap_frames = int(round(float(args.overlap_seconds) * float(DIFFUSION_SR) / float(DIFFUSION_HOP)))
    chunk_samples = int(round(float(args.chunk_seconds) * float(DIFFUSION_SR)))
    print(f"Chunks: {len(chunks)} (chunk={args.chunk_seconds}s, overlap={args.overlap_seconds}s, overlap_frames={overlap_frames})")

    generated_chunks: List[np.ndarray] = []
    generated_mels: List[np.ndarray] = []
    boundary_mel_mse: List[float] = []
    prev_tail_mel: torch.Tensor | None = None

    for i, ch in enumerate(chunks):
        feats = extract_chunk_features(
            ch["audio"],
            n_frames=int(args.n_frames),
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            lab1_encoder=lab1,
        )

        mel_src = torch.from_numpy(feats["mel_norm"]).unsqueeze(0).unsqueeze(0).to(device)
        cond_feat = torch.from_numpy(feats["cond_feat"]).unsqueeze(0).to(device)
        z_content = torch.from_numpy(feats["z_content"]).unsqueeze(0).to(device)
        z_style_src = torch.from_numpy(feats["z_style"]).unsqueeze(0).to(device)
        z_style_mix = F.normalize(
            (1.0 - style_strength) * z_style_src + style_strength * z_style_tgt_t,
            dim=-1,
        )

        is_reanchor = (
            int(args.reanchor_every) > 0 and i > 0 and (i % int(args.reanchor_every) == 0)
        )
        if int(args.t_start_end) >= 0 and len(chunks) > 1:
            frac = float(i) / float(len(chunks) - 1)
            base_t_start = int(round(float(args.t_start) + frac * (float(args.t_start_end) - float(args.t_start))))
        else:
            base_t_start = int(args.t_start)
        chunk_t_start = int(args.reanchor_t_start) if is_reanchor else int(base_t_start)
        prefix_for_chunk = None if is_reanchor else prev_tail_mel

        mel_gen = ddim_sample_v2_constrained(
            ema.shadow,
            schedule,
            cond_feat,
            z_content,
            z_style_mix,
            source_mel=mel_src,
            t_start=chunk_t_start,
            prefix_x0=prefix_for_chunk,
            prefix_frames=int(overlap_frames),
            prefix_blend=float(args.prefix_blend),
            source_prefix_x0=mel_src,
            source_prefix_blend=float(args.source_prefix_blend),
            n_steps=int(args.ddim_steps),
            guidance_scale=float(args.guidance_scale),
            eta=float(args.eta),
            device=device,
        )

        # Post-sampling stabilization in mel domain.
        mel_gen = smooth_mel_tensor(
            mel_gen,
            time_kernel=int(args.mel_time_smooth),
            freq_kernel=int(args.mel_freq_smooth),
        )
        src_blend = float(np.clip(float(args.source_mel_blend), 0.0, 1.0))
        if src_blend > 0.0:
            mel_gen = (1.0 - src_blend) * mel_gen + src_blend * mel_src
        hf_blend = float(np.clip(float(args.hf_source_blend), 0.0, 1.0))
        hf_start = int(np.clip(int(args.hf_start_bin), 0, 79))
        if hf_blend > 0.0 and hf_start < 80:
            mel_gen[..., hf_start:, :] = (
                (1.0 - hf_blend) * mel_gen[..., hf_start:, :]
                + hf_blend * mel_src[..., hf_start:, :]
            )
        mel_gen = torch.clamp(mel_gen, -1.0, 1.0)

        if prev_tail_mel is not None and overlap_frames > 0:
            head = mel_gen[..., :overlap_frames]
            mse = torch.mean((head - prev_tail_mel) ** 2).item()
            boundary_mel_mse.append(float(mse))

        if overlap_frames > 0:
            prev_tail_mel = mel_gen[..., -overlap_frames:].detach()
        else:
            prev_tail_mel = None

        wav = vocode_bigvgan(mel_gen, float(meta.mel_min), float(meta.mel_max), vocoder, device)[0]
        if len(wav) > chunk_samples:
            wav = wav[:chunk_samples]
        elif len(wav) < chunk_samples:
            wav = np.pad(wav, (0, chunk_samples - len(wav)))

        generated_chunks.append(wav.astype(np.float32))
        generated_mels.append(mel_gen[0, 0].detach().cpu().numpy().astype(np.float32))
        sf.write(str(out_dir / f"chunk_{i:03d}.wav"), wav, DIFFUSION_SR)
        if is_reanchor:
            print(f"  chunk {i+1:03d}/{len(chunks)} done [reanchor]")
        else:
            print(f"  chunk {i+1:03d}/{len(chunks)} done")

    if str(args.assemble_domain).lower() == "mel":
        mel_long = assemble_mel_crossfade(generated_mels, overlap_frames)
        mel_long_t = torch.from_numpy(mel_long).unsqueeze(0).unsqueeze(0).to(device)
        assembled = vocode_bigvgan(
            mel_long_t, float(meta.mel_min), float(meta.mel_max), vocoder, device
        )[0].astype(np.float32)
        expected_len = int(round(len(source_audio) + float(args.overlap_seconds) * DIFFUSION_SR))
        if len(assembled) > expected_len:
            assembled = assembled[:expected_len]
        elif len(assembled) < expected_len:
            assembled = np.pad(assembled, (0, expected_len - len(assembled)))
    else:
        assembled = assemble_audio_crossfade(generated_chunks, float(args.overlap_seconds), DIFFUSION_SR)
    peak = float(np.max(np.abs(assembled)))
    if peak > 0:
        assembled = assembled / peak * 0.95
    sf.write(str(out_dir / "longform_coherent.wav"), assembled, DIFFUSION_SR)

    disc_vals = chunk_boundary_discontinuity(
        assembled,
        chunk_seconds=float(args.chunk_seconds),
        overlap_seconds=float(args.overlap_seconds),
        sr=DIFFUSION_SR,
    )
    metrics = {
        "n_chunks": int(len(chunks)),
        "duration_sec": float(len(assembled) / DIFFUSION_SR),
        "boundary_mel_mse_mean": float(np.mean(boundary_mel_mse)) if boundary_mel_mse else 0.0,
        "boundary_mel_mse_p95": float(np.percentile(boundary_mel_mse, 95)) if boundary_mel_mse else 0.0,
        "boundary_disc_db_mean": float(np.mean(disc_vals)) if disc_vals else 0.0,
        "boundary_disc_db_p95": float(np.percentile(disc_vals, 95)) if disc_vals else 0.0,
    }
    with open(out_dir / "coherence_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(json.dumps(metrics, indent=2))
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
