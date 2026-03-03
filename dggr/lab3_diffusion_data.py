"""Diffusion mel-spectrogram cache builder and dataset.

Builds a large-scale cache from ALL manifest audio:
  - mel       [80, 432]  BigVGAN-compatible log mel spectrogram
  - chroma    [12, 432]  chromagram  (content conditioning)
  - onset     [432]      onset strength envelope
  - beat      [432]      binary beat grid
  - z_content [128]      Lab1 content embedding
  - z_style   [128]      Lab1 style embedding
  - mert_feat [768]      raw MERT features (optional)
  - genre_idx int64      genre label

Audio is processed at 22050 Hz in 5-second chunks.
Mel extraction uses BigVGAN's exact function for perfect round-trip fidelity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .lab3_bridge import (
    FrozenLab1Encoder,
    extract_log_mel,
    fix_log_mel_frames,
    load_audio_chunk,
)
from .lab3_data import (
    DEFAULT_GENRE_RULES,
    assign_genres,
    load_manifests,
    stratified_group_split_indices,
)

# ---------------------------------------------------------------------------
# Constants matching BigVGAN v2 22khz_80band_256x
# ---------------------------------------------------------------------------
DIFFUSION_SR = 22050
DIFFUSION_HOP = 256
DIFFUSION_N_FFT = 1024
DIFFUSION_WIN = 1024
DIFFUSION_N_MELS = 80
DIFFUSION_FMIN = 0
DIFFUSION_FMAX = None  # = sr / 2
DIFFUSION_CHUNK_SEC = 5.0
DIFFUSION_N_FRAMES = 432  # ceil(5*22050/256)=431, padded to next multiple of 16


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_bigvgan_mel_np(y: np.ndarray, sr: int = DIFFUSION_SR) -> np.ndarray:
    """Extract log mel spectrogram using BigVGAN's exact torch pipeline.

    Returns [80, T] float32 in log(clamp(mel_mag, 1e-5)) space.
    """
    import bigvgan as _bvg

    wav_t = torch.from_numpy(y).float().unsqueeze(0)  # [1, S]
    with torch.no_grad():
        mel = _bvg.mel_spectrogram(
            wav_t,
            n_fft=DIFFUSION_N_FFT,
            num_mels=DIFFUSION_N_MELS,
            sampling_rate=sr,
            hop_size=DIFFUSION_HOP,
            win_size=DIFFUSION_WIN,
            fmin=DIFFUSION_FMIN,
            fmax=DIFFUSION_FMAX,
            center=False,
        )  # [1, 80, T]
    return mel.squeeze(0).numpy().astype(np.float32)


def extract_chroma(y: np.ndarray, sr: int = DIFFUSION_SR) -> np.ndarray:
    """Chromagram [12, T]."""
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=DIFFUSION_N_FFT, hop_length=DIFFUSION_HOP,
        win_length=DIFFUSION_WIN, center=False,
    )
    return chroma.astype(np.float32)


def extract_onset(y: np.ndarray, sr: int = DIFFUSION_SR) -> np.ndarray:
    """Onset strength envelope [T]."""
    onset = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=DIFFUSION_HOP, center=False,
    )
    return onset.astype(np.float32)


def extract_beat_grid(y: np.ndarray, sr: int = DIFFUSION_SR, n_frames: int = 0) -> np.ndarray:
    """Binary beat grid [T].  1.0 at beat frame positions, 0.0 elsewhere."""
    _tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=DIFFUSION_HOP,
    )
    if n_frames <= 0:
        # estimate from audio length
        n_frames = int(np.ceil(len(y) / DIFFUSION_HOP))
    grid = np.zeros(n_frames, dtype=np.float32)
    for bf in beat_frames:
        if 0 <= bf < n_frames:
            grid[bf] = 1.0
    return grid


def pad_or_trim(arr: np.ndarray, target: int, axis: int = -1, pad_val: float = 0.0) -> np.ndarray:
    """Pad or trim *arr* along *axis* to exactly *target* frames."""
    length = arr.shape[axis]
    if length == target:
        return arr
    if length > target:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target)
        return arr[tuple(slices)].copy()
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, target - length)
    return np.pad(arr, pad_width, mode="constant", constant_values=pad_val)


# ---------------------------------------------------------------------------
# Cache metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffusionCacheMeta:
    sr: int = DIFFUSION_SR
    hop: int = DIFFUSION_HOP
    n_fft: int = DIFFUSION_N_FFT
    n_mels: int = DIFFUSION_N_MELS
    chunk_sec: float = DIFFUSION_CHUNK_SEC
    n_frames: int = DIFFUSION_N_FRAMES
    mel_min: float = -11.5
    mel_max: float = 2.0
    n_samples: int = 0
    has_mert: bool = False


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def _track_id(path_str: str) -> str:
    return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:16]


def _duration_seconds(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


def _flush_shard(
    shard_dir: Path,
    shard_id: int,
    mel_list: list, chroma_list: list, onset_list: list, beat_list: list,
    zc_list: list, zs_list: list, mert_list: list, gidx_list: list,
) -> None:
    """Flush accumulated chunks to a numbered shard on disk."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    tag = f"shard_{shard_id:04d}"
    np.save(shard_dir / f"{tag}_mel.npy", np.stack(mel_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_chroma.npy", np.stack(chroma_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_onset.npy", np.stack(onset_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_beat.npy", np.stack(beat_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_z_content.npy", np.stack(zc_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_z_style.npy", np.stack(zs_list).astype(np.float32))
    np.save(shard_dir / f"{tag}_genre_idx.npy", np.asarray(gidx_list, dtype=np.int64))
    if mert_list:
        np.save(shard_dir / f"{tag}_mert_feat.npy", np.stack(mert_list).astype(np.float32))


def _merge_shards(shard_dir: Path, n_shards: int, has_mert: bool, final_dir: Path) -> None:
    """Concatenate shard files into final .npy files using pre-allocated memmap.

    Avoids OOM by writing directly to disk-backed arrays instead of loading all
    shards into memory at once.
    """
    import shutil

    final_dir.mkdir(parents=True, exist_ok=True)
    array_names = ["mel", "chroma", "onset", "beat", "z_content", "z_style", "genre_idx"]
    if has_mert:
        array_names.append("mert_feat")

    for name in array_names:
        # First pass: determine total length and dtype/shape
        shard_paths = []
        total_len = 0
        sample_shape = None
        sample_dtype = None
        for sid in range(n_shards):
            p = shard_dir / f"shard_{sid:04d}_{name}.npy"
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                shard_paths.append((p, arr.shape[0]))
                total_len += arr.shape[0]
                if sample_shape is None:
                    sample_shape = arr.shape[1:]
                    sample_dtype = arr.dtype
                del arr

        if total_len == 0 or sample_shape is None:
            continue

        # Create output memmap and fill from shards
        out_path = final_dir / f"diff_{name}.npy"
        full_shape = (total_len, *sample_shape)
        out = np.lib.format.open_memmap(
            str(out_path), mode="w+", dtype=sample_dtype, shape=full_shape,
        )
        offset = 0
        for p, n in shard_paths:
            chunk = np.load(p)
            out[offset:offset + n] = chunk
            offset += n
            del chunk
        out.flush()
        del out
        print(f"  merged {name}: {full_shape}")

    shutil.rmtree(shard_dir, ignore_errors=True)


def build_diffusion_cache(
    manifests_root: Path,
    manifest_files: List[str],
    lab1_encoder: FrozenLab1Encoder,
    cache_dir: Path,
    *,
    mert=None,
    chunk_sec: float = DIFFUSION_CHUNK_SEC,
    max_chunks_per_track: int = 10,
    seed: int = 328,
    progress_every: int = 200,
    shard_size: int = 5000,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int], DiffusionCacheMeta]:
    """Build diffusion cache from ALL manifest audio (no genre sampling cap).

    Uses incremental shard-based saving to avoid OOM with large datasets.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / "_shards"

    # Load and genre-assign ALL tracks
    all_df = load_manifests(manifests_root, manifest_files)
    all_df = assign_genres(all_df, rules=DEFAULT_GENRE_RULES)
    all_df = all_df[all_df["genre"] != "unassigned"].reset_index(drop=True)
    print(f"[diffusion-cache] {len(all_df)} tracks across genres: "
          f"{dict(all_df['genre'].value_counts().sort_index())}")

    genres = sorted(all_df["genre"].astype(str).unique().tolist())
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    rng = np.random.default_rng(int(seed))
    lab1_sr = int(lab1_encoder.cfg.sample_rate)
    lab1_nf = 256
    mert_sr = int(mert.cfg.sample_rate) if mert is not None else 0
    target_frames = DIFFUSION_N_FRAMES

    rows: List[Dict] = []
    mel_buf: List[np.ndarray] = []
    chroma_buf: List[np.ndarray] = []
    onset_buf: List[np.ndarray] = []
    beat_buf: List[np.ndarray] = []
    zc_buf: List[np.ndarray] = []
    zs_buf: List[np.ndarray] = []
    mert_buf: List[np.ndarray] = []
    gidx_buf: List[int] = []

    mel_running_min = 999.0
    mel_running_max = -999.0
    shard_id = 0
    total_chunks = 0
    has_mert = mert is not None

    def _maybe_flush():
        nonlocal shard_id, mel_buf, chroma_buf, onset_buf, beat_buf
        nonlocal zc_buf, zs_buf, mert_buf, gidx_buf
        if len(mel_buf) >= shard_size:
            _flush_shard(shard_dir, shard_id, mel_buf, chroma_buf, onset_buf,
                         beat_buf, zc_buf, zs_buf, mert_buf, gidx_buf)
            print(f"[diffusion-cache] flushed shard {shard_id} ({len(mel_buf)} chunks)")
            shard_id += 1
            mel_buf, chroma_buf, onset_buf, beat_buf = [], [], [], []
            zc_buf, zs_buf, mert_buf, gidx_buf = [], [], [], []

    for i, rec in all_df.iterrows():
        p = Path(str(rec["path"]))
        if not p.exists():
            continue

        dur = _duration_seconds(p)
        if dur < 1.0:
            continue

        n_chunks = min(int(max_chunks_per_track), max(1, int(dur // chunk_sec)))
        if n_chunks == 1:
            starts = [0.0]
        else:
            max_start = max(0.0, dur - chunk_sec)
            starts = np.linspace(0.0, max_start, n_chunks).tolist()

        for chunk_id, start_sec in enumerate(starts):
            try:
                y = load_audio_chunk(
                    path=p, sample_rate=DIFFUSION_SR,
                    seconds=chunk_sec, start_sec=float(start_sec),
                )
                mel = extract_bigvgan_mel_np(y, sr=DIFFUSION_SR)
                mel = pad_or_trim(mel, target_frames, axis=1, pad_val=-11.5)
                chroma = extract_chroma(y, sr=DIFFUSION_SR)
                chroma = pad_or_trim(chroma, target_frames, axis=1)
                onset = extract_onset(y, sr=DIFFUSION_SR)
                onset = pad_or_trim(onset, target_frames, axis=0)
                beat = extract_beat_grid(y, sr=DIFFUSION_SR, n_frames=target_frames)
                beat = pad_or_trim(beat, target_frames, axis=0)
                log_mel_lab1 = extract_log_mel(y, sr=lab1_sr)
                log_mel_lab1 = fix_log_mel_frames(log_mel_lab1, n_frames=lab1_nf)
                lat = lab1_encoder.infer_log_mel(log_mel_lab1)
                mert_feat: Optional[np.ndarray] = None
                if mert is not None:
                    if mert_sr != DIFFUSION_SR:
                        y_mert = librosa.resample(
                            y, orig_sr=DIFFUSION_SR, target_sr=mert_sr, res_type="soxr_hq",
                        )
                    else:
                        y_mert = y
                    mert_feat = mert.extract_features(y_mert)
            except Exception:
                continue

            genre = str(rec["genre"])
            gidx = int(genre_to_idx[genre])
            mel_running_min = min(mel_running_min, float(mel.min()))
            mel_running_max = max(mel_running_max, float(mel.max()))

            rows.append({
                "path": str(p),
                "track_id": _track_id(str(p)),
                "source": str(rec["source"]),
                "genre": genre,
                "genre_idx": gidx,
                "chunk_id": int(chunk_id),
                "start_sec": float(start_sec),
                "manifest_file": str(rec.get("manifest_file", "")),
            })
            mel_buf.append(mel)
            chroma_buf.append(chroma)
            onset_buf.append(onset)
            beat_buf.append(beat)
            zc_buf.append(lat["z_content"].astype(np.float32))
            zs_buf.append(lat["z_style"].astype(np.float32))
            if mert_feat is not None:
                mert_buf.append(mert_feat.astype(np.float32))
            gidx_buf.append(gidx)
            total_chunks += 1
            _maybe_flush()

        if progress_every > 0 and (int(i) + 1) % int(progress_every) == 0:
            print(f"[diffusion-cache] tracks={int(i)+1}/{len(all_df)}  chunks={total_chunks}"
                  f"  mel_range=[{mel_running_min:.2f}, {mel_running_max:.2f}]")

    # Flush remaining
    if mel_buf:
        _flush_shard(shard_dir, shard_id, mel_buf, chroma_buf, onset_buf,
                     beat_buf, zc_buf, zs_buf, mert_buf, gidx_buf)
        print(f"[diffusion-cache] flushed final shard {shard_id} ({len(mel_buf)} chunks)")
        shard_id += 1
        mel_buf, chroma_buf, onset_buf, beat_buf = [], [], [], []
        zc_buf, zs_buf, mert_buf, gidx_buf = [], [], [], []

    if total_chunks == 0:
        raise RuntimeError("No diffusion cache rows built. Check manifests and paths.")

    # Merge shards into final files
    print(f"[diffusion-cache] merging {shard_id} shards into final arrays...")
    _merge_shards(shard_dir, shard_id, has_mert, cache_dir)

    index_df = pd.DataFrame(rows)
    meta = DiffusionCacheMeta(
        mel_min=float(mel_running_min),
        mel_max=float(mel_running_max),
        n_samples=total_chunks,
        has_mert=has_mert,
    )

    # Save index + metadata (arrays already saved by _merge_shards)
    index_df.to_csv(cache_dir / "diff_index.csv", index=False)
    with (cache_dir / "diff_genre_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in genre_to_idx.items()}, f, indent=2)
    with (cache_dir / "diff_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "sr": meta.sr, "hop": meta.hop, "n_fft": meta.n_fft,
            "n_mels": meta.n_mels, "chunk_sec": meta.chunk_sec,
            "n_frames": meta.n_frames, "mel_min": meta.mel_min,
            "mel_max": meta.mel_max, "n_samples": meta.n_samples,
            "has_mert": meta.has_mert,
        }, f, indent=2)

    print(f"[diffusion-cache] DONE: {total_chunks} chunks from {len(index_df['track_id'].unique())} tracks")
    print(f"  mel range: [{meta.mel_min:.3f}, {meta.mel_max:.3f}]")
    print(f"  genres: {dict(index_df['genre'].value_counts().sort_index())}")

    # Return loaded arrays via mmap for downstream use
    _, arrays, _, _ = load_diffusion_cache(cache_dir, mmap=True)
    return index_df, arrays, genre_to_idx, meta


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_diffusion_cache(
    cache_dir: Path,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    genre_to_idx: Dict[str, int],
    meta: DiffusionCacheMeta,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(cache_dir / "diff_index.csv", index=False)
    # Save large arrays individually for mmap support
    for name, arr in arrays.items():
        np.save(cache_dir / f"diff_{name}.npy", arr)
    with (cache_dir / "diff_genre_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in genre_to_idx.items()}, f, indent=2)
    with (cache_dir / "diff_meta.json").open("w", encoding="utf-8") as f:
        json.dump({
            "sr": meta.sr, "hop": meta.hop, "n_fft": meta.n_fft,
            "n_mels": meta.n_mels, "chunk_sec": meta.chunk_sec,
            "n_frames": meta.n_frames, "mel_min": meta.mel_min,
            "mel_max": meta.mel_max, "n_samples": meta.n_samples,
            "has_mert": meta.has_mert,
        }, f, indent=2)


def load_diffusion_cache(
    cache_dir: Path,
    mmap: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int], DiffusionCacheMeta]:
    cache_dir = Path(cache_dir)
    index_df = pd.read_csv(cache_dir / "diff_index.csv")
    mmode = "r" if mmap else None
    arrays: Dict[str, np.ndarray] = {}
    for name in ("mel", "chroma", "onset", "beat", "z_content", "z_style", "genre_idx"):
        p = cache_dir / f"diff_{name}.npy"
        if p.exists():
            arrays[name] = np.load(p, mmap_mode=mmode)
    mert_p = cache_dir / "diff_mert_feat.npy"
    if mert_p.exists():
        arrays["mert_feat"] = np.load(mert_p, mmap_mode=mmode)
    with (cache_dir / "diff_genre_to_idx.json").open("r", encoding="utf-8") as f:
        genre_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
    with (cache_dir / "diff_meta.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = DiffusionCacheMeta(**{k: type(getattr(DiffusionCacheMeta, k, None))(v)
                                  if hasattr(DiffusionCacheMeta, k) else v
                                  for k, v in raw.items()})
    return index_df, arrays, genre_to_idx, meta


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _aug_time_shift(mel: np.ndarray, chroma: np.ndarray, onset: np.ndarray,
                    beat: np.ndarray, max_shift: int = 16,
                    rng: np.random.Generator = None) -> tuple:
    """Random circular time shift ±max_shift frames."""
    if rng is None:
        rng = np.random.default_rng()
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return mel, chroma, onset, beat
    mel = np.roll(mel, shift, axis=-1)
    chroma = np.roll(chroma, shift, axis=-1)
    onset = np.roll(onset, shift, axis=-1)
    beat = np.roll(beat, shift, axis=-1)
    return mel, chroma, onset, beat


def _aug_freq_mask(mel: np.ndarray, max_bands: int = 12,
                   rng: np.random.Generator = None) -> np.ndarray:
    """SpecAugment-style frequency masking (only on mel, not chroma/onset)."""
    if rng is None:
        rng = np.random.default_rng()
    n_bands = int(rng.integers(1, max_bands + 1))
    f0 = int(rng.integers(0, max(1, mel.shape[0] - n_bands)))
    mel = mel.copy()
    mel[f0:f0 + n_bands, :] = mel.min()
    return mel


def _aug_gain(mel: np.ndarray, max_db: float = 3.0,
              rng: np.random.Generator = None) -> np.ndarray:
    """Random gain ±max_db in log mel space (additive in dB ≈ log space)."""
    if rng is None:
        rng = np.random.default_rng()
    gain = float(rng.uniform(-max_db, max_db))
    # In log space, gain is additive (log(mel * g) = log(mel) + log(g))
    # max_db corresponds to ~0.69 in natural log scale
    gain_log = gain * np.log(10) / 20  # convert dB to neper
    return mel + gain_log


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DiffusionMelDataset(Dataset):
    """PyTorch dataset for diffusion training.

    Returns dict with:
        mel       [1, 80, 432]  — normalized to [-1, 1]
        cond_feat [14, 80, 432] — 12 chroma + 1 onset + 1 beat  (expanded to 80 height)
        z_content [128]
        z_style   [128]
        genre_idx int
    """

    def __init__(
        self,
        arrays: Dict[str, np.ndarray],
        indices: np.ndarray,
        mel_min: float = -11.5,
        mel_max: float = 2.0,
        augment: bool = False,
        seed: int = 42,
        style_source: str = "lab1",  # "lab1" or "mert"
        max_frames: Optional[int] = None,
    ):
        # Keep references to mmap arrays and index mapping for lazy loading
        self._arrays = arrays
        self._indices = np.asarray(indices)
        # Only eagerly load small arrays (genre_idx is just int per sample)
        self.genre_idx = arrays["genre_idx"][self._indices]
        self._has_mert = "mert_feat" in arrays

        self.mel_min = float(mel_min)
        self.mel_max = float(mel_max)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(int(seed))
        self.style_source = str(style_source)
        self.max_frames = max_frames

    def __len__(self) -> int:
        return int(len(self.genre_idx))

    def _normalize_mel(self, mel: np.ndarray) -> np.ndarray:
        """Normalize log mel from [mel_min, mel_max] to [-1, 1]."""
        span = self.mel_max - self.mel_min
        if span < 1e-6:
            span = 1.0
        return (2.0 * (mel - self.mel_min) / span - 1.0).astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self._indices[i])
        mel = np.array(self._arrays["mel"][real_idx])        # [80, 432]
        chroma = np.array(self._arrays["chroma"][real_idx])  # [12, 432]
        onset = np.array(self._arrays["onset"][real_idx])    # [432]
        beat = np.array(self._arrays["beat"][real_idx])      # [432]

        # Trim to max_frames if set (e.g., 256 frames = ~3 seconds)
        if self.max_frames is not None and mel.shape[-1] > self.max_frames:
            mel = mel[:, :self.max_frames]
            chroma = chroma[:, :self.max_frames]
            onset = onset[:self.max_frames]
            beat = beat[:self.max_frames]

        if self.augment:
            mel, chroma, onset, beat = _aug_time_shift(
                mel, chroma, onset, beat, max_shift=16, rng=self.rng)
            mel = _aug_freq_mask(mel, max_bands=12, rng=self.rng)
            mel = _aug_gain(mel, max_db=3.0, rng=self.rng)

        mel_norm = self._normalize_mel(mel)  # [80, 432]

        # Build conditioning using torch expand (zero-copy views, no RAM spike)
        mel_t = torch.from_numpy(mel_norm).unsqueeze(0)  # [1, 80, T]
        T = mel_t.shape[2]
        H = mel_t.shape[1]  # 80

        chroma_t = torch.from_numpy(chroma).unsqueeze(1).expand(-1, H, -1)  # [12, 80, T]
        onset_t = torch.from_numpy(onset).reshape(1, 1, T).expand(1, H, -1)  # [1, 80, T]
        beat_t = torch.from_numpy(beat).reshape(1, 1, T).expand(1, H, -1)    # [1, 80, T]
        cond_feat = torch.cat([chroma_t, onset_t, beat_t], dim=0).contiguous()  # [14, 80, T]

        # Style vector
        if self.style_source == "mert" and self._has_mert:
            z_style = np.array(self._arrays["mert_feat"][real_idx])
        else:
            z_style = np.array(self._arrays["z_style"][real_idx])

        return {
            "mel": mel_t,                                                    # [1, 80, 432]
            "cond_feat": cond_feat,                                          # [14, 80, 432]
            "z_content": torch.from_numpy(np.array(self._arrays["z_content"][real_idx])),  # [128]
            "z_style": torch.from_numpy(z_style),                           # [128]
            "genre_idx": torch.tensor(int(self.genre_idx[i]), dtype=torch.long),
        }


def denormalize_mel(mel_norm: torch.Tensor, mel_min: float, mel_max: float) -> torch.Tensor:
    """Convert [-1, 1] normalized mel back to BigVGAN log mel space."""
    span = mel_max - mel_min
    if span < 1e-6:
        span = 1.0
    return (mel_norm + 1.0) * 0.5 * span + mel_min
