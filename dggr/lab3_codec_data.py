from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .lab3_bridge import extract_log_mel, fix_log_mel_frames
from .lab3_codec_bridge import FrozenEncodec
from .lab3_mert_bridge import FrozenMERT
from .lab3_data import (
    DEFAULT_MANIFESTS,
    assign_genres,
    load_manifests,
    materialize_genre_samples,
    stratified_group_split_indices,
    stratified_split_indices,
)


@dataclass(frozen=True)
class CodecCacheMeta:
    codec_model_id: str
    codec_sample_rate: int
    codec_chunk_seconds: float
    codec_bandwidth: float
    codec_frames: int
    codec_channels: int
    lab1_n_frames: int


def _duration_seconds(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return 0.0


def _pick_chunk_starts(
    path: Path,
    chunk_seconds: float,
    chunks_per_track: int,
    chunk_sampling: str,
    min_start_sec: float,
    max_start_sec: Optional[float],
    rng: np.random.Generator,
) -> List[float]:
    n_chunks = max(1, int(chunks_per_track))
    dur = _duration_seconds(path)
    lo = max(0.0, float(min_start_sec))
    if max_start_sec is None:
        hi_raw = max(0.0, dur - float(chunk_seconds))
    else:
        hi_raw = max(0.0, float(max_start_sec))
    hi = max(lo, hi_raw)
    if n_chunks == 1 or hi <= lo + 1e-9:
        return [float(lo)] * n_chunks
    if str(chunk_sampling).strip().lower() == "random":
        return [float(rng.uniform(lo, hi)) for _ in range(n_chunks)]
    vals = np.linspace(lo, hi, num=n_chunks, dtype=np.float64)
    return [float(v) for v in vals]


def _track_id(path_str: str) -> str:
    return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:16]


def build_codec_cache(
    samples_df: pd.DataFrame,
    codec: FrozenEncodec,
    lab1_encoder,
    cache_dir: Path,
    lab1_n_frames: int = 256,
    chunks_per_track: int = 2,
    chunk_sampling: str = "uniform",
    min_start_sec: float = 0.0,
    max_start_sec: Optional[float] = None,
    seed: int = 328,
    progress_every: int = 100,
    mert: Optional[FrozenMERT] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int], CodecCacheMeta]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    genres = sorted(samples_df["genre"].astype(str).unique().tolist())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    rng = np.random.default_rng(int(seed))

    target_frames = int(codec.expected_num_frames())
    rows: List[Dict] = []
    q_emb_list: List[np.ndarray] = []
    codes_list: List[np.ndarray] = []
    zc_list: List[np.ndarray] = []
    zs_list: List[np.ndarray] = []
    mert_list: List[np.ndarray] = []
    gidx_list: List[int] = []

    for i, rec in samples_df.reset_index(drop=True).iterrows():
        p = Path(str(rec["path"]))
        if not p.exists():
            continue
        starts = _pick_chunk_starts(
            path=p,
            chunk_seconds=float(codec.cfg.chunk_seconds),
            chunks_per_track=int(chunks_per_track),
            chunk_sampling=str(chunk_sampling),
            min_start_sec=float(min_start_sec),
            max_start_sec=max_start_sec,
            rng=rng,
        )
        for chunk_id, start_sec in enumerate(starts):
            try:
                wav_codec = codec.load_codec_chunk(path=p, start_sec=float(start_sec))
                q_emb, codes = codec.encode_chunk_embeddings(wav_codec)
                q_emb = codec.fix_num_frames(q_emb, target_frames=target_frames).cpu().numpy().astype(np.float32)
                codes = codes[:, :target_frames].cpu().numpy().astype(np.int16)

                wav_lab1 = codec.resample_audio(
                    wav_codec,
                    sr_from=int(codec.cfg.sample_rate),
                    sr_to=int(lab1_encoder.cfg.sample_rate),
                )
                log_mel = extract_log_mel(wav_lab1, sr=int(lab1_encoder.cfg.sample_rate))
                log_mel = fix_log_mel_frames(log_mel, n_frames=int(lab1_n_frames))
                lat = lab1_encoder.infer_log_mel(log_mel)

                mert_feat: Optional[np.ndarray] = None
                if mert is not None:
                    wav_mert = codec.resample_audio(
                        wav_codec,
                        sr_from=int(codec.cfg.sample_rate),
                        sr_to=int(mert.cfg.sample_rate),
                    )
                    mert_feat = mert.extract_features(wav_mert)
            except Exception:
                continue

            genre = str(rec["genre"])
            gidx = int(genre_to_idx[genre])
            rows.append(
                {
                    "sample_id": int(rec.get("sample_id", i)),
                    "path": str(p),
                    "track_id": _track_id(str(p)),
                    "source": str(rec["source"]),
                    "genre": genre,
                    "genre_idx": gidx,
                    "chunk_id": int(chunk_id),
                    "start_sec": float(start_sec),
                    "manifest_file": str(rec.get("manifest_file", "")),
                }
            )
            q_emb_list.append(q_emb)
            codes_list.append(codes)
            zc_list.append(lat["z_content"].astype(np.float32))
            zs_list.append(lat["z_style"].astype(np.float32))
            if mert_feat is not None:
                mert_list.append(mert_feat)
            gidx_list.append(gidx)

        if progress_every > 0 and (i + 1) % int(progress_every) == 0:
            print(f"[codec-cache] processed={i + 1}/{len(samples_df)} kept={len(rows)}")

    if not rows:
        raise RuntimeError("No codec cache rows were built. Check manifests and paths.")

    index_df = pd.DataFrame(rows)
    arrays = {
        "q_emb": np.stack(q_emb_list).astype(np.float32),
        "codes": np.stack(codes_list).astype(np.int16),
        "z_content": np.stack(zc_list).astype(np.float32),
        "z_style": np.stack(zs_list).astype(np.float32),
        "genre_idx": np.asarray(gidx_list, dtype=np.int64),
    }
    if mert_list:
        arrays["mert_feat"] = np.stack(mert_list).astype(np.float32)
    meta = CodecCacheMeta(
        codec_model_id=str(codec.cfg.model_id),
        codec_sample_rate=int(codec.cfg.sample_rate),
        codec_chunk_seconds=float(codec.cfg.chunk_seconds),
        codec_bandwidth=float(codec.cfg.bandwidth),
        codec_frames=int(target_frames),
        codec_channels=int(codec.cfg.latent_channels),
        lab1_n_frames=int(lab1_n_frames),
    )
    return index_df, arrays, genre_to_idx, meta


def save_codec_cache(
    cache_dir: Path,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    genre_to_idx: Dict[str, int],
    meta: CodecCacheMeta,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(cache_dir / "codec_cache_index.csv", index=False)
    save_kw = {
        "q_emb": arrays["q_emb"],
        "codes": arrays["codes"],
        "z_content": arrays["z_content"],
        "z_style": arrays["z_style"],
        "genre_idx": arrays["genre_idx"],
    }
    if "mert_feat" in arrays:
        save_kw["mert_feat"] = arrays["mert_feat"]
    np.savez_compressed(cache_dir / "codec_cache_arrays.npz", **save_kw)
    with (cache_dir / "codec_genre_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in genre_to_idx.items()}, f, indent=2)
    with (cache_dir / "codec_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)


def load_codec_cache(cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int], CodecCacheMeta]:
    cache_dir = Path(cache_dir)
    idx_path = cache_dir / "codec_cache_index.csv"
    npz_path = cache_dir / "codec_cache_arrays.npz"
    gmap_path = cache_dir / "codec_genre_to_idx.json"
    meta_path = cache_dir / "codec_meta.json"
    if not idx_path.exists() or not npz_path.exists() or not gmap_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Codec cache files missing in {cache_dir}")
    index_df = pd.read_csv(idx_path)
    z = np.load(npz_path)
    arrays = {
        "q_emb": z["q_emb"].astype(np.float32),
        "codes": z["codes"].astype(np.int16),
        "z_content": z["z_content"].astype(np.float32),
        "z_style": z["z_style"].astype(np.float32),
        "genre_idx": z["genre_idx"].astype(np.int64),
    }
    if "mert_feat" in z:
        arrays["mert_feat"] = z["mert_feat"].astype(np.float32)
    with gmap_path.open("r", encoding="utf-8") as f:
        genre_to_idx = json.load(f)
    genre_to_idx = {str(k): int(v) for k, v in genre_to_idx.items()}
    with meta_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = CodecCacheMeta(
        codec_model_id=str(raw["codec_model_id"]),
        codec_sample_rate=int(raw["codec_sample_rate"]),
        codec_chunk_seconds=float(raw["codec_chunk_seconds"]),
        codec_bandwidth=float(raw["codec_bandwidth"]),
        codec_frames=int(raw["codec_frames"]),
        codec_channels=int(raw["codec_channels"]),
        lab1_n_frames=int(raw["lab1_n_frames"]),
    )
    return index_df, arrays, genre_to_idx, meta


class CachedCodecDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.q_emb = arrays["q_emb"][indices]
        self.codes = arrays["codes"][indices]
        self.z_content = arrays["z_content"][indices]
        self.z_style = arrays["z_style"][indices]
        self.genre_idx = arrays["genre_idx"][indices]
        self.mert_feat = arrays["mert_feat"][indices] if "mert_feat" in arrays else None

    def __len__(self) -> int:
        return int(len(self.genre_idx))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        out = {
            "q_emb": torch.from_numpy(self.q_emb[i]),
            "codes": torch.from_numpy(self.codes[i]).long(),
            "z_content": torch.from_numpy(self.z_content[i]),
            "z_style": torch.from_numpy(self.z_style[i]),
            "genre_idx": torch.tensor(int(self.genre_idx[i]), dtype=torch.long),
        }
        if self.mert_feat is not None:
            out["mert_feat"] = torch.from_numpy(self.mert_feat[i])
        return out


def build_genre_exemplar_index(genre_idx: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    g = np.asarray(genre_idx, dtype=np.int64)
    for cls in sorted(np.unique(g).tolist()):
        out[int(cls)] = np.where(g == int(cls))[0].astype(np.int64)
    return out


__all__ = [
    "DEFAULT_MANIFESTS",
    "assign_genres",
    "build_codec_cache",
    "build_genre_exemplar_index",
    "CachedCodecDataset",
    "CodecCacheMeta",
    "load_codec_cache",
    "load_manifests",
    "materialize_genre_samples",
    "save_codec_cache",
    "stratified_group_split_indices",
    "stratified_split_indices",
]
