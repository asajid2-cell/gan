from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .lab3_bridge import FrozenLab1Encoder


DEFAULT_MANIFESTS = [
    "xtc_audio_clean.csv",
    "hh_lfbb_audio_clean.csv",
    "cc0_audio_clean.csv",
    "phase1_symbolic_audio_manifest.csv",
]

BAROQUE_REGEX = re.compile(
    r"(bach|goldberg|minuet|fugue|sonata|baroque|bwv|vivaldi|handel|mozart|beethoven|prelude|concerto)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GenreRule:
    genre: str
    source_include: Optional[Sequence[str]] = None
    source_exclude: Optional[Sequence[str]] = None
    path_include_regex: Optional[re.Pattern] = None
    path_exclude_regex: Optional[re.Pattern] = None

    def matches(self, source: str, path: str) -> bool:
        if self.source_include is not None and source not in self.source_include:
            return False
        if self.source_exclude is not None and source in self.source_exclude:
            return False
        if self.path_include_regex is not None and not self.path_include_regex.search(path):
            return False
        if self.path_exclude_regex is not None and self.path_exclude_regex.search(path):
            return False
        return True


DEFAULT_GENRE_RULES = [
    GenreRule(genre="baroque_classical", source_include=["phase1_pdmx"]),
    GenreRule(
        genre="baroque_classical",
        source_include=["cc0_audio_clean"],
        path_include_regex=BAROQUE_REGEX,
    ),
    GenreRule(genre="hiphop_xtc", source_include=["xtc_audio_clean"]),
    GenreRule(genre="lofi_hh_lfbb", source_include=["hh_lfbb_audio_clean"]),
    GenreRule(
        genre="cc0_other",
        source_include=["cc0_audio_clean"],
        path_exclude_regex=BAROQUE_REGEX,
    ),
]

GENRE_SCHEMAS = {
    # Original Lab3 labeling.
    "default4": {
        "remap": None,
    },
    # Decouple "style" from single-source proxies by merging into 2 multi-source buckets.
    # acoustic: phase1_pdmx + cc0_audio_clean (both baroque_classical and cc0_other)
    # beats: xtc_audio_clean + hh_lfbb_audio_clean
    "binary_acoustic_beats": {
        "remap": {
            "baroque_classical": "acoustic",
            "cc0_other": "acoustic",
            "hiphop_xtc": "beats",
            "lofi_hh_lfbb": "beats",
        },
    },
}


def _read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError(f"Manifest missing 'path': {path}")
    if "source" not in df.columns:
        df["source"] = path.stem
    # Keep core columns plus any genre-related columns if present (for auto-labeling workflows).
    keep_core = ["source", "path", "ext", "size_bytes", "is_music"]
    keep_genre = [c for c in df.columns if str(c).startswith("genre")]
    keep = [c for c in (keep_core + keep_genre) if c in df.columns]
    out = df[keep].copy()
    out["manifest_file"] = path.name
    out["path"] = out["path"].astype(str)
    out["source"] = out["source"].astype(str)
    return out


def load_manifests(manifests_root: Path, manifest_files: Optional[Iterable[str]] = None) -> pd.DataFrame:
    manifests_root = Path(manifests_root)
    files = list(manifest_files or DEFAULT_MANIFESTS)
    parts: List[pd.DataFrame] = []
    for mf in files:
        p = manifests_root / mf
        if p.exists():
            parts.append(_read_manifest(p))
    if not parts:
        raise FileNotFoundError(f"No manifests loaded from {manifests_root}.")
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return out


def assign_genres(df: pd.DataFrame, rules: Sequence[GenreRule] = DEFAULT_GENRE_RULES) -> pd.DataFrame:
    out = df.copy()
    # If a manifest already provides a genre column (e.g., auto-labeled), treat it as authoritative.
    # This avoids leaking source-based fallback labels back into content-labeled workflows.
    has_manifest_genre = "genre" in out.columns
    genres: List[str] = []
    for _, r in out.iterrows():
        s = str(r["source"])
        p = str(r["path"])
        g = "unassigned"
        if has_manifest_genre:
            gv = r.get("genre", None)
            if gv is None:
                gv = ""
            gv = str(gv).strip()
            if gv and gv.lower() not in {"nan", "none"}:
                g = gv
            genres.append(g)
            continue
        for rule in rules:
            if rule.matches(s, p):
                g = rule.genre
                break
        genres.append(g)
    out["genre"] = genres
    return out


def apply_genre_schema(assigned_df: pd.DataFrame, schema: str = "default4") -> pd.DataFrame:
    """
    Remap assigned_df['genre'] into a schema that better supports unpaired transfer.
    """
    s = str(schema).strip()
    if not s or s == "default4":
        return assigned_df
    if s not in GENRE_SCHEMAS:
        raise ValueError(f"Unknown genre schema '{s}'. Available: {sorted(GENRE_SCHEMAS.keys())}")
    remap = GENRE_SCHEMAS[s].get("remap")
    if not remap:
        return assigned_df
    out = assigned_df.copy()
    out["genre"] = out["genre"].astype(str).map(lambda g: str(remap.get(str(g), str(g))))
    return out


def genre_source_table(df: pd.DataFrame) -> pd.DataFrame:
    if "genre" not in df.columns or "source" not in df.columns:
        return pd.DataFrame(columns=["genre", "source", "count"])
    return (
        df.groupby(["genre", "source"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["genre", "source"], kind="stable")
        .reset_index(drop=True)
    )


def genre_num_sources(df: pd.DataFrame) -> Dict[str, int]:
    if "genre" not in df.columns or "source" not in df.columns:
        return {}
    out: Dict[str, int] = {}
    for g, gdf in df.groupby("genre", sort=True):
        out[str(g)] = int(gdf["source"].astype(str).nunique())
    return out


def _path_exists(path_str: str) -> bool:
    try:
        return Path(path_str).exists()
    except OSError:
        return False


def materialize_genre_samples(
    assigned_df: pd.DataFrame,
    per_genre_samples: int,
    seed: int = 328,
    drop_unassigned: bool = True,
    require_existing_paths: bool = True,
    require_is_music: bool = False,
) -> pd.DataFrame:
    if "genre" not in assigned_df.columns:
        raise ValueError("Expected 'genre' column.")
    df = assigned_df.copy()
    if drop_unassigned:
        df = df[df["genre"] != "unassigned"].copy()
    if require_existing_paths:
        df = df[df["path"].map(_path_exists)].copy()
    if bool(require_is_music) and "is_music" in df.columns:
        # Manifests can have missing is_music values (e.g., older sources without audit).
        # If it's missing, treat as "unknown" and keep it; only drop explicit 0s.
        is_music = pd.to_numeric(df["is_music"], errors="coerce").fillna(1).astype(int)
        df = df[is_music == 1].copy()

    out_parts: List[pd.DataFrame] = []
    for genre, gdf in df.groupby("genre", sort=True):
        if len(gdf) == 0:
            continue
        n = min(int(per_genre_samples), len(gdf))
        if n > 0:
            out_parts.append(gdf.sample(n=n, random_state=seed))
    if not out_parts:
        raise ValueError("No samples materialized from assigned genres.")
    out = pd.concat(out_parts, ignore_index=True).reset_index(drop=True)
    out["sample_id"] = out.index.astype(int)
    return out


def materialize_genre_samples_balanced_sources(
    assigned_df: pd.DataFrame,
    per_genre_samples: int,
    seed: int = 328,
    drop_unassigned: bool = True,
    require_existing_paths: bool = True,
    require_is_music: bool = False,
) -> pd.DataFrame:
    """
    Sample per-genre while balancing across sources within each genre.
    This reduces source/dataset fingerprint dominance in training.
    """
    if "genre" not in assigned_df.columns:
        raise ValueError("Expected 'genre' column.")
    if "source" not in assigned_df.columns:
        raise ValueError("Expected 'source' column.")

    df = assigned_df.copy()
    if drop_unassigned:
        df = df[df["genre"] != "unassigned"].copy()
    if require_existing_paths:
        df = df[df["path"].map(_path_exists)].copy()
    if bool(require_is_music) and "is_music" in df.columns:
        # Manifests can have missing is_music values (e.g., older sources without audit).
        # If it's missing, treat as "unknown" and keep it; only drop explicit 0s.
        is_music = pd.to_numeric(df["is_music"], errors="coerce").fillna(1).astype(int)
        df = df[is_music == 1].copy()

    rng = np.random.default_rng(int(seed))
    out_parts: List[pd.DataFrame] = []
    for genre, gdf in df.groupby("genre", sort=True):
        if len(gdf) == 0:
            continue
        sources = sorted(gdf["source"].astype(str).unique().tolist())
        if not sources:
            continue
        per_src = max(1, int(per_genre_samples) // max(1, len(sources)))
        picked: List[pd.DataFrame] = []
        for s in sources:
            sdf = gdf[gdf["source"].astype(str) == str(s)]
            if len(sdf) == 0:
                continue
            n = min(int(per_src), len(sdf))
            if n > 0:
                # Keep original indices so we can top-up without duplicating rows.
                picked.append(sdf.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
        merged = pd.concat(picked, axis=0) if picked else gdf.head(0).copy()
        # Top up to per_genre_samples if possible.
        if len(merged) < int(per_genre_samples):
            need = int(per_genre_samples) - int(len(merged))
            rem = gdf.drop(index=merged.index, errors="ignore")
            if len(rem) > 0:
                extra = rem.sample(n=min(need, len(rem)), random_state=int(rng.integers(0, 2**31 - 1)))
                merged = pd.concat([merged, extra], axis=0)
        if len(merged) > int(per_genre_samples):
            merged = merged.sample(n=int(per_genre_samples), random_state=int(rng.integers(0, 2**31 - 1)))
        out_parts.append(merged)

    if not out_parts:
        raise ValueError("No samples materialized from assigned genres.")
    out = pd.concat(out_parts, axis=0).reset_index(drop=True)
    out["sample_id"] = out.index.astype(int)
    return out


def genre_count_table(df: pd.DataFrame) -> Dict[str, int]:
    if "genre" not in df.columns:
        return {}
    vc = df["genre"].value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def build_latent_cache(
    samples_df: pd.DataFrame,
    encoder: FrozenLab1Encoder,
    cache_dir: Path,
    n_frames: int,
    chunks_per_track: int = 1,
    chunk_sampling: str = "uniform",
    min_start_sec: float = 0.0,
    max_start_sec: Optional[float] = None,
    seed: int = 328,
    progress_every: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    genres = sorted(samples_df["genre"].astype(str).unique().tolist())
    genre_to_idx = {g: i for i, g in enumerate(genres)}

    rows: List[Dict] = []
    mel_norm_list: List[np.ndarray] = []
    zc_list: List[np.ndarray] = []
    zs_list: List[np.ndarray] = []
    gidx_list: List[int] = []

    rng = np.random.default_rng(int(seed))
    duration_cache: Dict[str, float] = {}

    def _track_id(path_str: str) -> str:
        return hashlib.md5(path_str.encode("utf-8")).hexdigest()[:16]

    def _duration_seconds(path_obj: Path) -> float:
        key = str(path_obj)
        if key in duration_cache:
            return duration_cache[key]
        dur = 0.0
        try:
            dur = float(librosa.get_duration(path=str(path_obj)))
        except Exception:
            dur = 0.0
        duration_cache[key] = dur
        return dur

    def _pick_starts(path_obj: Path) -> List[float]:
        n_chunks = max(1, int(chunks_per_track))
        dur = _duration_seconds(path_obj)
        chunk_sec = float(encoder.cfg.chunk_seconds)
        lo = max(0.0, float(min_start_sec))
        if max_start_sec is None:
            hi_raw = max(0.0, dur - chunk_sec)
        else:
            hi_raw = max(0.0, float(max_start_sec))
        hi = max(lo, hi_raw)
        if n_chunks == 1 or hi <= lo + 1e-9:
            return [float(lo)] * n_chunks
        mode = str(chunk_sampling).strip().lower()
        if mode == "random":
            return [float(rng.uniform(lo, hi)) for _ in range(n_chunks)]
        # uniform default
        vals = np.linspace(lo, hi, num=n_chunks, dtype=np.float64)
        return [float(v) for v in vals]

    for i, r in samples_df.reset_index(drop=True).iterrows():
        p = Path(str(r["path"]))
        if not p.exists():
            continue
        starts = _pick_starts(p)
        for chunk_id, start_sec in enumerate(starts):
            out = encoder.infer_file_with_mel(path=p, n_frames=n_frames, start_sec=float(start_sec))
            if out is None:
                continue
            genre = str(r["genre"])
            gidx = int(genre_to_idx[genre])
            rows.append(
                {
                    "sample_id": int(r.get("sample_id", i)),
                    "path": str(p),
                    "track_id": _track_id(str(p)),
                    "source": str(r["source"]),
                    "genre": genre,
                    "genre_idx": gidx,
                    "chunk_id": int(chunk_id),
                    "start_sec": float(start_sec),
                    "manifest_file": str(r.get("manifest_file", "")),
                }
            )
            mel_norm_list.append(out["mel_norm"])
            zc_list.append(out["z_content"])
            zs_list.append(out["z_style"])
            gidx_list.append(gidx)

        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[cache] processed={i + 1}/{len(samples_df)} kept={len(rows)}")

    if not rows:
        raise RuntimeError("No cache rows were built. Check manifests and paths.")

    index_df = pd.DataFrame(rows)
    arrays = {
        "mel_norm": np.stack(mel_norm_list).astype(np.float32),
        "z_content": np.stack(zc_list).astype(np.float32),
        "z_style": np.stack(zs_list).astype(np.float32),
        "genre_idx": np.asarray(gidx_list, dtype=np.int64),
    }
    return index_df, arrays, genre_to_idx


def save_cache(
    cache_dir: Path,
    index_df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    genre_to_idx: Dict[str, int],
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(cache_dir / "cache_index.csv", index=False)
    np.savez_compressed(
        cache_dir / "cache_arrays.npz",
        mel_norm=arrays["mel_norm"],
        z_content=arrays["z_content"],
        z_style=arrays["z_style"],
        genre_idx=arrays["genre_idx"],
    )
    with (cache_dir / "genre_to_idx.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(genre_to_idx, f, indent=2)


def load_cache(cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, int]]:
    cache_dir = Path(cache_dir)
    idx_path = cache_dir / "cache_index.csv"
    npz_path = cache_dir / "cache_arrays.npz"
    gmap_path = cache_dir / "genre_to_idx.json"
    if not idx_path.exists() or not npz_path.exists() or not gmap_path.exists():
        raise FileNotFoundError(f"Cache files missing in {cache_dir}")
    index_df = pd.read_csv(idx_path)
    z = np.load(npz_path)
    arrays = {
        "mel_norm": z["mel_norm"].astype(np.float32),
        "z_content": z["z_content"].astype(np.float32),
        "z_style": z["z_style"].astype(np.float32),
        "genre_idx": z["genre_idx"].astype(np.int64),
    }
    import json

    with gmap_path.open("r", encoding="utf-8") as f:
        genre_to_idx = json.load(f)
    genre_to_idx = {str(k): int(v) for k, v in genre_to_idx.items()}
    return index_df, arrays, genre_to_idx


class CachedSynthesisDataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray], indices: np.ndarray):
        self.mel_norm = arrays["mel_norm"][indices]
        self.z_content = arrays["z_content"][indices]
        self.z_style = arrays["z_style"][indices]
        self.genre_idx = arrays["genre_idx"][indices]

    def __len__(self) -> int:
        return int(len(self.genre_idx))

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "mel_norm": torch.from_numpy(self.mel_norm[i]),
            "z_content": torch.from_numpy(self.z_content[i]),
            "z_style": torch.from_numpy(self.z_style[i]),
            "genre_idx": torch.tensor(int(self.genre_idx[i]), dtype=torch.long),
        }


def stratified_split_indices(
    genre_idx: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 328,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(genre_idx))
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for g in sorted(np.unique(genre_idx).tolist()):
        gi = idx[genre_idx == g]
        gi = rng.permutation(gi)
        n_val = max(1, int(round(len(gi) * float(val_ratio)))) if len(gi) > 1 else 0
        val_parts.append(gi[:n_val])
        train_parts.append(gi[n_val:])
    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    if len(train_idx) == 0:
        train_idx = val_idx.copy()
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = train_idx[: min(64, len(train_idx))].copy()
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def stratified_group_split_indices(
    genre_idx: np.ndarray,
    group_ids: np.ndarray,
    val_ratio: float = 0.15,
    seed: int = 328,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split without group leakage (e.g., track_id).
    Assumes each group has a single stable genre in this dataset.
    """
    if len(genre_idx) != len(group_ids):
        raise ValueError("genre_idx and group_ids must have same length.")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "idx": np.arange(len(genre_idx), dtype=np.int64),
            "genre_idx": np.asarray(genre_idx, dtype=np.int64),
            "group_id": np.asarray(group_ids).astype(str),
        }
    )
    grp = df.groupby("group_id", sort=False)["genre_idx"].agg(lambda x: int(x.mode().iloc[0])).reset_index()
    train_groups: List[str] = []
    val_groups: List[str] = []
    for g in sorted(grp["genre_idx"].unique().tolist()):
        g_groups = grp.loc[grp["genre_idx"] == g, "group_id"].to_numpy()
        if len(g_groups) == 0:
            continue
        g_groups = rng.permutation(g_groups)
        n_val = max(1, int(round(len(g_groups) * float(val_ratio)))) if len(g_groups) > 1 else 0
        val_groups.extend(g_groups[:n_val].tolist())
        train_groups.extend(g_groups[n_val:].tolist())

    train_set = set(train_groups)
    val_set = set(val_groups)
    if train_set & val_set:
        raise RuntimeError("Group leakage detected in stratified_group_split_indices.")

    train_idx = df.loc[df["group_id"].isin(train_set), "idx"].to_numpy(dtype=np.int64)
    val_idx = df.loc[df["group_id"].isin(val_set), "idx"].to_numpy(dtype=np.int64)

    if len(train_idx) == 0:
        train_idx = val_idx.copy()
    if len(val_idx) == 0 and len(train_idx) > 1:
        val_idx = train_idx[: min(64, len(train_idx))].copy()
    return train_idx.astype(np.int64), val_idx.astype(np.int64)
