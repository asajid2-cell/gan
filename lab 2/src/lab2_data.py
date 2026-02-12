from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


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
    # Priority order matters. First match wins.
    GenreRule(
        genre="baroque_classical",
        source_include=["phase1_pdmx"],
    ),
    GenreRule(
        genre="baroque_classical",
        source_include=["cc0_audio_clean"],
        path_include_regex=BAROQUE_REGEX,
    ),
    GenreRule(
        genre="hiphop_xtc",
        source_include=["xtc_audio_clean"],
    ),
    GenreRule(
        genre="lofi_hh_lfbb",
        source_include=["hh_lfbb_audio_clean"],
    ),
    GenreRule(
        genre="cc0_other",
        source_include=["cc0_audio_clean"],
        path_exclude_regex=BAROQUE_REGEX,
    ),
]


def _read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError(f"Manifest missing 'path' column: {path}")
    if "source" not in df.columns:
        default_source = path.stem
        df["source"] = default_source
    keep = [c for c in ["source", "path", "ext", "size_bytes", "is_music"] if c in df.columns]
    out = df[keep].copy()
    out["manifest_file"] = path.name
    out["path"] = out["path"].astype(str)
    out["source"] = out["source"].astype(str)
    return out


def load_manifests(manifests_root: Path, manifest_files: Optional[Iterable[str]] = None) -> pd.DataFrame:
    manifests_root = Path(manifests_root)
    files = list(manifest_files or DEFAULT_MANIFESTS)
    chunks: List[pd.DataFrame] = []
    for mf in files:
        p = manifests_root / mf
        if not p.exists():
            continue
        chunks.append(_read_manifest(p))
    if not chunks:
        raise FileNotFoundError(
            f"No manifests were loaded from {manifests_root}. Checked: {files}"
        )
    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["path"]).reset_index(drop=True)
    return out


def assign_genres(df: pd.DataFrame, rules: Sequence[GenreRule] = DEFAULT_GENRE_RULES) -> pd.DataFrame:
    out = df.copy()
    genres: List[str] = []
    for _, row in out.iterrows():
        src = str(row["source"])
        path = str(row["path"])
        g = "unassigned"
        for rule in rules:
            if rule.matches(src, path):
                g = rule.genre
                break
        genres.append(g)
    out["genre"] = genres
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
) -> pd.DataFrame:
    if "genre" not in assigned_df.columns:
        raise ValueError("Expected 'genre' column. Call assign_genres first.")
    df = assigned_df.copy()
    if drop_unassigned:
        df = df[df["genre"] != "unassigned"].copy()
    if require_existing_paths:
        df = df[df["path"].map(_path_exists)].copy()

    sampled: List[pd.DataFrame] = []
    for genre, gdf in df.groupby("genre", sort=True):
        if len(gdf) == 0:
            continue
        n = min(int(per_genre_samples), len(gdf))
        if n <= 0:
            continue
        sampled.append(gdf.sample(n=n, random_state=seed))

    if not sampled:
        raise ValueError("No genre samples could be materialized with the current rules.")

    out = pd.concat(sampled, ignore_index=True).reset_index(drop=True)
    out["sample_id"] = out.index.astype(int)
    return out


def genre_count_table(df: pd.DataFrame) -> Dict[str, int]:
    if "genre" not in df.columns:
        return {}
    vc = df["genre"].value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}

