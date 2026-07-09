from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
DEFAULT_REAL_MUSIC_ROOT = Path(
    r"Z:\DataSets\annas_archive_data__aacid__spotify_files_pop_0__20260116T000346Z--20260116T000347Z"
)


@dataclass(frozen=True)
class RealMusicSource:
    root: Path
    genre: str
    source: str = ""


def infer_genre_from_path(path: Path, fallback: str = "real_music") -> str:
    text = str(path).replace("\\", "/").lower()
    patterns = [
        r"spotify_files_([a-z0-9]+)_\d+",
        r"(?:^|/)(pop|rock|hiphop|hip_hop|rap|classical|baroque|jazz|metal|country|electronic|edm|folk|rnb|soul)(?:_|-|/|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _normalize_genre(m.group(1))
    return _normalize_genre(fallback)


def _normalize_genre(raw: str) -> str:
    g = str(raw).strip().lower()
    g = re.sub(r"[^a-z0-9]+", "_", g).strip("_")
    aliases = {
        "hip_hop": "hiphop",
        "rap": "hiphop",
        "edm": "electronic",
        "baroque": "baroque_classical",
        "classical": "baroque_classical",
        "r_b": "rnb",
        "rhythm_and_blues": "rnb",
    }
    return aliases.get(g, g or "real_music")


def parse_source_specs(specs: Optional[Sequence[str]], default_root: Path = DEFAULT_REAL_MUSIC_ROOT) -> List[RealMusicSource]:
    if not specs:
        return [RealMusicSource(root=default_root, genre=infer_genre_from_path(default_root), source="spotify_pop_0")]
    out: List[RealMusicSource] = []
    for spec in specs:
        raw = str(spec).strip()
        if not raw:
            continue
        if "=" in raw:
            genre_raw, root_raw = raw.split("=", 1)
            root = Path(root_raw.strip().strip('"'))
            genre = _normalize_genre(genre_raw)
        else:
            root = Path(raw.strip().strip('"'))
            genre = infer_genre_from_path(root)
        source = f"real_{genre}_{len(out):02d}"
        out.append(RealMusicSource(root=root, genre=genre, source=source))
    if not out:
        raise ValueError("No real music source specs were provided.")
    return out


def _read_export_report(root: Path) -> Optional[pd.DataFrame]:
    report = root / "_ogg_export_report.csv"
    if not report.exists():
        return None
    df = pd.read_csv(report)
    if "ExportPath" not in df.columns:
        return None
    return df


def _rows_from_export_report(source: RealMusicSource, min_bytes: int) -> List[Dict[str, object]]:
    df = _read_export_report(source.root)
    if df is None:
        return []
    rows: List[Dict[str, object]] = []
    for _, rec in df.iterrows():
        status = str(rec.get("Status", "")).strip().lower()
        if status and status != "exported":
            continue
        path = Path(str(rec.get("ExportPath", "")).strip())
        if not path.exists():
            continue
        size = int(path.stat().st_size)
        if size < int(min_bytes):
            continue
        rows.append(
            {
                "source": source.source,
                "path": str(path),
                "ext": path.suffix.lower().lstrip("."),
                "size_bytes": size,
                "is_music": 1,
                "genre": source.genre,
                "title": str(rec.get("Title", "") or ""),
                "artist": str(rec.get("Artist", "") or ""),
                "album": str(rec.get("Album", "") or ""),
                "spotify_url": str(rec.get("SpotifyURL", "") or ""),
                "aacid": str(rec.get("AACID", "") or ""),
                "source_root": str(source.root),
            }
        )
    return rows


def _iter_audio_files(root: Path, prefer_ogg_export: bool) -> Iterable[Path]:
    search_roots = [root / "_ogg_export"] if prefer_ogg_export and (root / "_ogg_export").exists() else []
    search_roots.append(root)
    seen: set[str] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        stack = [search_root]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    entries = list(it)
            except OSError:
                continue
            for entry in entries:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(path)
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                if path.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                key = str(path).lower()
                if key in seen:
                    continue
                seen.add(key)
                yield path


def _rows_from_directory(source: RealMusicSource, min_bytes: int, prefer_ogg_export: bool) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in _iter_audio_files(source.root, prefer_ogg_export=prefer_ogg_export):
        size = int(path.stat().st_size)
        if size < int(min_bytes):
            continue
        rows.append(
            {
                "source": source.source,
                "path": str(path),
                "ext": path.suffix.lower().lstrip("."),
                "size_bytes": size,
                "is_music": 1,
                "genre": source.genre,
                "title": "",
                "artist": "",
                "album": "",
                "spotify_url": "",
                "aacid": path.stem,
                "source_root": str(source.root),
            }
        )
    return rows


def build_real_music_manifest(
    sources: Sequence[RealMusicSource],
    out_path: Path,
    *,
    min_bytes: int = 64_000,
    max_files_per_source: int = 0,
    seed: int = 328,
    prefer_ogg_export: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for source in sources:
        if not source.root.exists():
            raise FileNotFoundError(f"Real music source root does not exist: {source.root}")
        source_rows = _rows_from_export_report(source, min_bytes=min_bytes)
        if int(max_files_per_source) > 0 and len(source_rows) >= int(max_files_per_source):
            source_rows = (
                pd.DataFrame(source_rows)
                .sample(n=int(max_files_per_source), random_state=int(seed))
                .to_dict(orient="records")
            )
            rows.extend(source_rows)
            continue
        report_paths = {str(Path(str(r["path"])).resolve()).lower() for r in source_rows}
        for row in _rows_from_directory(source, min_bytes=min_bytes, prefer_ogg_export=prefer_ogg_export):
            key = str(Path(str(row["path"])).resolve()).lower()
            if key not in report_paths:
                source_rows.append(row)
        if int(max_files_per_source) > 0 and len(source_rows) > int(max_files_per_source):
            source_rows = (
                pd.DataFrame(source_rows)
                .sample(n=int(max_files_per_source), random_state=int(seed))
                .to_dict(orient="records")
            )
        rows.extend(source_rows)
    if not rows:
        raise RuntimeError("No usable audio files found for the requested real music sources.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).reset_index(drop=True)
    df = df.sort_values(["genre", "path"], kind="stable").reset_index(drop=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df
