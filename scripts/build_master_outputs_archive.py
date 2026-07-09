from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
HARMONIZER_ROOT = REPO_ROOT.parent / "harmonizer"
MASTER_OUTPUTS = REPO_ROOT / "Master Outputs"
MANIFEST_PATH = MASTER_OUTPUTS / "manifest.csv"
SITE_ROOT = HARMONIZER_ROOT / "frontend" / "master-outputs"
MEDIA_ROOT = SITE_ROOT / "media"
DATA_ROOT = SITE_ROOT / "data"

PLAYABLE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
PREFERRED_AUDIO_NAMES = [
    "hybrid_longform_coherent.wav",
    "longform_coherent.wav",
    "accompaniment_generated.wav",
    "codec_generated.wav",
    "diffusion_generated.wav",
    "source_codec_rate.wav",
    "source.wav",
]


@dataclass
class EntryAudio:
    name: str
    relative_url: str
    size_bytes: int
    priority: int


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "entry"


def _audio_priority(name: str) -> int:
    lowered = name.lower()
    for idx, preferred in enumerate(PREFERRED_AUDIO_NAMES):
        if lowered == preferred:
            return idx
    if lowered.startswith("chunk_"):
        return 90
    return 50


def _iter_direct_audio_files(path: Path) -> Iterable[Path]:
    if not path.exists() or not path.is_dir():
        return []
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in PLAYABLE_EXTENSIONS]
    return sorted(files, key=lambda p: (_audio_priority(p.name), p.name.lower()))


def _entry_title(row: dict[str, str], path: Path) -> str:
    rel = row["relative_path"].replace("\\", " / ")
    if "__" in row["link_name"]:
        stem = row["link_name"].split("_", 2)[-1].replace("__", " / ")
        return stem
    return rel


def _entry_family(path: Path) -> str:
    parts = path.parts
    if not parts:
        return "archive"
    if parts[0].isdigit():
        return parts[0]
    if len(parts) >= 2:
        return f"{parts[0]} / {parts[1]}"
    return parts[0]


def _copy_audio(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def build_archive() -> dict[str, object]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    SITE_ROOT.mkdir(parents=True, exist_ok=True)
    MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(MANIFEST_PATH.open(encoding="utf-8")))
    entries: list[dict[str, object]] = []
    total_files = 0
    total_bytes = 0

    for row in rows:
        full_path = Path(row["full_path"])
        direct_audio = list(_iter_direct_audio_files(full_path))
        if not direct_audio:
            continue

        order = int(row["order"])
        entry_slug = f"{order:04d}-{_slugify(full_path.name)}"
        media_dir = MEDIA_ROOT / entry_slug
        files: list[EntryAudio] = []

        for audio in direct_audio:
            out_path = media_dir / audio.name
            _copy_audio(audio, out_path)
            files.append(
                EntryAudio(
                    name=audio.name,
                    relative_url=f"./media/{entry_slug}/{audio.name}",
                    size_bytes=audio.stat().st_size,
                    priority=_audio_priority(audio.name),
                )
            )
            total_files += 1
            total_bytes += audio.stat().st_size

        parts = row["relative_path"].replace("\\", "/").split("/")
        entry = {
            "id": entry_slug,
            "order": order,
            "sort_key": row["sort_key"],
            "link_name": row["link_name"],
            "title": _entry_title(row, full_path),
            "family": _entry_family(Path(*parts)),
            "path_label": row["relative_path"].replace("\\", " / "),
            "source_name": full_path.name,
            "audio_count": len(files),
            "files": [
                {
                    "name": file.name,
                    "url": file.relative_url,
                    "size_bytes": file.size_bytes,
                    "priority": file.priority,
                }
                for file in files
            ],
        }
        entries.append(entry)

    entries.sort(key=lambda item: (int(item["order"]), str(item["title"])))
    catalog = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "entry_count": len(entries),
        "audio_file_count": total_files,
        "audio_total_bytes": total_bytes,
        "entries": entries,
    }
    (DATA_ROOT / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


if __name__ == "__main__":
    catalog = build_archive()
    print(json.dumps(
        {
            "site_root": str(SITE_ROOT),
            "entry_count": catalog["entry_count"],
            "audio_file_count": catalog["audio_file_count"],
            "audio_total_gb": round(catalog["audio_total_bytes"] / 1024 / 1024 / 1024, 3),
        },
        indent=2,
    ))
