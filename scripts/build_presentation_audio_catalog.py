from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
MASTER_OUTPUTS = ROOT / "Master Outputs"
MANIFEST = MASTER_OUTPUTS / "manifest.csv"
OUT_DIR = ROOT / "demo" / "data"
OUT_PATH = OUT_DIR / "presentation_audio_catalog.json"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
PREFERRED_NAMES = [
    "hybrid_longform_coherent.wav",
    "longform_coherent.wav",
    "backing_fixed.wav",
    "codec_generated.wav",
    "diffusion_generated.wav",
    "source.wav",
]


def encode_rel_path(path: Path) -> str:
    return "/".join(quote(part) for part in path.parts)


def audio_priority(name: str) -> tuple[int, str]:
    lowered = name.lower()
    for idx, preferred in enumerate(PREFERRED_NAMES):
        if lowered == preferred:
            return (idx, lowered)
    if lowered.startswith("chunk_"):
        return (100, lowered)
    return (50, lowered)


def build_catalog() -> dict:
    entries = []
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        full_path = Path(row["full_path"])
        if not full_path.exists() or not full_path.is_dir():
            continue

        direct_audio = []
        for child in sorted(full_path.iterdir()):
            if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS:
                try:
                    rel_to_root = child.relative_to(ROOT)
                except ValueError:
                    continue
                direct_audio.append(
                    {
                        "name": child.name,
                        "path": str(rel_to_root).replace("\\", "/"),
                        "url": "../" + encode_rel_path(rel_to_root),
                        "size_bytes": child.stat().st_size,
                    }
                )

        if not direct_audio:
            continue

        direct_audio.sort(key=lambda item: audio_priority(item["name"]))
        title = row["link_name"]
        family = row["relative_path"].replace("\\", " / ")
        entries.append(
            {
                "order": int(row["order"]),
                "sort_key": row["sort_key"],
                "link_name": row["link_name"],
                "title": title,
                "family": family,
                "source_path": row["relative_path"],
                "full_path": row["full_path"],
                "files": direct_audio,
            }
        )

    entries.sort(key=lambda item: (item["order"], item["sort_key"], item["title"]))
    flat_files = []
    for entry in entries:
        for file in entry["files"]:
            flat_files.append(
                {
                    "entry_order": entry["order"],
                    "entry_sort_key": entry["sort_key"],
                    "entry_title": entry["title"],
                    "entry_family": entry["family"],
                    "name": file["name"],
                    "path": file["path"],
                    "url": file["url"],
                    "size_bytes": file["size_bytes"],
                }
            )

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "entry_count": len(entries),
        "file_count": len(flat_files),
        "entries": entries,
        "files": flat_files,
    }


def main() -> None:
    catalog = build_catalog()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(json.dumps({"out_path": str(OUT_PATH), "entry_count": catalog["entry_count"], "file_count": catalog["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
