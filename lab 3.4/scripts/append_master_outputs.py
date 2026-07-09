from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
MASTER_ROOT = REPO_ROOT / "Master Outputs"
MANIFEST_PATH = MASTER_ROOT / "manifest.csv"
STAMP_RE = re.compile(r"(20\d{6})[_-](\d{6})")
INDICATOR_FILES = {
    "summary.json",
    "manifest.csv",
    "diagnosis_report.md",
    "winner_map.json",
    "history.json",
}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _extract_sort_key(path: Path) -> Optional[str]:
    joined = str(path).replace("-", "_")
    matches = STAMP_RE.findall(joined)
    if not matches:
        return None
    ymd, hms = matches[-1]
    return f"{ymd}_{hms}"


def _has_audio(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
                return True
    except OSError:
        return False
    return False


def _qualifies(path: Path) -> bool:
    if not path.is_dir():
        return False
    names = set()
    try:
        for child in path.iterdir():
            names.add(child.name)
            if child.is_file() and child.suffix.lower() in AUDIO_EXTS:
                return True
    except OSError:
        return False
    return bool(names & INDICATOR_FILES)


def _relative_slug(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT)
    raw = str(rel).replace("\\", "__").replace("/", "__").replace(" ", "_").replace(":", "")
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw[:160]


def _load_existing() -> List[Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _make_junction(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        return
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
        check=True,
        cwd=str(MASTER_ROOT),
        capture_output=True,
        text=True,
    )


def _collect_candidates(add_root: Path) -> List[Path]:
    roots: List[Path] = []
    if _qualifies(add_root):
        roots.append(add_root)
    for path in add_root.rglob("*"):
        if _qualifies(path):
            roots.append(path)
    dedup = sorted({p.resolve() for p in roots}, key=lambda p: (str(_extract_sort_key(p) or ""), str(p)))
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser(description="Append new timestamped output directories into Master Outputs.")
    ap.add_argument("--add-root", type=Path, required=True)
    args = ap.parse_args()

    add_root = Path(args.add_root).resolve()
    MASTER_ROOT.mkdir(parents=True, exist_ok=True)
    existing_rows = _load_existing()
    existing_paths = {str(Path(row["full_path"]).resolve()) for row in existing_rows}
    next_order = len(existing_rows) + 1
    new_rows: List[Dict[str, str]] = []

    for path in _collect_candidates(add_root):
        full_path = str(path.resolve())
        if full_path in existing_paths:
            continue
        sort_key = _extract_sort_key(path)
        if not sort_key:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        link_name = f"{next_order:04d}_{sort_key}_{_relative_slug(path)}"
        link_path = MASTER_ROOT / link_name
        _make_junction(link_path, path)
        row = {
            "order": str(next_order),
            "sort_key": sort_key,
            "link_name": link_name,
            "relative_path": rel,
            "full_path": full_path,
            "has_audio": str(_has_audio(path)),
            "has_manifest": str((path / "manifest.csv").exists()),
            "has_summary": str((path / "summary.json").exists()),
            "has_epoch_samples": str((path / "epoch_samples").exists()),
        }
        new_rows.append(row)
        existing_rows.append(row)
        existing_paths.add(full_path)
        next_order += 1

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "order",
                "sort_key",
                "link_name",
                "relative_path",
                "full_path",
                "has_audio",
                "has_manifest",
                "has_summary",
                "has_epoch_samples",
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(existing_rows)

    print({"added": len(new_rows), "last_order": next_order - 1, "manifest": str(MANIFEST_PATH)})


if __name__ == "__main__":
    main()
