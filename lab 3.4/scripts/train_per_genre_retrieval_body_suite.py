from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


TARGET_CONFIGS = {
    "baroque_classical": {"epochs": 2, "max_batches": 420},
    "hiphop_xtc": {"epochs": 2, "max_batches": 420},
    "lofi_hh_lfbb": {"epochs": 2, "max_batches": 420},
    "cc0_other": {"epochs": 2, "max_batches": 360},
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs in {root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _run_target(script: Path, suite_root: Path, target: str, max_frames: int) -> Dict[str, Any]:
    target_root = suite_root / target
    target_root.mkdir(parents=True, exist_ok=True)
    cfg = TARGET_CONFIGS[target]
    cmd = [
        "python",
        "-u",
        str(script),
        "--out-root",
        str(target_root),
        "--epochs",
        str(int(cfg["epochs"])),
        "--max-batches-per-epoch",
        str(int(cfg["max_batches"])),
        "--batch-size",
        "2",
        "--max-frames",
        str(int(max_frames)),
        "--single-genre-target",
        str(target),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{target} run failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    run_dir = _latest_run(target_root)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"target": target, "run_dir": str(run_dir), "summary": summary, "stdout": proc.stdout}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-genre scratch retrieval body-style models and aggregate their production pack.")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "Desktop Outputs" / "dggr_per_genre_retrieval_body_style_suite")
    ap.add_argument("--max-frames", type=int, default=320)
    args = ap.parse_args()

    suite_dir = Path(args.out_root) / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style.py"

    target_runs: List[Dict[str, Any]] = []
    for target in ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]:
        target_runs.append(_run_target(script, suite_dir, target, max_frames=int(args.max_frames)))

    combined_dir = suite_dir / "combined_pack"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    combined_rows: List[Dict[str, Any]] = []
    for result in target_runs:
        target = str(result["target"])
        final_pack = Path(str(result["summary"]["final_pack_dir"]))
        rows = list(result["summary"]["final_summary"]["rows"])
        for row in rows:
            song = str(row["song"])
            src_dir = final_pack / "renders" / song / target
            dst_dir = combined_dir / song / target
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ["hybrid_longform_coherent.wav", "accompaniment_generated.wav", "longform_coherent.wav", "backing_fixed.wav"]:
                src = src_dir / name
                if src.exists():
                    shutil.copyfile(src, dst_dir / name)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
                writer.writerow(
                    {
                        "target": target,
                        "source_song": song,
                        "source_target_dir": str(src_dir),
                        "hybrid_wav": str(dst_dir / "hybrid_longform_coherent.wav"),
                        "accompaniment_wav": str(dst_dir / "accompaniment_generated.wav"),
                    }
                )
            combined_rows.append(dict(row))

    summary = {
        "suite_dir": str(suite_dir),
        "combined_pack": str(combined_dir),
        "target_runs": target_runs,
        "mean_overall": float(sum(float(r["overall"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_target_conf": float(sum(float(r["target_conf"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_target_margin": float(sum(float(r["target_margin"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_warble": float(sum(float(r["warble"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_fullness": float(sum(float(r["fullness"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_structure": float(sum(float(r["structure"]) for r in combined_rows) / max(1, len(combined_rows))),
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (suite_dir / "winner_map.json").write_text(
        json.dumps({str(r["target"]): str(Path(str(r["summary"]["best_checkpoint"]))) for r in target_runs}, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Per-Genre Retrieval Fusion Suite",
        "",
        "- Each target genre was trained as its own scratch retrieval-conditioned accompaniment generator with explicit body and groove retention losses.",
        f"- Mean overall: {summary['mean_overall']:.4f}",
        f"- Mean target confidence: {summary['mean_target_conf']:.4f}",
        f"- Mean target margin: {summary['mean_target_margin']:.4f}",
        f"- Mean fullness: {summary['mean_fullness']:.4f}",
        f"- Mean warble: {summary['mean_warble']:.4f}",
        f"- Mean structure: {summary['mean_structure']:.4f}",
    ]
    (suite_dir / "diagnosis_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
