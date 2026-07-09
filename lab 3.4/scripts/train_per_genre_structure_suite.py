from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


TARGET_CONFIGS = {
    "baroque_classical": {"epochs": 1, "max_batches": 120},
    "hiphop_xtc": {"epochs": 1, "max_batches": 140},
    "lofi_hh_lfbb": {"epochs": 1, "max_batches": 120},
    "cc0_other": {"epochs": 1, "max_batches": 100},
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
        "1",
        "--max-frames",
        str(int(max_frames)),
        "--single-genre-target",
        str(target),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{target} run failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    run_dir = _latest_run(target_root)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing summary for {target}: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {"target": target, "run_dir": str(run_dir), "summary": summary, "stdout": proc.stdout}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train per-genre scratch structure diffusion models and aggregate their final packs.")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite")
    ap.add_argument("--max-frames", type=int, default=256)
    args = ap.parse_args()

    suite_dir = Path(args.out_root) / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_structure_diffusion.py"

    target_runs: List[Dict[str, Any]] = []
    for target in ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]:
        result = _run_target(script, suite_dir, target, max_frames=int(args.max_frames))
        target_runs.append(result)

    combined_dir = suite_dir / "combined_pack"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        f.write("target,source_song,source_target_dir,hybrid_wav,accompaniment_wav\n")

    combined_rows: List[Dict[str, Any]] = []
    mean_overall = []
    mean_target = []
    mean_margin = []
    mean_warble = []
    mean_fullness = []
    mean_structure = []
    for result in target_runs:
        target = str(result["target"])
        run_dir = Path(str(result["run_dir"]))
        final_pack = Path(str(result["summary"]["final_pack_dir"]))
        rows = result["summary"]["final_summary"]["rows"]
        for row in rows:
            song = str(row["song"])
            src_dir = final_pack / "renders" / song / target
            dst_dir = combined_dir / song / target
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ["hybrid_longform_coherent.wav", "accompaniment_generated.wav", "longform_coherent.wav", "source.wav", "backing_fixed.wav"]:
                src = src_dir / name
                if src.exists():
                    shutil.copyfile(src, dst_dir / name)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                f.write(f"{target},{song},{src_dir},{dst_dir / 'hybrid_longform_coherent.wav'},{dst_dir / 'accompaniment_generated.wav'}\n")
            combined_rows.append(row)
            mean_overall.append(float(row["overall"]))
            mean_target.append(float(row["target_conf"]))
            mean_margin.append(float(row["target_margin"]))
            mean_warble.append(float(row["warble"]))
            mean_fullness.append(float(row["fullness"]))
            mean_structure.append(float(row["structure"]))

    summary = {
        "suite_dir": str(suite_dir),
        "combined_pack": str(combined_dir),
        "target_runs": target_runs,
        "mean_overall": float(sum(mean_overall) / max(1, len(mean_overall))),
        "mean_target_conf": float(sum(mean_target) / max(1, len(mean_target))),
        "mean_target_margin": float(sum(mean_margin) / max(1, len(mean_margin))),
        "mean_warble": float(sum(mean_warble) / max(1, len(mean_warble))),
        "mean_fullness": float(sum(mean_fullness) / max(1, len(mean_fullness))),
        "mean_structure": float(sum(mean_structure) / max(1, len(mean_structure))),
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (suite_dir / "winner_map.json").write_text(
        json.dumps({str(r["target"]): str(Path(str(r["summary"]["best_checkpoint"]))) for r in target_runs}, indent=2),
        encoding="utf-8",
    )
    report_lines = [
        "# Per-Genre Scratch Structure Suite",
        "",
        "- Each target genre was trained as its own structure-conditioned accompaniment model from random initialization.",
        "- Combined pack aggregates the target-specific winners into one production-facing output folder.",
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
