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

TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baroque_classical": {
        "epochs": 1,
        "max_batches": 140,
        "lr_model": 3.0e-4,
        "lr_decoder": 1.0e-5,
        "source_low": 0.55,
        "source_mid": 0.22,
        "source_high": 0.03,
        "proposal": 0.95,
        "skip_mix": 0.55,
        "w_lat": 0.70,
        "w_dt": 0.10,
        "w_mel": 0.08,
        "w_low": 0.08,
        "w_timbre": 0.40,
        "w_donor_band": 0.18,
    },
    "hiphop_xtc": {
        "epochs": 1,
        "max_batches": 160,
        "lr_model": 3.2e-4,
        "lr_decoder": 1.2e-5,
        "source_low": 0.42,
        "source_mid": 0.16,
        "source_high": 0.02,
        "proposal": 1.05,
        "skip_mix": 0.42,
        "w_lat": 0.58,
        "w_dt": 0.08,
        "w_mel": 0.06,
        "w_low": 0.06,
        "w_timbre": 0.48,
        "w_donor_band": 0.28,
    },
    "lofi_hh_lfbb": {
        "epochs": 1,
        "max_batches": 150,
        "lr_model": 3.0e-4,
        "lr_decoder": 1.0e-5,
        "source_low": 0.48,
        "source_mid": 0.16,
        "source_high": 0.03,
        "proposal": 1.00,
        "skip_mix": 0.45,
        "w_lat": 0.60,
        "w_dt": 0.08,
        "w_mel": 0.06,
        "w_low": 0.06,
        "w_timbre": 0.46,
        "w_donor_band": 0.24,
    },
    "cc0_other": {
        "epochs": 1,
        "max_batches": 120,
        "lr_model": 2.8e-4,
        "lr_decoder": 8.0e-6,
        "source_low": 0.62,
        "source_mid": 0.28,
        "source_high": 0.05,
        "proposal": 0.90,
        "skip_mix": 0.62,
        "w_lat": 0.78,
        "w_dt": 0.12,
        "w_mel": 0.10,
        "w_low": 0.10,
        "w_timbre": 0.32,
        "w_donor_band": 0.14,
    },
}


def _latest_run(root: Path) -> Path:
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs in {root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def _run_target(script: Path, suite_root: Path, target: str) -> Dict[str, Any]:
    cfg = TARGET_CONFIGS[target]
    target_root = suite_root / target
    target_root.mkdir(parents=True, exist_ok=True)
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
        "--single-genre-target",
        str(target),
        "--lr-model",
        str(float(cfg["lr_model"])),
        "--lr-decoder",
        str(float(cfg["lr_decoder"])),
        "--source-hint-low-keep",
        str(float(cfg["source_low"])),
        "--source-hint-mid-keep",
        str(float(cfg["source_mid"])),
        "--source-hint-high-keep",
        str(float(cfg["source_high"])),
        "--proposal-scale",
        str(float(cfg["proposal"])),
        "--source-skip-mix",
        str(float(cfg["skip_mix"])),
        "--loss-lat-weight",
        str(float(cfg["w_lat"])),
        "--loss-dt-weight",
        str(float(cfg["w_dt"])),
        "--loss-mel-weight",
        str(float(cfg["w_mel"])),
        "--loss-low-weight",
        str(float(cfg["w_low"])),
        "--loss-timbre-weight",
        str(float(cfg["w_timbre"])),
        "--loss-donor-band-weight",
        str(float(cfg["w_donor_band"])),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{target} run failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    run_dir = _latest_run(target_root)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return {"target": target, "run_dir": str(run_dir), "summary": summary, "stdout": proc.stdout, "config": cfg}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an aggressive per-target pretrained Encodec suite.")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_pretrained_encodec_aggressive_suite")
    args = ap.parse_args()

    suite_dir = Path(args.out_root) / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "lab 3.4" / "scripts" / "train_pretrained_encodec_fusion.py"

    target_runs: List[Dict[str, Any]] = []
    for target in ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]:
        target_runs.append(_run_target(script, suite_dir, target))

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
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (suite_dir / "winner_map.json").write_text(
        json.dumps({str(r["target"]): str(Path(str(r["summary"]["best_checkpoint"]))) for r in target_runs}, indent=2),
        encoding="utf-8",
    )
    (suite_dir / "diagnosis_report.md").write_text(
        "\n".join(
            [
                "# Aggressive Per-Genre Pretrained Encodec Suite",
                "",
                "- Hypothesis: the first pretrained path was still too source-anchored.",
                "- This suite weakens source-hint carryover and raises donor/style pressure per target.",
                f"- Mean overall: {summary['mean_overall']:.4f}",
                f"- Mean target confidence: {summary['mean_target_conf']:.4f}",
                f"- Mean target margin: {summary['mean_target_margin']:.4f}",
                f"- Mean fullness: {summary['mean_fullness']:.4f}",
                f"- Mean warble: {summary['mean_warble']:.4f}",
                f"- Mean structure: {summary['mean_structure']:.4f}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
