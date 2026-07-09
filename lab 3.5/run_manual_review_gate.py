#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "saves2" / "real_music_transfer"
DEFAULT_REVIEW_DIR = DEFAULT_BASE / "manual_review_envelope075_strength3"
DEFAULT_VALIDATION_PACK = DEFAULT_BASE / "final_pack_envelope075_strength3"
DEFAULT_VALIDATION_REPORT = DEFAULT_VALIDATION_PACK / "validation_report.json"
DEFAULT_FINAL_PACK = DEFAULT_BASE / "delivery_pack_envelope075_strength3"
DEFAULT_SEPARATION_REPORT = DEFAULT_BASE / "genre_separation_report_envelope075_strength3.json"
DEFAULT_BASELINE_REPORT = DEFAULT_BASE / "baseline_compare_report_envelope075_strength3.json"
DEFAULT_LISTENING_AUDIT = DEFAULT_BASE / "listening_audit_envelope075_manual.json"
DEFAULT_GATE_REPORT = DEFAULT_BASE / "completion_gate_report_envelope075_manual.json"
DEFAULT_DISCOVERY_REPORT = REPO_ROOT / "data" / "real_music_manifests" / "spotify_discovered_genres_report.json"
DEFAULT_CACHE_DIR = DEFAULT_BASE / "spotify_discovered_genres_cache"
DEFAULT_TRAIN_SUMMARY = DEFAULT_BASE / "runs" / "real_transfer_20260512_182741" / "summary.json"
DEFAULT_VALIDATION_PLAN = DEFAULT_BASE / "validation_plan.json"

PASS_FIELDS = [
    "realism_pass",
    "source_identity_pass",
    "target_recognizable_pass",
    "artifact_free_pass",
    "novelty_pass",
]
REQUIRED_FIELDS = ["review_complete", *PASS_FIELDS, "baseline_preference"]
COMPLETE_VALUES = {"1", "true", "yes", "y"}
PASS_VALUES = {"0", "1"}
BASELINE_OPTIONS = {"new", "baseline", "tie", "unclear"}


def _read_manual_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _missing_review_fields(row: Dict[str, str]) -> List[str]:
    missing: List[str] = []
    if str(row.get("review_complete", "")).strip().lower() not in COMPLETE_VALUES:
        missing.append("review_complete")
    for field in PASS_FIELDS:
        if str(row.get(field, "")).strip() not in PASS_VALUES:
            missing.append(field)
    if str(row.get("baseline_preference", "")).strip() not in BASELINE_OPTIONS:
        missing.append("baseline_preference")
    return missing


def manual_status(manual_csv: Path) -> Dict[str, Any]:
    if not manual_csv.exists():
        return {
            "manual_notes_csv": str(manual_csv),
            "exists": False,
            "n_cases": 0,
            "reviewed": 0,
            "remaining": 0,
            "invalid_cases": 0,
            "gate_ready": False,
            "first_invalid_cases": [{"case_id": "", "missing": ["manual_notes_csv"]}],
        }

    rows = _read_manual_rows(manual_csv)
    invalid: List[Dict[str, Any]] = []
    for row in rows:
        missing = _missing_review_fields(row)
        if missing:
            invalid.append({"case_id": str(row.get("case_id", "")), "missing": missing})

    reviewed = len(rows) - len(invalid)
    return {
        "manual_notes_csv": str(manual_csv),
        "exists": True,
        "n_cases": len(rows),
        "reviewed": reviewed,
        "remaining": len(invalid),
        "invalid_cases": len(invalid),
        "gate_ready": bool(rows and not invalid),
        "required_fields": REQUIRED_FIELDS,
        "first_invalid_cases": invalid[:12],
    }


def _run(cmd: List[str]) -> None:
    print("+ " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Validate completed manual listening notes, then rerun the real-music listening audit "
            "and completion gate for the envelope-0.75 production candidate."
        )
    )
    ap.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    ap.add_argument("--manual-notes-csv", type=Path, default=None)
    ap.add_argument("--validation-pack-dir", type=Path, default=DEFAULT_VALIDATION_PACK)
    ap.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    ap.add_argument("--final-pack-dir", type=Path, default=DEFAULT_FINAL_PACK)
    ap.add_argument("--separation-report", type=Path, default=DEFAULT_SEPARATION_REPORT)
    ap.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    ap.add_argument("--listening-audit", type=Path, default=DEFAULT_LISTENING_AUDIT)
    ap.add_argument("--gate-report", type=Path, default=DEFAULT_GATE_REPORT)
    ap.add_argument("--discovery-report", type=Path, default=DEFAULT_DISCOVERY_REPORT)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--train-summary", type=Path, default=DEFAULT_TRAIN_SUMMARY)
    ap.add_argument("--validation-plan", type=Path, default=DEFAULT_VALIDATION_PLAN)
    ap.add_argument("--check-only", action="store_true", help="Only validate manual CSV completeness; do not run audit/gate.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    manual_csv: Path = args.manual_notes_csv if args.manual_notes_csv else args.review_dir / "manual_notes_template.csv"
    status = manual_status(manual_csv)
    print(json.dumps(status, indent=2))

    if not status["gate_ready"]:
        print("Manual review is incomplete; audit and completion gate were not run.", file=sys.stderr)
        return 2

    if args.check_only:
        return 0

    reports = REPO_ROOT / "lab 3.5" / "run_real_music_reports.py"
    _run(
        [
            sys.executable,
            str(reports),
            "--action",
            "audit",
            "--validation-pack-dir",
            str(args.validation_pack_dir),
            "--validation-report",
            str(args.validation_report),
            "--listening-audit",
            str(args.listening_audit),
            "--manual-notes-csv",
            str(manual_csv),
        ]
    )
    _run(
        [
            sys.executable,
            str(reports),
            "--action",
            "gate",
            "--validation-pack-dir",
            str(args.validation_pack_dir),
            "--validation-report",
            str(args.validation_report),
            "--final-pack-dir",
            str(args.final_pack_dir),
            "--separation-report",
            str(args.separation_report),
            "--listening-audit",
            str(args.listening_audit),
            "--baseline-report",
            str(args.baseline_report),
            "--discovery-report",
            str(args.discovery_report),
            "--cache-dir",
            str(args.cache_dir),
            "--train-summary",
            str(args.train_summary),
            "--validation-plan",
            str(args.validation_plan),
            "--gate-report",
            str(args.gate_report),
        ]
    )

    gate = json.loads(args.gate_report.read_text(encoding="utf-8"))
    print(json.dumps({"listening_audit": str(args.listening_audit), "gate_report": str(args.gate_report), "passed": gate.get("passed")}, indent=2))
    return 0 if bool(gate.get("passed")) else 3


if __name__ == "__main__":
    raise SystemExit(main())
