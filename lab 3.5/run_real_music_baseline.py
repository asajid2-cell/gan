#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.real_music_baseline import render_codec_baseline_pack


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render the historical codec baseline on the real-music validation plan.")
    ap.add_argument("--plan-path", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "validation_plan.json")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "codec_baseline_pack")
    ap.add_argument("--codec-run", type=str, default="run1055")
    ap.add_argument("--codec-checkpoint", type=Path, default=None)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--overlap-seconds", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=328)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = render_codec_baseline_pack(
        plan_path=args.plan_path,
        out_dir=args.out_dir,
        codec_run=str(args.codec_run),
        codec_checkpoint=args.codec_checkpoint,
        max_cases=int(args.max_cases),
        seconds=float(args.seconds),
        overlap_seconds=float(args.overlap_seconds),
        device=str(args.device),
        seed=int(args.seed),
    )
    print(json.dumps({"out_dir": str(args.out_dir), "n_cases": len(result.get("rows", []))}, indent=2))


if __name__ == "__main__":
    main()
