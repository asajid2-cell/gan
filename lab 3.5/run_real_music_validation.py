#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.real_music_validation import (  # noqa: E402
    build_reference_profiles,
    create_validation_plan,
    evaluate_validation_pack,
    render_validation_pack,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create, render, and evaluate fixed validation packs for the real-music reset.")
    ap.add_argument("--action", choices=["plan", "profiles", "render", "evaluate", "all"], default="all")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--plan-path", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "validation_plan.json")
    ap.add_argument("--pack-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "final_pack")
    ap.add_argument("--report-path", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "final_pack" / "validation_report.json")
    ap.add_argument("--profiles-path", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "reference_profiles.json")
    ap.add_argument("--sources-per-genre", type=int, default=2)
    ap.add_argument("--targets-per-source", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--profile-max-per-genre", type=int, default=96)
    ap.add_argument("--render-chunk-seconds", type=float, default=3.0)
    ap.add_argument("--render-overlap-seconds", type=float, default=0.5)
    ap.add_argument("--render-style-strength", type=float, default=1.0)
    ap.add_argument("--render-envelope-strength", type=float, default=0.35)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=328)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = None
    if args.action in {"plan", "all"}:
        result = create_validation_plan(
            args.cache_dir,
            args.plan_path,
            sources_per_genre=int(args.sources_per_genre),
            targets_per_source=int(args.targets_per_source),
            seconds=float(args.seconds),
            seed=int(args.seed),
        )
        if args.action == "plan":
            print(json.dumps(result, indent=2))
            return
    if args.action in {"profiles", "all"}:
        result = build_reference_profiles(args.cache_dir, max_per_genre=int(args.profile_max_per_genre), seconds=float(args.seconds), seed=int(args.seed))
        args.profiles_path.parent.mkdir(parents=True, exist_ok=True)
        args.profiles_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        if args.action == "profiles":
            print(json.dumps({"profiles_path": str(args.profiles_path), "genres": sorted(result.get("profiles", {}).keys())}, indent=2))
            return
    if args.action in {"render", "all"}:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for render/all")
        result = render_validation_pack(
            checkpoint=args.checkpoint,
            cache_dir=args.cache_dir,
            plan_path=args.plan_path,
            out_dir=args.pack_dir,
            device=str(args.device),
            chunk_seconds=float(args.render_chunk_seconds),
            overlap_seconds=float(args.render_overlap_seconds),
            style_strength=float(args.render_style_strength),
            envelope_strength=float(args.render_envelope_strength),
        )
        if args.action == "render":
            print(json.dumps({"pack_dir": str(args.pack_dir), "n_cases": len(result.get("rows", []))}, indent=2))
            return
    if args.action in {"evaluate", "all"}:
        result = evaluate_validation_pack(
            cache_dir=args.cache_dir,
            plan_path=args.plan_path,
            pack_dir=args.pack_dir,
            out_path=args.report_path,
            reference_profiles_path=args.profiles_path,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
