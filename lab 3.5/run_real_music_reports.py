#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.real_music_reports import (  # noqa: E402
    baseline_compare_report,
    completion_gate_report,
    content_structure_report,
    genre_separation_report,
    lab1_bottleneck_audit_report,
    longform_coherence_report,
    manual_review_packet,
    mert_realism_report,
    musical_element_shift_report,
    novelty_and_listening_audit,
    realism_distribution_report,
    write_final_pack,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build real-music delivery, audit, comparison, and completion-gate reports.")
    ap.add_argument("--action", choices=["final-pack", "separation", "audit", "manual-packet", "baseline", "realism", "mert-realism", "content", "elements", "bottleneck", "longform", "gate", "all"], default="all")
    ap.add_argument("--validation-pack-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "final_pack")
    ap.add_argument("--validation-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "final_pack" / "validation_report.json")
    ap.add_argument("--final-pack-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "delivery_pack")
    ap.add_argument("--separation-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "genre_separation_report.json")
    ap.add_argument("--listening-audit", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "listening_audit.json")
    ap.add_argument("--manual-notes-csv", type=Path, default=None)
    ap.add_argument("--baseline-validation-report", type=Path, default=None)
    ap.add_argument("--baseline-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "baseline_compare_report.json")
    ap.add_argument("--baseline-pack-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "codec_baseline_pack")
    ap.add_argument("--manual-review-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "manual_review")
    ap.add_argument("--manual-review-title", type=str, default="Real-Music Transfer Manual Review")
    ap.add_argument("--realism-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "realism_distribution_report.json")
    ap.add_argument("--realism-reference-profiles", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "reference_profiles.json")
    ap.add_argument("--mert-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "mert_realism_report.json")
    ap.add_argument("--mert-model-name", type=str, default="m-a-p/MERT-v1-95M")
    ap.add_argument("--mert-seconds", type=float, default=12.0)
    ap.add_argument("--mert-refs-per-target", type=int, default=6)
    ap.add_argument("--mert-max-cases", type=int, default=0)
    ap.add_argument("--discovery-report", type=Path, default=REPO_ROOT / "data" / "real_music_manifests" / "spotify_discovered_genres_report.json")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "spotify_discovered_genres_cache")
    ap.add_argument("--train-summary", type=Path, default=Path(""))
    ap.add_argument("--validation-plan", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "validation_plan.json")
    ap.add_argument("--gate-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "completion_gate_report.json")
    ap.add_argument("--longform-generated-wav", type=Path, default=Path(""))
    ap.add_argument("--longform-source-audio", type=Path, default=Path(""))
    ap.add_argument("--longform-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "longform_coherence_report.json")
    ap.add_argument("--longform-source-genre", type=str, default="")
    ap.add_argument("--longform-target-genre", type=str, default="")
    ap.add_argument("--longform-reference-profiles", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "reference_profiles.json")
    ap.add_argument("--longform-seconds", type=float, default=0.0)
    ap.add_argument("--longform-expected-seconds", type=float, default=0.0)
    ap.add_argument("--longform-chunk-seconds", type=float, default=3.0)
    ap.add_argument("--longform-overlap-seconds", type=float, default=0.5)
    ap.add_argument("--longform-boundary-window-seconds", type=float, default=0.25)
    ap.add_argument("--element-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "musical_element_report.json")
    ap.add_argument("--element-reference-profiles", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "reference_profiles.json")
    ap.add_argument("--content-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "content_structure_report.json")
    ap.add_argument("--content-seconds", type=float, default=0.0)
    ap.add_argument("--bottleneck-report", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "lab1_real_music_bottleneck_report.json")
    ap.add_argument("--bottleneck-sample-size", type=int, default=40000)
    ap.add_argument("--bottleneck-retrieval-sample-size", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=328)
    return ap.parse_args()


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main() -> None:
    args = parse_args()
    result = None
    if args.action in {"final-pack", "all"}:
        result = write_final_pack(
            validation_pack_dir=args.validation_pack_dir,
            out_dir=args.final_pack_dir,
            validation_report=args.validation_report if args.validation_report.exists() else None,
        )
        if args.action == "final-pack":
            _print(result)
            return
    if args.action in {"separation", "all"}:
        result = genre_separation_report(validation_report=args.validation_report, out_path=args.separation_report)
        if args.action == "separation":
            _print(result)
            return
    if args.action in {"audit", "all"}:
        manual = args.manual_notes_csv if args.manual_notes_csv and args.manual_notes_csv.is_file() else None
        result = novelty_and_listening_audit(
            validation_pack_dir=args.validation_pack_dir,
            validation_report=args.validation_report,
            out_path=args.listening_audit,
            manual_notes_csv=manual,
        )
        if args.action == "audit":
            _print(result)
            return
    if args.action == "manual-packet":
        result = manual_review_packet(
            validation_pack_dir=args.validation_pack_dir,
            validation_report=args.validation_report,
            baseline_pack_dir=args.baseline_pack_dir,
            baseline_validation_report=args.baseline_validation_report if args.baseline_validation_report and args.baseline_validation_report.is_file() else args.baseline_pack_dir / "validation_report.json",
            out_dir=args.manual_review_dir,
            title=str(args.manual_review_title),
        )
        _print(result)
        return
    if args.action in {"baseline", "all"}:
        baseline_path = args.baseline_validation_report if args.baseline_validation_report and args.baseline_validation_report.is_file() else None
        result = baseline_compare_report(
            new_validation_report=args.validation_report,
            baseline_validation_report=baseline_path,
            out_path=args.baseline_report,
        )
        if args.action == "baseline":
            _print(result)
            return
    if args.action == "realism":
        result = realism_distribution_report(
            validation_pack_dir=args.validation_pack_dir,
            out_path=args.realism_report,
            reference_profiles=args.realism_reference_profiles,
        )
        _print(result)
        return
    if args.action == "mert-realism":
        result = mert_realism_report(
            validation_pack_dir=args.validation_pack_dir,
            out_path=args.mert_report,
            reference_profiles=args.realism_reference_profiles,
            model_name=str(args.mert_model_name),
            seconds=float(args.mert_seconds),
            refs_per_target=int(args.mert_refs_per_target),
            max_cases=int(args.mert_max_cases),
            device_arg="auto",
        )
        _print(result)
        return
    if args.action == "content":
        result = content_structure_report(
            validation_pack_dir=args.validation_pack_dir,
            out_path=args.content_report,
            seconds=float(args.content_seconds),
        )
        _print(result)
        return
    if args.action == "elements":
        result = musical_element_shift_report(
            validation_pack_dir=args.validation_pack_dir,
            out_path=args.element_report,
            reference_profiles=args.element_reference_profiles if args.element_reference_profiles.is_file() else None,
        )
        _print(result)
        return
    if args.action == "bottleneck":
        result = lab1_bottleneck_audit_report(
            cache_dir=args.cache_dir,
            out_path=args.bottleneck_report,
            sample_size=int(args.bottleneck_sample_size),
            retrieval_sample_size=int(args.bottleneck_retrieval_sample_size),
            seed=int(args.seed),
        )
        _print(result)
        return
    if args.action == "longform":
        if not str(args.longform_generated_wav) or not args.longform_generated_wav.is_file():
            raise ValueError("--longform-generated-wav is required for --action longform")
        if not str(args.longform_source_audio) or not args.longform_source_audio.is_file():
            raise ValueError("--longform-source-audio is required for --action longform")
        result = longform_coherence_report(
            generated_wav=args.longform_generated_wav,
            source_audio=args.longform_source_audio,
            out_path=args.longform_report,
            source_genre=str(args.longform_source_genre),
            target_genre=str(args.longform_target_genre),
            reference_profiles=args.longform_reference_profiles if args.longform_reference_profiles.is_file() else None,
            seconds=float(args.longform_seconds),
            expected_seconds=float(args.longform_expected_seconds),
            chunk_seconds=float(args.longform_chunk_seconds),
            overlap_seconds=float(args.longform_overlap_seconds),
            boundary_window_seconds=float(args.longform_boundary_window_seconds),
        )
        _print(result)
        return
    if args.action in {"gate", "all"}:
        if not str(args.train_summary):
            raise ValueError("--train-summary is required for gate/all")
        result = completion_gate_report(
            discovery_report=args.discovery_report,
            cache_dir=args.cache_dir,
            train_summary=args.train_summary,
            validation_plan=args.validation_plan,
            validation_pack_dir=args.validation_pack_dir,
            validation_report=args.validation_report,
            separation_report=args.separation_report,
            final_pack_dir=args.final_pack_dir,
            listening_audit=args.listening_audit,
            baseline_report=args.baseline_report,
            out_path=args.gate_report,
        )
    _print(result)


if __name__ == "__main__":
    main()
