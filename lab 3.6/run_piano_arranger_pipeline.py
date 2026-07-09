#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.piano_arranger_data import (
    DEFAULT_DISCOVERY_REPORT,
    DEFAULT_MIDI_MANIFEST,
    DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST,
    DEFAULT_PIANO_MANIFEST,
    DEFAULT_SOURCE_MANIFEST,
    audit_paired_audio_midi_manifest,
    build_paired_audio_midi_manifest,
    build_piano_candidate_manifest,
    build_midi_piano_target_manifest,
)
from dggr.piano_arranger_baseline import HeuristicPianoConfig, render_heuristic_baseline
from dggr.piano_arranger_batch import PairedCheckpointBatchEvalConfig, SourceManifestAuditConfig, audit_source_manifest, validate_paired_checkpoint
from dggr.piano_arranger_cache import (
    DEFAULT_PIANO_CACHE_DIR,
    MidiPianoCacheConfig,
    PairedAudioMidiCacheConfig,
    PianoCacheConfig,
    build_midi_piano_target_cache,
    build_paired_audio_midi_target_cache,
    build_piano_arranger_cache,
)
from dggr.piano_arranger_eval import PianoEvalConfig, PianoSectionReportConfig, evaluate_arrangement_file, section_report_arrangement_file
from dggr.piano_arranger_infer import (
    PianoChunkedInferenceConfig,
    PianoInferenceConfig,
    infer_piano_arrangement,
    infer_piano_arrangement_chunked,
)
from dggr.piano_arranger_train import DEFAULT_PIANO_RUN_ROOT, PianoTrainConfig, train_piano_roll_model


DEFAULT_OUTPUT_DIR = REPO_ROOT / "saves2" / "piano_arranger" / "outputs"


def _print_json(obj: Dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DGGR Lab 3.6 piano-arranger pipeline.")
    ap.add_argument(
        "--action",
        choices=[
            "discover-piano",
            "discover-midi",
            "discover-paired",
            "review-paired",
            "cache",
            "midi-cache",
            "paired-cache",
            "heuristic-baseline",
            "train",
            "infer",
            "infer-chunked",
            "audit-sources",
            "validate-paired",
            "evaluate",
            "section-report",
        ],
        default="discover-piano",
    )
    ap.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    ap.add_argument("--discovery-report", type=Path, default=DEFAULT_DISCOVERY_REPORT)
    ap.add_argument("--piano-manifest", type=Path, default=DEFAULT_PIANO_MANIFEST)
    ap.add_argument("--midi-manifest", type=Path, default=DEFAULT_MIDI_MANIFEST)
    ap.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST)
    ap.add_argument("--audio-root", type=Path, action="append", default=[])
    ap.add_argument("--midi-root", type=Path, action="append", default=[])
    ap.add_argument("--include-package-midi-examples", action="store_true")
    ap.add_argument("--min-midi-notes", type=int, default=8)
    ap.add_argument("--midi-source-preview-mode", choices=["piano", "ensemble"], default="piano")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_PIANO_CACHE_DIR)
    ap.add_argument("--train-out-root", type=Path, default=DEFAULT_PIANO_RUN_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--run-name", type=str, default="")
    ap.add_argument("--report-path", type=Path, default=None)
    ap.add_argument("--batch-out-dir", type=Path, default=None)
    ap.add_argument("--batch-chunked", action="store_true")
    ap.add_argument("--arrangement-json", type=Path, default=None)
    ap.add_argument("--eval-report", type=Path, default=None)
    ap.add_argument("--eval-label", type=str, default="")
    ap.add_argument("--min-piano-score", type=float, default=8.0)
    ap.add_argument("--max-pair-duration-delta", type=float, default=5.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=0)
    ap.add_argument("--source-audio", type=Path, default=None)
    ap.add_argument("--out-stem", type=Path, default=Path(""))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--cache-seconds", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--eval-max-frames", type=int, default=0)
    ap.add_argument("--frame-hz", type=float, default=25.0)
    ap.add_argument("--section-seconds", type=float, default=8.0)
    ap.add_argument("--chunk-seconds", type=float, default=12.0)
    ap.add_argument("--chunk-hop-seconds", type=float, default=0.0)
    ap.add_argument("--section-profile", choices=["flat", "arc"], default="flat")
    ap.add_argument("--seed", type=int, default=328)
    ap.add_argument("--warm-start-checkpoint", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--density-loss-weight", type=float, default=0.35)
    ap.add_argument("--chroma-loss-weight", type=float, default=0.35)
    ap.add_argument("--pitch-usage-loss-weight", type=float, default=0.35)
    ap.add_argument("--hierarchy-loss-weight", type=float, default=0.25)
    ap.add_argument("--musical-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--event-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-f1-loss-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-alignment-loss-weight", type=float, default=0.0)
    ap.add_argument("--role-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--texture-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--melody-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--melody-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--texture-role-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--texture-role-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--section-role-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--section-role-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--arranger-state-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--bass-continuity-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--body-melody-state-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--body-melody-state-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--section-diversity-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--section-diversity-balance-loss-weight", type=float, default=0.0)
    ap.add_argument("--anti-collapse-loss-weight", type=float, default=0.0)
    ap.add_argument("--source-onset-loss-weight", type=float, default=0.0)
    ap.add_argument("--source-chroma-loss-weight", type=float, default=0.0)
    ap.add_argument("--harmonic-plan-loss-weight", type=float, default=0.0)
    ap.add_argument("--hidden-channels", type=int, default=96)
    ap.add_argument("--n-blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument(
        "--model-architecture",
        choices=[
            "conv1d",
            "key_conditioned",
            "chroma_key_conditioned",
            "harmony_conditioned",
            "musical_plan_conditioned",
        ],
        default="conv1d",
    )
    ap.add_argument("--key-embed-dim", type=int, default=32)
    ap.add_argument("--max-batches-per-epoch", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=1)
    ap.add_argument("--sample-count", type=int, default=1)
    ap.add_argument("--sample-score-pass-weight", type=float, default=1000.0)
    ap.add_argument("--sample-score-warning-penalty", type=float, default=10.0)
    ap.add_argument("--sample-score-source-active-weight", type=float, default=1.0)
    ap.add_argument("--sample-score-source-onset-weight", type=float, default=0.25)
    ap.add_argument("--sample-score-target-active-weight", type=float, default=1.0)
    ap.add_argument("--sample-score-target-onset-weight", type=float, default=0.25)
    ap.add_argument("--sample-score-role-balance-weight", type=float, default=25.0)
    ap.add_argument("--sample-score-chord-frame-target", type=float, default=0.80)
    ap.add_argument("--sample-score-melody-coverage-target", type=float, default=0.30)
    ap.add_argument("--sample-score-bass-coverage-min", type=float, default=0.35)
    ap.add_argument("--sample-score-bass-coverage-max", type=float, default=0.85)
    ap.add_argument("--sample-score-polyphony-target", type=float, default=5.5)
    ap.add_argument("--sample-score-rms-target", type=float, default=0.18)
    ap.add_argument("--sample-score-quality-penalty-weight", type=float, default=25.0)
    ap.add_argument("--sample-score-min-notes-per-second", type=float, default=6.0)
    ap.add_argument("--sample-score-min-section-notes", type=float, default=12.0)
    ap.add_argument("--sample-score-min-section-unique-pitches", type=float, default=8.0)
    ap.add_argument("--sample-score-min-section-chord-frame", type=float, default=0.40)
    ap.add_argument("--sample-score-min-section-fullness", type=float, default=0.70)
    ap.add_argument("--sample-score-max-single-pitch-class-fraction", type=float, default=0.32)
    ap.add_argument("--sample-score-min-mid-note-fraction", type=float, default=0.20)
    ap.add_argument("--sample-score-max-high-note-fraction", type=float, default=0.75)
    ap.add_argument("--onset-threshold", type=float, default=0.35)
    ap.add_argument("--frame-threshold", type=float, default=0.35)
    ap.add_argument("--max-notes-per-second", type=float, default=32.0)
    ap.add_argument("--max-simultaneous-notes", type=int, default=12)
    ap.add_argument("--max-onsets-per-frame", type=int, default=6)
    ap.add_argument("--max-pitch-fraction", type=float, default=0.22)
    ap.add_argument("--max-pitch-class-fraction", type=float, default=0.32)
    ap.add_argument("--min-note-duration", type=float, default=0.08)
    ap.add_argument("--max-note-duration", type=float, default=1.5)
    ap.add_argument("--bass-min-note-duration", type=float, default=0.0)
    ap.add_argument("--min-selected-notes", type=int, default=24)
    ap.add_argument("--min-unique-pitches", type=int, default=8)
    ap.add_argument("--register-coverage-chunk-seconds", type=float, default=0.0)
    ap.add_argument("--section-bass-repair", action="store_true")
    ap.add_argument("--section-bass-repair-min-coverage", type=float, default=0.05)
    ap.add_argument("--section-diversity-repair", action="store_true")
    ap.add_argument("--section-diversity-repair-min-unique-pitches", type=int, default=8)
    ap.add_argument("--section-diversity-repair-min-chord-frame", type=float, default=0.15)
    ap.add_argument("--section-diversity-repair-max-notes", type=int, default=4)
    ap.add_argument("--diversity-fallback-threshold", type=float, default=0.05)
    ap.add_argument("--source-onset-guidance-weight", type=float, default=0.0)
    ap.add_argument("--source-onset-snap-frames", type=int, default=0)
    ap.add_argument("--source-onset-peak-threshold", type=float, default=0.35)
    ap.add_argument("--density-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--density-plan-snap-frames", type=int, default=0)
    ap.add_argument("--density-plan-peak-threshold", type=float, default=0.35)
    ap.add_argument("--event-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--event-plan-snap-frames", type=int, default=0)
    ap.add_argument("--event-plan-peak-threshold", type=float, default=0.35)
    ap.add_argument("--pc-onset-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-reserve-threshold", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-reserve-max-per-frame", type=int, default=0)
    ap.add_argument("--pc-onset-plan-reserve-min-note-score", type=float, default=0.02)
    ap.add_argument("--pc-onset-plan-select-reserve-fraction", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-assign-threshold", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-assign-fraction", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-assign-window-frames", type=int, default=1)
    ap.add_argument("--pc-onset-plan-assign-min-note-score", type=float, default=0.02)
    ap.add_argument("--pc-onset-plan-assign-source-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-assign-event-weight", type=float, default=0.0)
    ap.add_argument("--pc-onset-plan-assign-distance-penalty", type=float, default=1.0)
    ap.add_argument("--source-chroma-guidance-weight", type=float, default=0.0)
    ap.add_argument("--harmonic-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--chord-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--bass-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--voicing-plan-guidance-weight", type=float, default=0.0)
    ap.add_argument("--section-diversity-guidance-weight", type=float, default=0.0)
    ap.add_argument("--section-diversity-reserve-fraction", type=float, default=0.0)
    ap.add_argument("--section-diversity-reserve-min-note-score", type=float, default=0.02)
    ap.add_argument("--section-diversity-unique-weight", type=float, default=1.0)
    ap.add_argument("--section-diversity-pc-weight", type=float, default=1.0)
    ap.add_argument("--section-diversity-range-weight", type=float, default=0.5)
    ap.add_argument("--section-diversity-onset-weight", type=float, default=0.5)
    ap.add_argument("--section-diversity-section-seconds", type=float, default=4.0)
    ap.add_argument("--source-energy-velocity-weight", type=float, default=0.0)
    ap.add_argument("--density-plan-velocity-weight", type=float, default=0.0)
    ap.add_argument("--no-target-eval", action="store_true")
    ap.add_argument("--no-section-sample-eval", action="store_true")
    ap.add_argument("--sample-section-seconds", type=float, default=4.0)
    ap.add_argument("--min-target-global-chroma-cosine", type=float, default=0.20)
    ap.add_argument("--min-target-active-chroma-cosine", type=float, default=0.20)
    ap.add_argument("--min-target-onset-correlation", type=float, default=0.02)
    ap.add_argument("--min-target-onset-frame-f1", type=float, default=0.0)
    ap.add_argument("--min-target-pitch-class-onset-f1", type=float, default=0.0)
    ap.add_argument("--min-target-note-count-ratio", type=float, default=0.0)
    ap.add_argument("--max-target-note-count-ratio", type=float, default=0.0)
    ap.add_argument("--no-register-coverage", action="store_true")
    ap.add_argument("--no-sample-eval", action="store_true")
    ap.add_argument("--no-source-sample-eval", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--fullness", type=float, default=0.85)
    ap.add_argument("--melody-focus", type=float, default=0.80)
    ap.add_argument("--rhythmic-drive", type=float, default=0.65)
    ap.add_argument("--harmonic-adventure", type=float, default=0.25)
    ap.add_argument("--register-width", type=float, default=0.85)
    ap.add_argument("--pedal-amount", type=float, default=0.70)
    ap.add_argument("--no-wav", action="store_true")
    return ap.parse_args()


def _not_ready(action: str) -> None:
    raise NotImplementedError(
        f"Action '{action}' is planned in docs/PIANO_ARRANGER_INSTRUCT.md but is not implemented yet. "
        "Run --action discover-piano first, then build cache/render/train modules in the documented order."
    )


def main() -> None:
    args = parse_args()
    if args.action == "discover-piano":
        summary = build_piano_candidate_manifest(
            source_manifest=args.source_manifest,
            discovery_report=args.discovery_report,
            out_csv=args.piano_manifest,
            report_path=args.report_path,
            min_score=float(args.min_piano_score),
            max_rows=int(args.max_rows),
        )
        _print_json(summary.__dict__)
        return
    if args.action == "discover-midi":
        summary = build_midi_piano_target_manifest(
            roots=args.midi_root,
            out_csv=args.midi_manifest,
            report_path=args.report_path,
            max_rows=int(args.max_rows),
            min_notes=int(args.min_midi_notes),
            include_package_examples=bool(args.include_package_midi_examples),
        )
        _print_json(summary.__dict__)
        return
    if args.action == "discover-paired":
        summary = build_paired_audio_midi_manifest(
            audio_roots=args.audio_root,
            midi_roots=args.midi_root,
            out_csv=args.paired_manifest,
            report_path=args.report_path,
            max_rows=int(args.max_rows),
            min_notes=int(args.min_midi_notes),
            include_package_examples=bool(args.include_package_midi_examples),
        )
        _print_json(summary.__dict__)
        return
    if args.action == "review-paired":
        summary = audit_paired_audio_midi_manifest(
            manifest=args.paired_manifest,
            report_path=args.report_path,
            min_notes=int(args.min_midi_notes),
            max_duration_delta=float(args.max_pair_duration_delta),
        )
        _print_json(summary.__dict__)
        return
    if args.action == "cache":
        cfg = PianoCacheConfig(
            manifest=args.piano_manifest,
            cache_dir=args.cache_dir,
            seconds=float(args.cache_seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            max_tracks=int(args.max_tracks),
            seed=int(args.seed),
            fullness=float(args.fullness),
            melody_focus=float(args.melody_focus),
            rhythmic_drive=float(args.rhythmic_drive),
            harmonic_adventure=float(args.harmonic_adventure),
            register_width=float(args.register_width),
            pedal_amount=float(args.pedal_amount),
        )
        _print_json(build_piano_arranger_cache(cfg))
        return
    if args.action == "midi-cache":
        cfg = MidiPianoCacheConfig(
            manifest=args.midi_manifest,
            cache_dir=args.cache_dir,
            seconds=float(args.cache_seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            max_tracks=int(args.max_tracks),
            min_notes=int(args.min_midi_notes),
            source_preview_mode=str(args.midi_source_preview_mode),
        )
        _print_json(build_midi_piano_target_cache(cfg))
        return
    if args.action == "paired-cache":
        cfg = PairedAudioMidiCacheConfig(
            manifest=args.paired_manifest,
            cache_dir=args.cache_dir,
            seconds=float(args.cache_seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            max_tracks=int(args.max_tracks),
            min_notes=int(args.min_midi_notes),
        )
        _print_json(build_paired_audio_midi_target_cache(cfg))
        return
    if args.action == "heuristic-baseline":
        if args.source_audio is None:
            raise ValueError("--source-audio is required for --action heuristic-baseline")
        out_stem = args.out_stem
        if str(out_stem) == "":
            out_stem = DEFAULT_OUTPUT_DIR / "heuristic_baseline" / f"{Path(args.source_audio).stem}__piano_heuristic"
        cfg = HeuristicPianoConfig(
            seconds=float(args.seconds),
            fullness=float(args.fullness),
            melody_focus=float(args.melody_focus),
            rhythmic_drive=float(args.rhythmic_drive),
            harmonic_adventure=float(args.harmonic_adventure),
            register_width=float(args.register_width),
            pedal_amount=float(args.pedal_amount),
            render_wav=not bool(args.no_wav),
        )
        _print_json(render_heuristic_baseline(source_audio=args.source_audio, out_stem=out_stem, config=cfg))
        return
    if args.action == "train":
        cfg = PianoTrainConfig(
            cache_dir=args.cache_dir,
            out_root=args.train_out_root,
            run_name=str(args.run_name),
            warm_start_checkpoint=args.warm_start_checkpoint,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            density_loss_weight=float(args.density_loss_weight),
            chroma_loss_weight=float(args.chroma_loss_weight),
            pitch_usage_loss_weight=float(args.pitch_usage_loss_weight),
            hierarchy_loss_weight=float(args.hierarchy_loss_weight),
            musical_plan_loss_weight=float(args.musical_plan_loss_weight),
            event_plan_loss_weight=float(args.event_plan_loss_weight),
            pc_onset_plan_loss_weight=float(args.pc_onset_plan_loss_weight),
            pc_onset_f1_loss_weight=float(args.pc_onset_f1_loss_weight),
            pc_onset_alignment_loss_weight=float(args.pc_onset_alignment_loss_weight),
            role_plan_loss_weight=float(args.role_plan_loss_weight),
            texture_balance_loss_weight=float(args.texture_balance_loss_weight),
            melody_plan_loss_weight=float(args.melody_plan_loss_weight),
            melody_balance_loss_weight=float(args.melody_balance_loss_weight),
            texture_role_plan_loss_weight=float(args.texture_role_plan_loss_weight),
            texture_role_balance_loss_weight=float(args.texture_role_balance_loss_weight),
            section_role_plan_loss_weight=float(args.section_role_plan_loss_weight),
            section_role_balance_loss_weight=float(args.section_role_balance_loss_weight),
            arranger_state_plan_loss_weight=float(args.arranger_state_plan_loss_weight),
            bass_continuity_plan_loss_weight=float(args.bass_continuity_plan_loss_weight),
            body_melody_state_plan_loss_weight=float(args.body_melody_state_plan_loss_weight),
            body_melody_state_balance_loss_weight=float(args.body_melody_state_balance_loss_weight),
            section_diversity_plan_loss_weight=float(args.section_diversity_plan_loss_weight),
            section_diversity_balance_loss_weight=float(args.section_diversity_balance_loss_weight),
            anti_collapse_loss_weight=float(args.anti_collapse_loss_weight),
            source_onset_loss_weight=float(args.source_onset_loss_weight),
            source_chroma_loss_weight=float(args.source_chroma_loss_weight),
            harmonic_plan_loss_weight=float(args.harmonic_plan_loss_weight),
            hidden_channels=int(args.hidden_channels),
            n_blocks=int(args.n_blocks),
            dropout=float(args.dropout),
            model_architecture=str(args.model_architecture),
            key_embed_dim=int(args.key_embed_dim),
            max_batches_per_epoch=int(args.max_batches_per_epoch),
            sample_every=int(args.sample_every),
            sample_count=int(args.sample_count),
            sample_section_eval=not bool(args.no_section_sample_eval),
            sample_section_seconds=float(args.sample_section_seconds),
            sample_score_pass_weight=float(args.sample_score_pass_weight),
            sample_score_warning_penalty=float(args.sample_score_warning_penalty),
            sample_score_source_active_weight=float(args.sample_score_source_active_weight),
            sample_score_source_onset_weight=float(args.sample_score_source_onset_weight),
            sample_score_target_active_weight=float(args.sample_score_target_active_weight),
            sample_score_target_onset_weight=float(args.sample_score_target_onset_weight),
            sample_score_role_balance_weight=float(args.sample_score_role_balance_weight),
            sample_score_chord_frame_target=float(args.sample_score_chord_frame_target),
            sample_score_melody_coverage_target=float(args.sample_score_melody_coverage_target),
            sample_score_bass_coverage_min=float(args.sample_score_bass_coverage_min),
            sample_score_bass_coverage_max=float(args.sample_score_bass_coverage_max),
            sample_score_polyphony_target=float(args.sample_score_polyphony_target),
            sample_score_rms_target=float(args.sample_score_rms_target),
            sample_score_quality_penalty_weight=float(args.sample_score_quality_penalty_weight),
            sample_score_min_notes_per_second=float(args.sample_score_min_notes_per_second),
            sample_score_min_section_notes=float(args.sample_score_min_section_notes),
            sample_score_min_section_unique_pitches=float(args.sample_score_min_section_unique_pitches),
            sample_score_min_section_chord_frame=float(args.sample_score_min_section_chord_frame),
            sample_score_min_section_fullness=float(args.sample_score_min_section_fullness),
            sample_score_max_single_pitch_class_fraction=float(args.sample_score_max_single_pitch_class_fraction),
            sample_score_min_mid_note_fraction=float(args.sample_score_min_mid_note_fraction),
            sample_score_max_high_note_fraction=float(args.sample_score_max_high_note_fraction),
            onset_threshold=float(args.onset_threshold),
            frame_threshold=float(args.frame_threshold),
            max_notes_per_second=float(args.max_notes_per_second),
            max_simultaneous_notes=int(args.max_simultaneous_notes),
            max_onsets_per_frame=int(args.max_onsets_per_frame),
            max_pitch_fraction=float(args.max_pitch_fraction),
            max_pitch_class_fraction=float(args.max_pitch_class_fraction),
            min_note_duration=float(args.min_note_duration),
            max_note_duration=float(args.max_note_duration),
            bass_min_note_duration=float(args.bass_min_note_duration),
            min_selected_notes=int(args.min_selected_notes),
            min_unique_pitches=int(args.min_unique_pitches),
            require_register_coverage=not bool(args.no_register_coverage),
            register_coverage_chunk_seconds=float(args.register_coverage_chunk_seconds),
            section_bass_repair=bool(args.section_bass_repair),
            section_bass_repair_min_coverage=float(args.section_bass_repair_min_coverage),
            diversity_fallback_threshold=float(args.diversity_fallback_threshold),
            source_onset_guidance_weight=float(args.source_onset_guidance_weight),
            source_onset_snap_frames=int(args.source_onset_snap_frames),
            source_onset_peak_threshold=float(args.source_onset_peak_threshold),
            density_plan_guidance_weight=float(args.density_plan_guidance_weight),
            density_plan_snap_frames=int(args.density_plan_snap_frames),
            density_plan_peak_threshold=float(args.density_plan_peak_threshold),
            event_plan_guidance_weight=float(args.event_plan_guidance_weight),
            event_plan_snap_frames=int(args.event_plan_snap_frames),
            event_plan_peak_threshold=float(args.event_plan_peak_threshold),
            pc_onset_plan_guidance_weight=float(args.pc_onset_plan_guidance_weight),
            pc_onset_plan_reserve_threshold=float(args.pc_onset_plan_reserve_threshold),
            pc_onset_plan_reserve_max_per_frame=int(args.pc_onset_plan_reserve_max_per_frame),
            pc_onset_plan_reserve_min_note_score=float(args.pc_onset_plan_reserve_min_note_score),
            pc_onset_plan_select_reserve_fraction=float(args.pc_onset_plan_select_reserve_fraction),
            pc_onset_plan_assign_threshold=float(args.pc_onset_plan_assign_threshold),
            pc_onset_plan_assign_fraction=float(args.pc_onset_plan_assign_fraction),
            pc_onset_plan_assign_window_frames=int(args.pc_onset_plan_assign_window_frames),
            pc_onset_plan_assign_min_note_score=float(args.pc_onset_plan_assign_min_note_score),
            pc_onset_plan_assign_source_weight=float(args.pc_onset_plan_assign_source_weight),
            pc_onset_plan_assign_event_weight=float(args.pc_onset_plan_assign_event_weight),
            pc_onset_plan_assign_distance_penalty=float(args.pc_onset_plan_assign_distance_penalty),
            source_chroma_guidance_weight=float(args.source_chroma_guidance_weight),
            harmonic_plan_guidance_weight=float(args.harmonic_plan_guidance_weight),
            chord_plan_guidance_weight=float(args.chord_plan_guidance_weight),
            bass_plan_guidance_weight=float(args.bass_plan_guidance_weight),
            voicing_plan_guidance_weight=float(args.voicing_plan_guidance_weight),
            section_diversity_guidance_weight=float(args.section_diversity_guidance_weight),
            section_diversity_reserve_fraction=float(args.section_diversity_reserve_fraction),
            section_diversity_reserve_min_note_score=float(args.section_diversity_reserve_min_note_score),
            section_diversity_unique_weight=float(args.section_diversity_unique_weight),
            section_diversity_pc_weight=float(args.section_diversity_pc_weight),
            section_diversity_range_weight=float(args.section_diversity_range_weight),
            section_diversity_onset_weight=float(args.section_diversity_onset_weight),
            section_diversity_section_seconds=float(args.section_diversity_section_seconds),
            source_energy_velocity_weight=float(args.source_energy_velocity_weight),
            density_plan_velocity_weight=float(args.density_plan_velocity_weight),
            sample_eval=not bool(args.no_sample_eval),
            sample_source_eval=not bool(args.no_source_sample_eval),
            sample_target_eval=not bool(args.no_target_eval),
            min_target_global_chroma_cosine=float(args.min_target_global_chroma_cosine),
            min_target_active_chroma_cosine=float(args.min_target_active_chroma_cosine),
            min_target_onset_correlation=float(args.min_target_onset_correlation),
            min_target_onset_frame_f1=float(args.min_target_onset_frame_f1),
            min_target_pitch_class_onset_f1=float(args.min_target_pitch_class_onset_f1),
            min_target_note_count_ratio=float(args.min_target_note_count_ratio),
            max_target_note_count_ratio=float(args.max_target_note_count_ratio),
            device=str(args.device),
            seed=int(args.seed),
        )
        _print_json(train_piano_roll_model(cfg))
        return
    if args.action == "infer":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --action infer")
        if args.source_audio is None:
            raise ValueError("--source-audio is required for --action infer")
        cfg = PianoInferenceConfig(
            checkpoint=args.checkpoint,
            source_audio=args.source_audio,
            out_stem=args.out_stem,
            seconds=float(args.seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            onset_threshold=float(args.onset_threshold),
            frame_threshold=float(args.frame_threshold),
            max_notes_per_second=float(args.max_notes_per_second),
            max_simultaneous_notes=int(args.max_simultaneous_notes),
            max_onsets_per_frame=int(args.max_onsets_per_frame),
            max_pitch_fraction=float(args.max_pitch_fraction),
            max_pitch_class_fraction=float(args.max_pitch_class_fraction),
            min_note_duration=float(args.min_note_duration),
            max_note_duration=float(args.max_note_duration),
            bass_min_note_duration=float(args.bass_min_note_duration),
            min_selected_notes=int(args.min_selected_notes),
            min_unique_pitches=int(args.min_unique_pitches),
            require_register_coverage=not bool(args.no_register_coverage),
            register_coverage_chunk_seconds=float(args.register_coverage_chunk_seconds),
            section_bass_repair=bool(args.section_bass_repair),
            section_bass_repair_min_coverage=float(args.section_bass_repair_min_coverage),
            section_diversity_repair=bool(args.section_diversity_repair),
            section_diversity_repair_min_unique_pitches=int(args.section_diversity_repair_min_unique_pitches),
            section_diversity_repair_min_chord_frame=float(args.section_diversity_repair_min_chord_frame),
            section_diversity_repair_max_notes=int(args.section_diversity_repair_max_notes),
            diversity_fallback_threshold=float(args.diversity_fallback_threshold),
            source_onset_guidance_weight=float(args.source_onset_guidance_weight),
            source_onset_snap_frames=int(args.source_onset_snap_frames),
            source_onset_peak_threshold=float(args.source_onset_peak_threshold),
            density_plan_guidance_weight=float(args.density_plan_guidance_weight),
            density_plan_snap_frames=int(args.density_plan_snap_frames),
            density_plan_peak_threshold=float(args.density_plan_peak_threshold),
            event_plan_guidance_weight=float(args.event_plan_guidance_weight),
            event_plan_snap_frames=int(args.event_plan_snap_frames),
            event_plan_peak_threshold=float(args.event_plan_peak_threshold),
            pc_onset_plan_guidance_weight=float(args.pc_onset_plan_guidance_weight),
            pc_onset_plan_reserve_threshold=float(args.pc_onset_plan_reserve_threshold),
            pc_onset_plan_reserve_max_per_frame=int(args.pc_onset_plan_reserve_max_per_frame),
            pc_onset_plan_reserve_min_note_score=float(args.pc_onset_plan_reserve_min_note_score),
            pc_onset_plan_select_reserve_fraction=float(args.pc_onset_plan_select_reserve_fraction),
            pc_onset_plan_assign_threshold=float(args.pc_onset_plan_assign_threshold),
            pc_onset_plan_assign_fraction=float(args.pc_onset_plan_assign_fraction),
            pc_onset_plan_assign_window_frames=int(args.pc_onset_plan_assign_window_frames),
            pc_onset_plan_assign_min_note_score=float(args.pc_onset_plan_assign_min_note_score),
            pc_onset_plan_assign_source_weight=float(args.pc_onset_plan_assign_source_weight),
            pc_onset_plan_assign_event_weight=float(args.pc_onset_plan_assign_event_weight),
            pc_onset_plan_assign_distance_penalty=float(args.pc_onset_plan_assign_distance_penalty),
            source_chroma_guidance_weight=float(args.source_chroma_guidance_weight),
            harmonic_plan_guidance_weight=float(args.harmonic_plan_guidance_weight),
            chord_plan_guidance_weight=float(args.chord_plan_guidance_weight),
            bass_plan_guidance_weight=float(args.bass_plan_guidance_weight),
            voicing_plan_guidance_weight=float(args.voicing_plan_guidance_weight),
            section_diversity_guidance_weight=float(args.section_diversity_guidance_weight),
            section_diversity_reserve_fraction=float(args.section_diversity_reserve_fraction),
            section_diversity_reserve_min_note_score=float(args.section_diversity_reserve_min_note_score),
            section_diversity_unique_weight=float(args.section_diversity_unique_weight),
            section_diversity_pc_weight=float(args.section_diversity_pc_weight),
            section_diversity_range_weight=float(args.section_diversity_range_weight),
            section_diversity_onset_weight=float(args.section_diversity_onset_weight),
            section_diversity_section_seconds=float(args.section_diversity_section_seconds),
            source_energy_velocity_weight=float(args.source_energy_velocity_weight),
            density_plan_velocity_weight=float(args.density_plan_velocity_weight),
            device=str(args.device),
            render_wav=not bool(args.no_wav),
        )
        _print_json(infer_piano_arrangement(cfg))
        return
    if args.action == "infer-chunked":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --action infer-chunked")
        if args.source_audio is None:
            raise ValueError("--source-audio is required for --action infer-chunked")
        cfg = PianoChunkedInferenceConfig(
            checkpoint=args.checkpoint,
            source_audio=args.source_audio,
            out_stem=args.out_stem,
            seconds=float(args.seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            onset_threshold=float(args.onset_threshold),
            frame_threshold=float(args.frame_threshold),
            max_notes_per_second=float(args.max_notes_per_second),
            max_simultaneous_notes=int(args.max_simultaneous_notes),
            max_onsets_per_frame=int(args.max_onsets_per_frame),
            max_pitch_fraction=float(args.max_pitch_fraction),
            max_pitch_class_fraction=float(args.max_pitch_class_fraction),
            min_note_duration=float(args.min_note_duration),
            max_note_duration=float(args.max_note_duration),
            bass_min_note_duration=float(args.bass_min_note_duration),
            min_selected_notes=int(args.min_selected_notes),
            min_unique_pitches=int(args.min_unique_pitches),
            require_register_coverage=not bool(args.no_register_coverage),
            register_coverage_chunk_seconds=float(args.register_coverage_chunk_seconds),
            section_bass_repair=bool(args.section_bass_repair),
            section_bass_repair_min_coverage=float(args.section_bass_repair_min_coverage),
            section_diversity_repair=bool(args.section_diversity_repair),
            section_diversity_repair_min_unique_pitches=int(args.section_diversity_repair_min_unique_pitches),
            section_diversity_repair_min_chord_frame=float(args.section_diversity_repair_min_chord_frame),
            section_diversity_repair_max_notes=int(args.section_diversity_repair_max_notes),
            diversity_fallback_threshold=float(args.diversity_fallback_threshold),
            source_onset_guidance_weight=float(args.source_onset_guidance_weight),
            source_onset_snap_frames=int(args.source_onset_snap_frames),
            source_onset_peak_threshold=float(args.source_onset_peak_threshold),
            density_plan_guidance_weight=float(args.density_plan_guidance_weight),
            density_plan_snap_frames=int(args.density_plan_snap_frames),
            density_plan_peak_threshold=float(args.density_plan_peak_threshold),
            event_plan_guidance_weight=float(args.event_plan_guidance_weight),
            event_plan_snap_frames=int(args.event_plan_snap_frames),
            event_plan_peak_threshold=float(args.event_plan_peak_threshold),
            pc_onset_plan_guidance_weight=float(args.pc_onset_plan_guidance_weight),
            pc_onset_plan_reserve_threshold=float(args.pc_onset_plan_reserve_threshold),
            pc_onset_plan_reserve_max_per_frame=int(args.pc_onset_plan_reserve_max_per_frame),
            pc_onset_plan_reserve_min_note_score=float(args.pc_onset_plan_reserve_min_note_score),
            pc_onset_plan_select_reserve_fraction=float(args.pc_onset_plan_select_reserve_fraction),
            pc_onset_plan_assign_threshold=float(args.pc_onset_plan_assign_threshold),
            pc_onset_plan_assign_fraction=float(args.pc_onset_plan_assign_fraction),
            pc_onset_plan_assign_window_frames=int(args.pc_onset_plan_assign_window_frames),
            pc_onset_plan_assign_min_note_score=float(args.pc_onset_plan_assign_min_note_score),
            pc_onset_plan_assign_source_weight=float(args.pc_onset_plan_assign_source_weight),
            pc_onset_plan_assign_event_weight=float(args.pc_onset_plan_assign_event_weight),
            pc_onset_plan_assign_distance_penalty=float(args.pc_onset_plan_assign_distance_penalty),
            source_chroma_guidance_weight=float(args.source_chroma_guidance_weight),
            harmonic_plan_guidance_weight=float(args.harmonic_plan_guidance_weight),
            chord_plan_guidance_weight=float(args.chord_plan_guidance_weight),
            bass_plan_guidance_weight=float(args.bass_plan_guidance_weight),
            voicing_plan_guidance_weight=float(args.voicing_plan_guidance_weight),
            section_diversity_guidance_weight=float(args.section_diversity_guidance_weight),
            section_diversity_reserve_fraction=float(args.section_diversity_reserve_fraction),
            section_diversity_reserve_min_note_score=float(args.section_diversity_reserve_min_note_score),
            section_diversity_unique_weight=float(args.section_diversity_unique_weight),
            section_diversity_pc_weight=float(args.section_diversity_pc_weight),
            section_diversity_range_weight=float(args.section_diversity_range_weight),
            section_diversity_onset_weight=float(args.section_diversity_onset_weight),
            section_diversity_section_seconds=float(args.section_diversity_section_seconds),
            source_energy_velocity_weight=float(args.source_energy_velocity_weight),
            density_plan_velocity_weight=float(args.density_plan_velocity_weight),
            device=str(args.device),
            render_wav=not bool(args.no_wav),
            chunk_seconds=float(args.chunk_seconds),
            chunk_hop_seconds=float(args.chunk_hop_seconds),
            section_profile=str(args.section_profile),
        )
        _print_json(infer_piano_arrangement_chunked(cfg))
        return
    if args.action == "validate-paired":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --action validate-paired")
        cfg = PairedCheckpointBatchEvalConfig(
            checkpoint=args.checkpoint,
            paired_manifest=args.paired_manifest,
            out_dir=args.batch_out_dir
            if args.batch_out_dir is not None
            else DEFAULT_OUTPUT_DIR.parent / "batch_eval" / "paired_checkpoint",
            seconds=float(args.seconds),
            max_frames=int(args.max_frames),
            frame_hz=float(args.frame_hz),
            max_rows=int(args.max_rows),
            onset_threshold=float(args.onset_threshold),
            frame_threshold=float(args.frame_threshold),
            max_notes_per_second=float(args.max_notes_per_second),
            max_simultaneous_notes=int(args.max_simultaneous_notes),
            max_onsets_per_frame=int(args.max_onsets_per_frame),
            max_pitch_fraction=float(args.max_pitch_fraction),
            max_pitch_class_fraction=float(args.max_pitch_class_fraction),
            min_note_duration=float(args.min_note_duration),
            max_note_duration=float(args.max_note_duration),
            bass_min_note_duration=float(args.bass_min_note_duration),
            min_selected_notes=int(args.min_selected_notes),
            min_unique_pitches=int(args.min_unique_pitches),
            require_register_coverage=not bool(args.no_register_coverage),
            register_coverage_chunk_seconds=float(args.register_coverage_chunk_seconds),
            section_bass_repair=bool(args.section_bass_repair),
            section_bass_repair_min_coverage=float(args.section_bass_repair_min_coverage),
            section_diversity_repair=bool(args.section_diversity_repair),
            section_diversity_repair_min_unique_pitches=int(args.section_diversity_repair_min_unique_pitches),
            section_diversity_repair_min_chord_frame=float(args.section_diversity_repair_min_chord_frame),
            section_diversity_repair_max_notes=int(args.section_diversity_repair_max_notes),
            diversity_fallback_threshold=float(args.diversity_fallback_threshold),
            source_onset_guidance_weight=float(args.source_onset_guidance_weight),
            source_onset_snap_frames=int(args.source_onset_snap_frames),
            source_onset_peak_threshold=float(args.source_onset_peak_threshold),
            density_plan_guidance_weight=float(args.density_plan_guidance_weight),
            density_plan_snap_frames=int(args.density_plan_snap_frames),
            density_plan_peak_threshold=float(args.density_plan_peak_threshold),
            event_plan_guidance_weight=float(args.event_plan_guidance_weight),
            event_plan_snap_frames=int(args.event_plan_snap_frames),
            event_plan_peak_threshold=float(args.event_plan_peak_threshold),
            pc_onset_plan_guidance_weight=float(args.pc_onset_plan_guidance_weight),
            pc_onset_plan_reserve_threshold=float(args.pc_onset_plan_reserve_threshold),
            pc_onset_plan_reserve_max_per_frame=int(args.pc_onset_plan_reserve_max_per_frame),
            pc_onset_plan_reserve_min_note_score=float(args.pc_onset_plan_reserve_min_note_score),
            pc_onset_plan_select_reserve_fraction=float(args.pc_onset_plan_select_reserve_fraction),
            pc_onset_plan_assign_threshold=float(args.pc_onset_plan_assign_threshold),
            pc_onset_plan_assign_fraction=float(args.pc_onset_plan_assign_fraction),
            pc_onset_plan_assign_window_frames=int(args.pc_onset_plan_assign_window_frames),
            pc_onset_plan_assign_min_note_score=float(args.pc_onset_plan_assign_min_note_score),
            pc_onset_plan_assign_source_weight=float(args.pc_onset_plan_assign_source_weight),
            pc_onset_plan_assign_event_weight=float(args.pc_onset_plan_assign_event_weight),
            pc_onset_plan_assign_distance_penalty=float(args.pc_onset_plan_assign_distance_penalty),
            source_chroma_guidance_weight=float(args.source_chroma_guidance_weight),
            harmonic_plan_guidance_weight=float(args.harmonic_plan_guidance_weight),
            chord_plan_guidance_weight=float(args.chord_plan_guidance_weight),
            bass_plan_guidance_weight=float(args.bass_plan_guidance_weight),
            voicing_plan_guidance_weight=float(args.voicing_plan_guidance_weight),
            section_diversity_guidance_weight=float(args.section_diversity_guidance_weight),
            section_diversity_reserve_fraction=float(args.section_diversity_reserve_fraction),
            section_diversity_reserve_min_note_score=float(args.section_diversity_reserve_min_note_score),
            section_diversity_unique_weight=float(args.section_diversity_unique_weight),
            section_diversity_pc_weight=float(args.section_diversity_pc_weight),
            section_diversity_range_weight=float(args.section_diversity_range_weight),
            section_diversity_onset_weight=float(args.section_diversity_onset_weight),
            section_diversity_section_seconds=float(args.section_diversity_section_seconds),
            source_energy_velocity_weight=float(args.source_energy_velocity_weight),
            density_plan_velocity_weight=float(args.density_plan_velocity_weight),
            target_eval=not bool(args.no_target_eval),
            min_target_global_chroma_cosine=float(args.min_target_global_chroma_cosine),
            min_target_active_chroma_cosine=float(args.min_target_active_chroma_cosine),
            min_target_onset_correlation=float(args.min_target_onset_correlation),
            min_target_onset_frame_f1=float(args.min_target_onset_frame_f1),
            min_target_pitch_class_onset_f1=float(args.min_target_pitch_class_onset_f1),
            min_target_note_count_ratio=float(args.min_target_note_count_ratio),
            max_target_note_count_ratio=float(args.max_target_note_count_ratio),
            chunked=bool(args.batch_chunked),
            chunk_seconds=float(args.chunk_seconds),
            chunk_hop_seconds=float(args.chunk_hop_seconds),
            section_profile=str(args.section_profile),
            section_seconds=float(args.section_seconds),
            device=str(args.device),
            render_wav=not bool(args.no_wav),
        )
        _print_json(validate_paired_checkpoint(cfg))
        return
    if args.action == "audit-sources":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for --action audit-sources")
        cfg = SourceManifestAuditConfig(
            checkpoint=args.checkpoint,
            source_manifest=args.source_manifest,
            out_dir=args.batch_out_dir
            if args.batch_out_dir is not None
            else DEFAULT_OUTPUT_DIR.parent / "source_audit" / "checkpoint",
            seconds=float(args.seconds),
            max_frames=int(args.max_frames),
            eval_max_frames=int(args.eval_max_frames),
            frame_hz=float(args.frame_hz),
            max_rows=int(args.max_rows),
            onset_threshold=float(args.onset_threshold),
            frame_threshold=float(args.frame_threshold),
            max_notes_per_second=float(args.max_notes_per_second),
            max_simultaneous_notes=int(args.max_simultaneous_notes),
            max_onsets_per_frame=int(args.max_onsets_per_frame),
            max_pitch_fraction=float(args.max_pitch_fraction),
            max_pitch_class_fraction=float(args.max_pitch_class_fraction),
            min_note_duration=float(args.min_note_duration),
            max_note_duration=float(args.max_note_duration),
            bass_min_note_duration=float(args.bass_min_note_duration),
            min_selected_notes=int(args.min_selected_notes),
            min_unique_pitches=int(args.min_unique_pitches),
            require_register_coverage=not bool(args.no_register_coverage),
            register_coverage_chunk_seconds=float(args.register_coverage_chunk_seconds),
            section_bass_repair=bool(args.section_bass_repair),
            section_bass_repair_min_coverage=float(args.section_bass_repair_min_coverage),
            section_diversity_repair=bool(args.section_diversity_repair),
            section_diversity_repair_min_unique_pitches=int(args.section_diversity_repair_min_unique_pitches),
            section_diversity_repair_min_chord_frame=float(args.section_diversity_repair_min_chord_frame),
            section_diversity_repair_max_notes=int(args.section_diversity_repair_max_notes),
            diversity_fallback_threshold=float(args.diversity_fallback_threshold),
            source_onset_guidance_weight=float(args.source_onset_guidance_weight),
            source_onset_snap_frames=int(args.source_onset_snap_frames),
            source_onset_peak_threshold=float(args.source_onset_peak_threshold),
            density_plan_guidance_weight=float(args.density_plan_guidance_weight),
            density_plan_snap_frames=int(args.density_plan_snap_frames),
            density_plan_peak_threshold=float(args.density_plan_peak_threshold),
            event_plan_guidance_weight=float(args.event_plan_guidance_weight),
            event_plan_snap_frames=int(args.event_plan_snap_frames),
            event_plan_peak_threshold=float(args.event_plan_peak_threshold),
            pc_onset_plan_guidance_weight=float(args.pc_onset_plan_guidance_weight),
            pc_onset_plan_reserve_threshold=float(args.pc_onset_plan_reserve_threshold),
            pc_onset_plan_reserve_max_per_frame=int(args.pc_onset_plan_reserve_max_per_frame),
            pc_onset_plan_reserve_min_note_score=float(args.pc_onset_plan_reserve_min_note_score),
            pc_onset_plan_select_reserve_fraction=float(args.pc_onset_plan_select_reserve_fraction),
            pc_onset_plan_assign_threshold=float(args.pc_onset_plan_assign_threshold),
            pc_onset_plan_assign_fraction=float(args.pc_onset_plan_assign_fraction),
            pc_onset_plan_assign_window_frames=int(args.pc_onset_plan_assign_window_frames),
            pc_onset_plan_assign_min_note_score=float(args.pc_onset_plan_assign_min_note_score),
            pc_onset_plan_assign_source_weight=float(args.pc_onset_plan_assign_source_weight),
            pc_onset_plan_assign_event_weight=float(args.pc_onset_plan_assign_event_weight),
            pc_onset_plan_assign_distance_penalty=float(args.pc_onset_plan_assign_distance_penalty),
            source_chroma_guidance_weight=float(args.source_chroma_guidance_weight),
            harmonic_plan_guidance_weight=float(args.harmonic_plan_guidance_weight),
            chord_plan_guidance_weight=float(args.chord_plan_guidance_weight),
            bass_plan_guidance_weight=float(args.bass_plan_guidance_weight),
            voicing_plan_guidance_weight=float(args.voicing_plan_guidance_weight),
            section_diversity_guidance_weight=float(args.section_diversity_guidance_weight),
            section_diversity_reserve_fraction=float(args.section_diversity_reserve_fraction),
            section_diversity_reserve_min_note_score=float(args.section_diversity_reserve_min_note_score),
            section_diversity_unique_weight=float(args.section_diversity_unique_weight),
            section_diversity_pc_weight=float(args.section_diversity_pc_weight),
            section_diversity_range_weight=float(args.section_diversity_range_weight),
            section_diversity_onset_weight=float(args.section_diversity_onset_weight),
            section_diversity_section_seconds=float(args.section_diversity_section_seconds),
            source_energy_velocity_weight=float(args.source_energy_velocity_weight),
            density_plan_velocity_weight=float(args.density_plan_velocity_weight),
            chunked=bool(args.batch_chunked),
            chunk_seconds=float(args.chunk_seconds),
            chunk_hop_seconds=float(args.chunk_hop_seconds),
            section_profile=str(args.section_profile),
            section_seconds=float(args.section_seconds),
            device=str(args.device),
            render_wav=not bool(args.no_wav),
        )
        _print_json(audit_source_manifest(cfg))
        return
    if args.action == "evaluate":
        arrangement_json = args.arrangement_json
        if arrangement_json is None and str(args.out_stem) != "":
            arrangement_json = args.out_stem.with_suffix(".json")
        if arrangement_json is None:
            raise ValueError("--arrangement-json or --out-stem is required for --action evaluate")
        cfg = PianoEvalConfig(
            arrangement_json=arrangement_json,
            report_path=args.eval_report,
            label=str(args.eval_label),
            source_audio=args.source_audio,
            source_seconds=float(args.seconds),
            frame_hz=float(args.frame_hz),
            max_frames=int(args.max_frames),
        )
        _print_json(evaluate_arrangement_file(cfg))
        return
    if args.action == "section-report":
        arrangement_json = args.arrangement_json
        if arrangement_json is None and str(args.out_stem) != "":
            arrangement_json = args.out_stem.with_suffix(".json")
        if arrangement_json is None:
            raise ValueError("--arrangement-json or --out-stem is required for --action section-report")
        cfg = PianoSectionReportConfig(
            arrangement_json=arrangement_json,
            report_path=args.eval_report,
            label=str(args.eval_label),
            section_seconds=float(args.section_seconds),
        )
        _print_json(section_report_arrangement_file(cfg))
        return
    _not_ready(str(args.action))


if __name__ == "__main__":
    main()
