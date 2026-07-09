from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .piano_arranger_cache import DIFFUSION_SR, extract_source_condition, load_audio_for_cache, midi_to_arrangement
from .piano_arranger_eval import (
    PianoEvalConfig,
    PianoSectionReportConfig,
    evaluate_arrangement_file,
    section_report_arrangement_file,
    target_midi_alignment_metrics,
    target_midi_alignment_warnings,
)
from .piano_arranger_infer import (
    PianoChunkedInferenceConfig,
    PianoInferenceConfig,
    _device,
    _load_model,
    _resolved_frames_and_rate,
    infer_piano_arrangement,
    infer_piano_arrangement_chunked,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRED_BATCH_EVAL_DIR = REPO_ROOT / "saves2" / "piano_arranger" / "batch_eval" / "paired_checkpoint"
DEFAULT_SOURCE_AUDIT_DIR = REPO_ROOT / "saves2" / "piano_arranger" / "source_audit"


@dataclass(frozen=True)
class PairedCheckpointBatchEvalConfig:
    checkpoint: Path
    paired_manifest: Path
    out_dir: Path = DEFAULT_PAIRED_BATCH_EVAL_DIR
    seconds: float = 30.0
    max_frames: int = 256
    frame_hz: float = 25.0
    max_rows: int = 0
    onset_threshold: float = 0.35
    frame_threshold: float = 0.35
    max_notes_per_second: float = 32.0
    max_simultaneous_notes: int = 12
    max_onsets_per_frame: int = 6
    max_pitch_fraction: float = 0.22
    max_pitch_class_fraction: float = 0.32
    min_note_duration: float = 0.08
    max_note_duration: float = 1.5
    bass_min_note_duration: float = 0.0
    min_selected_notes: int = 24
    min_unique_pitches: int = 8
    require_register_coverage: bool = True
    register_coverage_chunk_seconds: float = 0.0
    section_bass_repair: bool = False
    section_bass_repair_min_coverage: float = 0.05
    section_diversity_repair: bool = False
    section_diversity_repair_min_unique_pitches: int = 8
    section_diversity_repair_min_chord_frame: float = 0.15
    section_diversity_repair_max_notes: int = 4
    diversity_fallback_threshold: float = 0.05
    source_onset_guidance_weight: float = 0.0
    source_onset_snap_frames: int = 0
    source_onset_peak_threshold: float = 0.35
    density_plan_guidance_weight: float = 0.0
    density_plan_snap_frames: int = 0
    density_plan_peak_threshold: float = 0.35
    event_plan_guidance_weight: float = 0.0
    event_plan_snap_frames: int = 0
    event_plan_peak_threshold: float = 0.35
    pc_onset_plan_guidance_weight: float = 0.0
    pc_onset_plan_reserve_threshold: float = 0.0
    pc_onset_plan_reserve_max_per_frame: int = 0
    pc_onset_plan_reserve_min_note_score: float = 0.02
    pc_onset_plan_select_reserve_fraction: float = 0.0
    pc_onset_plan_assign_threshold: float = 0.0
    pc_onset_plan_assign_fraction: float = 0.0
    pc_onset_plan_assign_window_frames: int = 1
    pc_onset_plan_assign_min_note_score: float = 0.02
    pc_onset_plan_assign_source_weight: float = 0.0
    pc_onset_plan_assign_event_weight: float = 0.0
    pc_onset_plan_assign_distance_penalty: float = 1.0
    source_chroma_guidance_weight: float = 0.0
    harmonic_plan_guidance_weight: float = 0.0
    chord_plan_guidance_weight: float = 0.0
    bass_plan_guidance_weight: float = 0.0
    voicing_plan_guidance_weight: float = 0.0
    section_diversity_guidance_weight: float = 0.0
    section_diversity_reserve_fraction: float = 0.0
    section_diversity_reserve_min_note_score: float = 0.02
    section_diversity_unique_weight: float = 1.0
    section_diversity_pc_weight: float = 1.0
    section_diversity_range_weight: float = 0.5
    section_diversity_onset_weight: float = 0.5
    section_diversity_section_seconds: float = 4.0
    source_energy_velocity_weight: float = 0.0
    density_plan_velocity_weight: float = 0.0
    target_eval: bool = True
    min_target_global_chroma_cosine: float = 0.20
    min_target_active_chroma_cosine: float = 0.20
    min_target_onset_correlation: float = 0.02
    min_target_onset_frame_f1: float = 0.0
    min_target_pitch_class_onset_f1: float = 0.0
    min_target_note_count_ratio: float = 0.0
    max_target_note_count_ratio: float = 0.0
    chunked: bool = False
    chunk_seconds: float = 12.0
    chunk_hop_seconds: float = 0.0
    section_profile: str = "flat"
    section_seconds: float = 8.0
    device: str = "auto"
    render_wav: bool = True


@dataclass(frozen=True)
class SourceManifestAuditConfig:
    checkpoint: Path
    source_manifest: Path
    out_dir: Path = DEFAULT_SOURCE_AUDIT_DIR
    seconds: float = 24.0
    max_frames: int = 256
    eval_max_frames: int = 0
    frame_hz: float = 25.0
    max_rows: int = 0
    onset_threshold: float = 0.35
    frame_threshold: float = 0.35
    max_notes_per_second: float = 24.0
    max_simultaneous_notes: int = 10
    max_onsets_per_frame: int = 4
    max_pitch_fraction: float = 0.18
    max_pitch_class_fraction: float = 0.30
    min_note_duration: float = 0.08
    max_note_duration: float = 1.5
    bass_min_note_duration: float = 0.35
    min_selected_notes: int = 112
    min_unique_pitches: int = 48
    require_register_coverage: bool = True
    register_coverage_chunk_seconds: float = 4.0
    section_bass_repair: bool = True
    section_bass_repair_min_coverage: float = 0.05
    section_diversity_repair: bool = False
    section_diversity_repair_min_unique_pitches: int = 8
    section_diversity_repair_min_chord_frame: float = 0.15
    section_diversity_repair_max_notes: int = 4
    diversity_fallback_threshold: float = 0.05
    source_onset_guidance_weight: float = 0.8
    source_onset_snap_frames: int = 2
    source_onset_peak_threshold: float = 0.25
    density_plan_guidance_weight: float = 0.8
    density_plan_snap_frames: int = 1
    density_plan_peak_threshold: float = 0.25
    event_plan_guidance_weight: float = 0.8
    event_plan_snap_frames: int = 2
    event_plan_peak_threshold: float = 0.25
    pc_onset_plan_guidance_weight: float = 0.10
    pc_onset_plan_reserve_threshold: float = 0.0
    pc_onset_plan_reserve_max_per_frame: int = 0
    pc_onset_plan_reserve_min_note_score: float = 0.02
    pc_onset_plan_select_reserve_fraction: float = 0.0
    pc_onset_plan_assign_threshold: float = 0.65
    pc_onset_plan_assign_fraction: float = 1.0
    pc_onset_plan_assign_window_frames: int = 2
    pc_onset_plan_assign_min_note_score: float = 0.02
    pc_onset_plan_assign_source_weight: float = 2.0
    pc_onset_plan_assign_event_weight: float = 1.0
    pc_onset_plan_assign_distance_penalty: float = 0.25
    source_chroma_guidance_weight: float = 0.2
    harmonic_plan_guidance_weight: float = 0.2
    chord_plan_guidance_weight: float = 0.2
    bass_plan_guidance_weight: float = 0.25
    voicing_plan_guidance_weight: float = 0.2
    section_diversity_guidance_weight: float = 0.0
    section_diversity_reserve_fraction: float = 0.0
    section_diversity_reserve_min_note_score: float = 0.02
    section_diversity_unique_weight: float = 1.0
    section_diversity_pc_weight: float = 1.0
    section_diversity_range_weight: float = 0.5
    section_diversity_onset_weight: float = 0.5
    section_diversity_section_seconds: float = 4.0
    source_energy_velocity_weight: float = 0.6
    density_plan_velocity_weight: float = 0.65
    chunked: bool = True
    chunk_seconds: float = 4.0
    chunk_hop_seconds: float = 2.0
    section_profile: str = "arc"
    section_seconds: float = 4.0
    device: str = "auto"
    render_wav: bool = True


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _resolve_manifest_path(raw: Any, manifest: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(manifest).parent / path


def _safe_stem(raw: Any, fallback: str) -> str:
    stem = Path(str(raw)).stem if str(raw) else fallback
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return stem or fallback


def _metric(row: Dict[str, Any], key: str) -> float:
    value = row.get("metrics", {}).get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _warning_code(warning: str) -> str:
    return str(warning).split(":", 1)[0].strip()


def _mean_metric(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = np.asarray([_metric(row, key) for row in rows], dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _mean_row_value(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key)))
        except (TypeError, ValueError):
            continue
    vals_arr = np.asarray(vals, dtype=np.float32)
    vals_arr = vals_arr[np.isfinite(vals_arr)]
    if vals_arr.size == 0:
        return None
    return float(np.mean(vals_arr))


def _min_row_value(rows: List[Dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key)))
        except (TypeError, ValueError):
            continue
    vals_arr = np.asarray(vals, dtype=np.float32)
    vals_arr = vals_arr[np.isfinite(vals_arr)]
    if vals_arr.size == 0:
        return None
    return float(np.min(vals_arr))


def _source_audio_column(df: pd.DataFrame, manifest: Path) -> str:
    for col in ("source_audio", "path", "audio_path"):
        if col in df.columns:
            return str(col)
    raise ValueError(f"Source manifest missing source path column; expected source_audio, path, or audio_path: {manifest}")


def _wav_level_metrics(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"wav_rms": None, "wav_peak": None, "wav_duration_seconds": None}
    audio, sr = sf.read(str(path), always_2d=False)
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    if arr.size == 0:
        return {"wav_rms": 0.0, "wav_peak": 0.0, "wav_duration_seconds": 0.0}
    return {
        "wav_rms": float(np.sqrt(np.mean(np.square(arr)))),
        "wav_peak": float(np.max(np.abs(arr))),
        "wav_duration_seconds": float(arr.shape[0] / max(1, int(sr))),
    }


def _source_infer_kwargs(cfg: SourceManifestAuditConfig | PairedCheckpointBatchEvalConfig, source_audio: Path, out_stem: Path) -> Dict[str, Any]:
    return {
        "checkpoint": Path(cfg.checkpoint),
        "source_audio": Path(source_audio),
        "out_stem": Path(out_stem),
        "seconds": float(cfg.seconds),
        "max_frames": int(cfg.max_frames),
        "frame_hz": float(cfg.frame_hz),
        "onset_threshold": float(cfg.onset_threshold),
        "frame_threshold": float(cfg.frame_threshold),
        "max_notes_per_second": float(cfg.max_notes_per_second),
        "max_simultaneous_notes": int(cfg.max_simultaneous_notes),
        "max_onsets_per_frame": int(cfg.max_onsets_per_frame),
        "max_pitch_fraction": float(cfg.max_pitch_fraction),
        "max_pitch_class_fraction": float(cfg.max_pitch_class_fraction),
        "min_note_duration": float(cfg.min_note_duration),
        "max_note_duration": float(cfg.max_note_duration),
        "bass_min_note_duration": float(cfg.bass_min_note_duration),
        "min_selected_notes": int(cfg.min_selected_notes),
        "min_unique_pitches": int(cfg.min_unique_pitches),
        "require_register_coverage": bool(cfg.require_register_coverage),
        "register_coverage_chunk_seconds": float(cfg.register_coverage_chunk_seconds),
        "section_bass_repair": bool(cfg.section_bass_repair),
        "section_bass_repair_min_coverage": float(cfg.section_bass_repair_min_coverage),
        "section_diversity_repair": bool(cfg.section_diversity_repair),
        "section_diversity_repair_min_unique_pitches": int(cfg.section_diversity_repair_min_unique_pitches),
        "section_diversity_repair_min_chord_frame": float(cfg.section_diversity_repair_min_chord_frame),
        "section_diversity_repair_max_notes": int(cfg.section_diversity_repair_max_notes),
        "diversity_fallback_threshold": float(cfg.diversity_fallback_threshold),
        "source_onset_guidance_weight": float(cfg.source_onset_guidance_weight),
        "source_onset_snap_frames": int(cfg.source_onset_snap_frames),
        "source_onset_peak_threshold": float(cfg.source_onset_peak_threshold),
        "density_plan_guidance_weight": float(cfg.density_plan_guidance_weight),
        "density_plan_snap_frames": int(cfg.density_plan_snap_frames),
        "density_plan_peak_threshold": float(cfg.density_plan_peak_threshold),
        "event_plan_guidance_weight": float(cfg.event_plan_guidance_weight),
        "event_plan_snap_frames": int(cfg.event_plan_snap_frames),
        "event_plan_peak_threshold": float(cfg.event_plan_peak_threshold),
        "pc_onset_plan_guidance_weight": float(cfg.pc_onset_plan_guidance_weight),
        "pc_onset_plan_reserve_threshold": float(cfg.pc_onset_plan_reserve_threshold),
        "pc_onset_plan_reserve_max_per_frame": int(cfg.pc_onset_plan_reserve_max_per_frame),
        "pc_onset_plan_reserve_min_note_score": float(cfg.pc_onset_plan_reserve_min_note_score),
        "pc_onset_plan_select_reserve_fraction": float(cfg.pc_onset_plan_select_reserve_fraction),
        "pc_onset_plan_assign_threshold": float(cfg.pc_onset_plan_assign_threshold),
        "pc_onset_plan_assign_fraction": float(cfg.pc_onset_plan_assign_fraction),
        "pc_onset_plan_assign_window_frames": int(cfg.pc_onset_plan_assign_window_frames),
        "pc_onset_plan_assign_min_note_score": float(cfg.pc_onset_plan_assign_min_note_score),
        "pc_onset_plan_assign_source_weight": float(cfg.pc_onset_plan_assign_source_weight),
        "pc_onset_plan_assign_event_weight": float(cfg.pc_onset_plan_assign_event_weight),
        "pc_onset_plan_assign_distance_penalty": float(cfg.pc_onset_plan_assign_distance_penalty),
        "source_chroma_guidance_weight": float(cfg.source_chroma_guidance_weight),
        "harmonic_plan_guidance_weight": float(cfg.harmonic_plan_guidance_weight),
        "chord_plan_guidance_weight": float(cfg.chord_plan_guidance_weight),
        "bass_plan_guidance_weight": float(cfg.bass_plan_guidance_weight),
        "voicing_plan_guidance_weight": float(cfg.voicing_plan_guidance_weight),
        "section_diversity_guidance_weight": float(cfg.section_diversity_guidance_weight),
        "section_diversity_reserve_fraction": float(cfg.section_diversity_reserve_fraction),
        "section_diversity_reserve_min_note_score": float(cfg.section_diversity_reserve_min_note_score),
        "section_diversity_unique_weight": float(cfg.section_diversity_unique_weight),
        "section_diversity_pc_weight": float(cfg.section_diversity_pc_weight),
        "section_diversity_range_weight": float(cfg.section_diversity_range_weight),
        "section_diversity_onset_weight": float(cfg.section_diversity_onset_weight),
        "section_diversity_section_seconds": float(cfg.section_diversity_section_seconds),
        "source_energy_velocity_weight": float(cfg.source_energy_velocity_weight),
        "density_plan_velocity_weight": float(cfg.density_plan_velocity_weight),
        "device": str(cfg.device),
        "render_wav": bool(cfg.render_wav),
    }


def _target_alignment_warnings(cfg: PairedCheckpointBatchEvalConfig, metrics: Dict[str, Any]) -> List[str]:
    if not bool(cfg.target_eval):
        return []
    return target_midi_alignment_warnings(
        metrics,
        min_target_global_chroma_cosine=float(cfg.min_target_global_chroma_cosine),
        min_target_active_chroma_cosine=float(cfg.min_target_active_chroma_cosine),
        min_target_onset_correlation=float(cfg.min_target_onset_correlation),
        min_target_onset_frame_f1=float(cfg.min_target_onset_frame_f1),
        min_target_pitch_class_onset_f1=float(cfg.min_target_pitch_class_onset_f1),
        min_target_note_count_ratio=float(cfg.min_target_note_count_ratio),
        max_target_note_count_ratio=float(cfg.max_target_note_count_ratio),
    )


def _prf(matches: int, pred_count: int, target_count: int) -> Dict[str, float]:
    precision = float(matches / pred_count) if int(pred_count) > 0 else 0.0
    recall = float(matches / target_count) if int(target_count) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 1e-8 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _match_count_pitch_class(
    pred: List[tuple[int, int]],
    target: List[tuple[int, int]],
    *,
    tolerance: int = 1,
) -> int:
    remaining = [(int(pc), int(frame)) for pc, frame in target]
    matched = 0
    for raw_pc, raw_frame in sorted((int(pc), int(frame)) for pc, frame in pred):
        best_idx = None
        best_dist = int(tolerance) + 1
        for i, (target_pc, target_frame) in enumerate(remaining):
            if int(raw_pc) != int(target_pc):
                continue
            dist = abs(int(raw_frame) - int(target_frame))
            if dist <= int(tolerance) and dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            matched += 1
            remaining.pop(best_idx)
    return int(matched)


def _target_pc_onsets_from_midi(target_midi: Path, *, seconds: float, frame_hz: float, frames: int) -> List[tuple[int, int]]:
    target = midi_to_arrangement(Path(target_midi), seconds=float(seconds))
    out: List[tuple[int, int]] = []
    for note in target.notes:
        frame = max(0, min(int(frames) - 1, int(round(float(note.start) * float(frame_hz)))))
        out.append((int(note.pitch) % 12, int(frame)))
    return out


def _pc_plan_events(plan: np.ndarray, *, threshold: float, local_maxima: bool = True) -> List[tuple[int, int]]:
    arr = np.asarray(plan, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 12:
        return []
    events: List[tuple[int, int]] = []
    for pc in range(12):
        curve = arr[pc]
        for frame, value in enumerate(curve.tolist()):
            if float(value) < float(threshold):
                continue
            if bool(local_maxima):
                left = float(curve[frame - 1]) if int(frame) > 0 else -1.0
                right = float(curve[frame + 1]) if int(frame) + 1 < curve.shape[0] else -1.0
                if float(value) < max(left, right):
                    continue
            events.append((int(pc), int(frame)))
    return events


def _pc_onset_plan_diagnostics(
    *,
    model: Any,
    checkpoint_payload: Dict[str, Any],
    source_audio: Path,
    target_midi: Path,
    cfg: PairedCheckpointBatchEvalConfig,
    device: torch.device,
) -> Dict[str, Any]:
    max_frames, frame_hz = _resolved_frames_and_rate(
        PianoInferenceConfig(
            checkpoint=Path(cfg.checkpoint),
            source_audio=Path(source_audio),
            seconds=float(cfg.seconds),
            max_frames=int(cfg.max_frames),
            frame_hz=float(cfg.frame_hz),
        ),
        checkpoint_payload,
    )
    y = load_audio_for_cache(Path(source_audio), seconds=float(cfg.seconds), sr=DIFFUSION_SR)
    source = extract_source_condition(y, sr=DIFFUSION_SR, frame_hz=float(frame_hz), max_frames=int(max_frames))
    with torch.no_grad():
        pred = model(torch.from_numpy(source).unsqueeze(0).to(device))
    if "pc_onset" not in pred:
        return {"pc_onset_plan_available": False}
    plan = np.clip(pred["pc_onset"][0].detach().cpu().numpy()[:, : int(max_frames)], 0.0, 1.0)
    target_events = _target_pc_onsets_from_midi(
        Path(target_midi),
        seconds=float(cfg.seconds),
        frame_hz=float(frame_hz),
        frames=int(max_frames),
    )
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80]
    best: Dict[str, Any] = {
        "pc_onset_plan_available": True,
        "pc_onset_plan_best_f1": 0.0,
        "pc_onset_plan_target_count": int(len(target_events)),
        "pc_onset_plan_max": float(np.max(plan)) if plan.size else 0.0,
        "pc_onset_plan_mean": float(np.mean(plan)) if plan.size else 0.0,
    }
    sweep: List[Dict[str, Any]] = []
    for threshold in thresholds:
        events = _pc_plan_events(plan, threshold=float(threshold), local_maxima=True)
        matches = _match_count_pitch_class(events, target_events, tolerance=1)
        prf = _prf(matches, len(events), len(target_events))
        row = {
            "threshold": float(threshold),
            "pred_count": int(len(events)),
            "target_count": int(len(target_events)),
            "matches": int(matches),
            "precision": float(prf["precision"]),
            "recall": float(prf["recall"]),
            "f1": float(prf["f1"]),
        }
        sweep.append(row)
        if float(row["f1"]) > float(best["pc_onset_plan_best_f1"]):
            best.update(
                {
                    "pc_onset_plan_best_threshold": float(threshold),
                    "pc_onset_plan_best_pred_count": int(len(events)),
                    "pc_onset_plan_best_matches": int(matches),
                    "pc_onset_plan_best_precision": float(prf["precision"]),
                    "pc_onset_plan_best_recall": float(prf["recall"]),
                    "pc_onset_plan_best_f1": float(prf["f1"]),
                }
            )
    best["pc_onset_plan_threshold_sweep"] = sweep
    return best


def validate_paired_checkpoint(cfg: PairedCheckpointBatchEvalConfig) -> Dict[str, Any]:
    manifest = Path(cfg.paired_manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing paired manifest: {manifest}")
    checkpoint = Path(cfg.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    df = pd.read_csv(manifest)
    required = {"source_audio", "target_midi"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Paired manifest is missing required columns {missing}: {manifest}")
    if int(cfg.max_rows) > 0:
        df = df.head(int(cfg.max_rows)).copy()

    out_dir = Path(cfg.out_dir)
    outputs_dir = out_dir / "outputs"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_device = _device(str(cfg.device))
    plan_model, plan_payload, _plan_model_cfg = _load_model(checkpoint, plan_device)

    rows: List[Dict[str, Any]] = []
    warning_counts: Dict[str, int] = {}
    for row_idx, rec in df.reset_index(drop=True).iterrows():
        source_audio = _resolve_manifest_path(rec["source_audio"], manifest)
        target_midi = _resolve_manifest_path(rec["target_midi"], manifest)
        row_name = f"{int(row_idx):04d}_{_safe_stem(source_audio, f'row_{int(row_idx):04d}')}"
        out_stem = outputs_dir / row_name
        report_path = reports_dir / f"{row_name}.eval.json"
        row_out: Dict[str, Any] = {
            "row": int(row_idx),
            "source_audio": str(source_audio),
            "target_midi": str(target_midi),
            "out_stem": str(out_stem),
            "eval_report": str(report_path),
        }
        try:
            infer_kwargs = {
                "checkpoint": checkpoint,
                "source_audio": source_audio,
                "out_stem": out_stem,
                "seconds": float(cfg.seconds),
                "max_frames": int(cfg.max_frames),
                "frame_hz": float(cfg.frame_hz),
                "onset_threshold": float(cfg.onset_threshold),
                "frame_threshold": float(cfg.frame_threshold),
                "max_notes_per_second": float(cfg.max_notes_per_second),
                "max_simultaneous_notes": int(cfg.max_simultaneous_notes),
                "max_onsets_per_frame": int(cfg.max_onsets_per_frame),
                "max_pitch_fraction": float(cfg.max_pitch_fraction),
                "max_pitch_class_fraction": float(cfg.max_pitch_class_fraction),
                "min_note_duration": float(cfg.min_note_duration),
                "max_note_duration": float(cfg.max_note_duration),
                "bass_min_note_duration": float(cfg.bass_min_note_duration),
                "min_selected_notes": int(cfg.min_selected_notes),
                "min_unique_pitches": int(cfg.min_unique_pitches),
                "require_register_coverage": bool(cfg.require_register_coverage),
                "register_coverage_chunk_seconds": float(cfg.register_coverage_chunk_seconds),
                "section_bass_repair": bool(cfg.section_bass_repair),
                "section_bass_repair_min_coverage": float(cfg.section_bass_repair_min_coverage),
                "section_diversity_repair": bool(cfg.section_diversity_repair),
                "section_diversity_repair_min_unique_pitches": int(cfg.section_diversity_repair_min_unique_pitches),
                "section_diversity_repair_min_chord_frame": float(cfg.section_diversity_repair_min_chord_frame),
                "section_diversity_repair_max_notes": int(cfg.section_diversity_repair_max_notes),
                "diversity_fallback_threshold": float(cfg.diversity_fallback_threshold),
                "source_onset_guidance_weight": float(cfg.source_onset_guidance_weight),
                "source_onset_snap_frames": int(cfg.source_onset_snap_frames),
                "source_onset_peak_threshold": float(cfg.source_onset_peak_threshold),
                "density_plan_guidance_weight": float(cfg.density_plan_guidance_weight),
                "density_plan_snap_frames": int(cfg.density_plan_snap_frames),
                "density_plan_peak_threshold": float(cfg.density_plan_peak_threshold),
                "event_plan_guidance_weight": float(cfg.event_plan_guidance_weight),
                "event_plan_snap_frames": int(cfg.event_plan_snap_frames),
                "event_plan_peak_threshold": float(cfg.event_plan_peak_threshold),
                "pc_onset_plan_guidance_weight": float(cfg.pc_onset_plan_guidance_weight),
                "pc_onset_plan_reserve_threshold": float(cfg.pc_onset_plan_reserve_threshold),
                "pc_onset_plan_reserve_max_per_frame": int(cfg.pc_onset_plan_reserve_max_per_frame),
                "pc_onset_plan_reserve_min_note_score": float(cfg.pc_onset_plan_reserve_min_note_score),
                "pc_onset_plan_select_reserve_fraction": float(cfg.pc_onset_plan_select_reserve_fraction),
                "pc_onset_plan_assign_threshold": float(cfg.pc_onset_plan_assign_threshold),
                "pc_onset_plan_assign_fraction": float(cfg.pc_onset_plan_assign_fraction),
                "pc_onset_plan_assign_window_frames": int(cfg.pc_onset_plan_assign_window_frames),
                "pc_onset_plan_assign_min_note_score": float(cfg.pc_onset_plan_assign_min_note_score),
                "pc_onset_plan_assign_source_weight": float(cfg.pc_onset_plan_assign_source_weight),
                "pc_onset_plan_assign_event_weight": float(cfg.pc_onset_plan_assign_event_weight),
                "pc_onset_plan_assign_distance_penalty": float(cfg.pc_onset_plan_assign_distance_penalty),
                "source_chroma_guidance_weight": float(cfg.source_chroma_guidance_weight),
                "harmonic_plan_guidance_weight": float(cfg.harmonic_plan_guidance_weight),
                "chord_plan_guidance_weight": float(cfg.chord_plan_guidance_weight),
                "bass_plan_guidance_weight": float(cfg.bass_plan_guidance_weight),
                "voicing_plan_guidance_weight": float(cfg.voicing_plan_guidance_weight),
                "section_diversity_guidance_weight": float(cfg.section_diversity_guidance_weight),
                "section_diversity_reserve_fraction": float(cfg.section_diversity_reserve_fraction),
                "section_diversity_reserve_min_note_score": float(cfg.section_diversity_reserve_min_note_score),
                "section_diversity_unique_weight": float(cfg.section_diversity_unique_weight),
                "section_diversity_pc_weight": float(cfg.section_diversity_pc_weight),
                "section_diversity_range_weight": float(cfg.section_diversity_range_weight),
                "section_diversity_onset_weight": float(cfg.section_diversity_onset_weight),
                "section_diversity_section_seconds": float(cfg.section_diversity_section_seconds),
                "source_energy_velocity_weight": float(cfg.source_energy_velocity_weight),
                "density_plan_velocity_weight": float(cfg.density_plan_velocity_weight),
                "device": str(cfg.device),
                "render_wav": bool(cfg.render_wav),
            }
            if bool(cfg.chunked):
                infer_summary = infer_piano_arrangement_chunked(
                    PianoChunkedInferenceConfig(
                        **infer_kwargs,
                        chunk_seconds=float(cfg.chunk_seconds),
                        chunk_hop_seconds=float(cfg.chunk_hop_seconds),
                        section_profile=str(cfg.section_profile),
                    )
                )
            else:
                infer_summary = infer_piano_arrangement(PianoInferenceConfig(**infer_kwargs))
            eval_report = evaluate_arrangement_file(
                PianoEvalConfig(
                    arrangement_json=out_stem.with_suffix(".json"),
                    report_path=report_path,
                    label=row_name,
                    source_audio=source_audio,
                    source_seconds=float(cfg.seconds),
                    frame_hz=float(cfg.frame_hz),
                    max_frames=int(cfg.max_frames),
                )
            )
            section_report_path = reports_dir / f"{row_name}.sections.json"
            section_report = section_report_arrangement_file(
                PianoSectionReportConfig(
                    arrangement_json=out_stem.with_suffix(".json"),
                    report_path=section_report_path,
                    label=row_name,
                    section_seconds=float(cfg.section_seconds),
                )
            )
            metrics = dict(eval_report.get("metrics", {}))
            if bool(cfg.target_eval):
                metrics.update(
                    target_midi_alignment_metrics(
                        arrangement=json.loads(out_stem.with_suffix(".json").read_text(encoding="utf-8")),
                        target_midi=target_midi,
                        target_seconds=float(cfg.seconds),
                        frame_hz=float(cfg.frame_hz),
                        max_frames=int(cfg.max_frames),
                    )
                )
                plan_diag = _pc_onset_plan_diagnostics(
                    model=plan_model,
                    checkpoint_payload=plan_payload,
                    source_audio=source_audio,
                    target_midi=target_midi,
                    cfg=cfg,
                    device=plan_device,
                )
                metrics.update(
                    {
                        key: value
                        for key, value in plan_diag.items()
                        if isinstance(value, (bool, int, float, str)) and key != "pc_onset_plan_available"
                    }
                )
            else:
                plan_diag = {"pc_onset_plan_available": False, "target_eval_disabled": True}
            warnings = [str(w) for w in eval_report.get("warnings", [])]
            warnings.extend(_target_alignment_warnings(cfg, metrics))
            section_warnings = [str(w) for w in section_report.get("warnings", [])]
            warnings.extend(section_warnings)
            for warning in warnings:
                code = _warning_code(warning)
                warning_counts[code] = int(warning_counts.get(code, 0)) + 1
            eval_report["metrics"] = metrics
            eval_report["warnings"] = warnings
            eval_report["passed"] = bool(eval_report.get("passed", False)) and len(warnings) == 0
            eval_report["target_thresholds"] = {
                "enabled": bool(cfg.target_eval),
                "min_target_global_chroma_cosine": float(cfg.min_target_global_chroma_cosine),
                "min_target_active_chroma_cosine": float(cfg.min_target_active_chroma_cosine),
                "min_target_onset_correlation": float(cfg.min_target_onset_correlation),
                "min_target_onset_frame_f1": float(cfg.min_target_onset_frame_f1),
                "min_target_pitch_class_onset_f1": float(cfg.min_target_pitch_class_onset_f1),
                "min_target_note_count_ratio": float(cfg.min_target_note_count_ratio),
                "max_target_note_count_ratio": float(cfg.max_target_note_count_ratio),
            }
            _write_json(report_path, eval_report)
            row_out.update(
                {
                    "passed": bool(eval_report.get("passed", False)),
                    "warnings": warnings,
                    "metrics": metrics,
                    "notes": int(infer_summary.get("notes", 0)),
                    "post_chunk_section_diversity_repairs": int(infer_summary.get("post_chunk_section_diversity_repairs", 0)),
                    "section_diversity_guidance_chunks": int(infer_summary.get("section_diversity_guidance_chunks", 0)),
                    "section_diversity_reserved_notes": int(infer_summary.get("section_diversity_reserved_notes", 0)),
                    "arrangement_json": str(out_stem.with_suffix(".json")),
                    "midi": str(out_stem.with_suffix(".mid")),
                    "wav": str(out_stem.with_suffix(".wav")) if bool(cfg.render_wav) else "",
                    "section_report": str(section_report_path),
                    "section_warnings": section_warnings,
                    "pc_onset_plan_diagnostics": plan_diag,
                }
            )
        except Exception as exc:
            warning_counts["batch_row_error"] = int(warning_counts.get("batch_row_error", 0)) + 1
            row_out.update({"passed": False, "warnings": [f"batch_row_error: {exc}"], "metrics": {}, "notes": 0})
        rows.append(row_out)

    rows_csv = out_dir / "rows.csv"
    csv_fields = [
        "row",
        "passed",
        "source_audio",
        "target_midi",
        "notes",
        "warnings",
        "source_global_chroma_cosine",
        "source_active_chroma_cosine",
        "source_onset_correlation",
        "source_onset_peak_alignment",
        "target_global_chroma_cosine",
        "target_active_chroma_cosine",
        "target_onset_correlation",
        "target_onset_frame_f1",
        "target_pitch_class_onset_f1",
        "target_note_count_ratio",
        "pc_onset_plan_best_f1",
        "pc_onset_plan_best_threshold",
        "pc_onset_plan_best_precision",
        "pc_onset_plan_best_recall",
        "pc_onset_plan_best_pred_count",
        "pc_onset_plan_target_count",
        "target_note_count",
        "target_unique_pitches",
        "section_warnings",
        "section_diversity_guidance_chunks",
        "section_diversity_reserved_notes",
        "unique_pitches",
        "max_simultaneous_notes",
        "bass_note_fraction",
        "mid_note_fraction",
        "high_note_fraction",
        "arrangement_json",
        "midi",
        "wav",
        "eval_report",
        "section_report",
    ]
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row": row.get("row"),
                    "passed": row.get("passed"),
                    "source_audio": row.get("source_audio"),
                    "target_midi": row.get("target_midi"),
                    "notes": row.get("notes"),
                    "warnings": ";".join(str(w) for w in row.get("warnings", [])),
                    "source_global_chroma_cosine": _metric(row, "source_global_chroma_cosine"),
                    "source_active_chroma_cosine": _metric(row, "source_active_chroma_cosine"),
                    "source_onset_correlation": _metric(row, "source_onset_correlation"),
                    "source_onset_peak_alignment": _metric(row, "source_onset_peak_alignment"),
                    "target_global_chroma_cosine": _metric(row, "target_global_chroma_cosine"),
                    "target_active_chroma_cosine": _metric(row, "target_active_chroma_cosine"),
                    "target_onset_correlation": _metric(row, "target_onset_correlation"),
                    "target_onset_frame_f1": _metric(row, "target_onset_frame_f1"),
                    "target_pitch_class_onset_f1": _metric(row, "target_pitch_class_onset_f1"),
                    "target_note_count_ratio": _metric(row, "target_note_count_ratio"),
                    "pc_onset_plan_best_f1": _metric(row, "pc_onset_plan_best_f1"),
                    "pc_onset_plan_best_threshold": _metric(row, "pc_onset_plan_best_threshold"),
                    "pc_onset_plan_best_precision": _metric(row, "pc_onset_plan_best_precision"),
                    "pc_onset_plan_best_recall": _metric(row, "pc_onset_plan_best_recall"),
                    "pc_onset_plan_best_pred_count": _metric(row, "pc_onset_plan_best_pred_count"),
                    "pc_onset_plan_target_count": _metric(row, "pc_onset_plan_target_count"),
                    "target_note_count": _metric(row, "target_note_count"),
                    "target_unique_pitches": _metric(row, "target_unique_pitches"),
                    "section_warnings": ";".join(str(w) for w in row.get("section_warnings", [])),
                    "section_diversity_guidance_chunks": row.get("section_diversity_guidance_chunks"),
                    "section_diversity_reserved_notes": row.get("section_diversity_reserved_notes"),
                    "unique_pitches": _metric(row, "unique_pitches"),
                    "max_simultaneous_notes": _metric(row, "max_simultaneous_notes"),
                    "bass_note_fraction": _metric(row, "bass_note_fraction"),
                    "mid_note_fraction": _metric(row, "mid_note_fraction"),
                    "high_note_fraction": _metric(row, "high_note_fraction"),
                    "arrangement_json": row.get("arrangement_json", ""),
                    "midi": row.get("midi", ""),
                    "wav": row.get("wav", ""),
                    "eval_report": row.get("eval_report", ""),
                    "section_report": row.get("section_report", ""),
                }
            )

    metric_keys = [
        "note_count",
        "unique_pitches",
        "max_simultaneous_notes",
        "bass_note_fraction",
        "mid_note_fraction",
        "high_note_fraction",
        "source_global_chroma_cosine",
        "source_active_chroma_cosine",
        "source_onset_correlation",
        "source_onset_peak_alignment",
        "target_global_chroma_cosine",
        "target_active_chroma_cosine",
        "target_onset_correlation",
        "target_onset_frame_f1",
        "target_pitch_class_onset_f1",
        "target_note_count_ratio",
        "pc_onset_plan_best_f1",
        "pc_onset_plan_best_threshold",
        "pc_onset_plan_best_precision",
        "pc_onset_plan_best_recall",
        "pc_onset_plan_best_pred_count",
        "pc_onset_plan_target_count",
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "paired_manifest": str(manifest),
        "out_dir": str(out_dir),
        "rows_csv": str(rows_csv),
        "summary_json": str(out_dir / "summary.json"),
        "input_rows": int(len(df)),
        "processed_rows": int(len(rows)),
        "passed_rows": int(sum(1 for row in rows if bool(row.get("passed")))),
        "failed_rows": int(sum(1 for row in rows if not bool(row.get("passed")))),
        "warning_counts": warning_counts,
        "mean_metrics": {key: _mean_metric(rows, key) for key in metric_keys},
        "decode": {
            "seconds": float(cfg.seconds),
            "max_frames": int(cfg.max_frames),
            "frame_hz": float(cfg.frame_hz),
            "onset_threshold": float(cfg.onset_threshold),
            "frame_threshold": float(cfg.frame_threshold),
            "max_notes_per_second": float(cfg.max_notes_per_second),
            "max_simultaneous_notes": int(cfg.max_simultaneous_notes),
            "max_onsets_per_frame": int(cfg.max_onsets_per_frame),
            "bass_min_note_duration": float(cfg.bass_min_note_duration),
            "min_selected_notes": int(cfg.min_selected_notes),
            "min_unique_pitches": int(cfg.min_unique_pitches),
            "register_coverage_chunk_seconds": float(cfg.register_coverage_chunk_seconds),
            "section_diversity_repair": bool(cfg.section_diversity_repair),
            "section_diversity_repair_min_unique_pitches": int(cfg.section_diversity_repair_min_unique_pitches),
            "section_diversity_repair_min_chord_frame": float(cfg.section_diversity_repair_min_chord_frame),
            "section_diversity_repair_max_notes": int(cfg.section_diversity_repair_max_notes),
            "density_plan_guidance_weight": float(cfg.density_plan_guidance_weight),
            "density_plan_snap_frames": int(cfg.density_plan_snap_frames),
            "event_plan_guidance_weight": float(cfg.event_plan_guidance_weight),
            "event_plan_snap_frames": int(cfg.event_plan_snap_frames),
            "event_plan_peak_threshold": float(cfg.event_plan_peak_threshold),
            "pc_onset_plan_guidance_weight": float(cfg.pc_onset_plan_guidance_weight),
            "pc_onset_plan_reserve_threshold": float(cfg.pc_onset_plan_reserve_threshold),
            "pc_onset_plan_reserve_max_per_frame": int(cfg.pc_onset_plan_reserve_max_per_frame),
            "pc_onset_plan_reserve_min_note_score": float(cfg.pc_onset_plan_reserve_min_note_score),
            "pc_onset_plan_select_reserve_fraction": float(cfg.pc_onset_plan_select_reserve_fraction),
            "pc_onset_plan_assign_threshold": float(cfg.pc_onset_plan_assign_threshold),
            "pc_onset_plan_assign_fraction": float(cfg.pc_onset_plan_assign_fraction),
            "pc_onset_plan_assign_window_frames": int(cfg.pc_onset_plan_assign_window_frames),
            "pc_onset_plan_assign_min_note_score": float(cfg.pc_onset_plan_assign_min_note_score),
            "pc_onset_plan_assign_source_weight": float(cfg.pc_onset_plan_assign_source_weight),
            "pc_onset_plan_assign_event_weight": float(cfg.pc_onset_plan_assign_event_weight),
            "pc_onset_plan_assign_distance_penalty": float(cfg.pc_onset_plan_assign_distance_penalty),
            "source_chroma_guidance_weight": float(cfg.source_chroma_guidance_weight),
            "harmonic_plan_guidance_weight": float(cfg.harmonic_plan_guidance_weight),
            "chord_plan_guidance_weight": float(cfg.chord_plan_guidance_weight),
            "bass_plan_guidance_weight": float(cfg.bass_plan_guidance_weight),
            "voicing_plan_guidance_weight": float(cfg.voicing_plan_guidance_weight),
            "section_diversity_guidance_weight": float(cfg.section_diversity_guidance_weight),
            "section_diversity_reserve_fraction": float(cfg.section_diversity_reserve_fraction),
            "section_diversity_reserve_min_note_score": float(cfg.section_diversity_reserve_min_note_score),
            "section_diversity_unique_weight": float(cfg.section_diversity_unique_weight),
            "section_diversity_pc_weight": float(cfg.section_diversity_pc_weight),
            "section_diversity_range_weight": float(cfg.section_diversity_range_weight),
            "section_diversity_onset_weight": float(cfg.section_diversity_onset_weight),
            "section_diversity_section_seconds": float(cfg.section_diversity_section_seconds),
            "source_energy_velocity_weight": float(cfg.source_energy_velocity_weight),
            "density_plan_velocity_weight": float(cfg.density_plan_velocity_weight),
            "chunked": bool(cfg.chunked),
            "chunk_seconds": float(cfg.chunk_seconds),
            "chunk_hop_seconds": float(cfg.chunk_hop_seconds),
            "section_profile": str(cfg.section_profile),
            "section_seconds": float(cfg.section_seconds),
        },
        "target_eval": {
            "enabled": bool(cfg.target_eval),
            "min_target_global_chroma_cosine": float(cfg.min_target_global_chroma_cosine),
            "min_target_active_chroma_cosine": float(cfg.min_target_active_chroma_cosine),
            "min_target_onset_correlation": float(cfg.min_target_onset_correlation),
            "min_target_onset_frame_f1": float(cfg.min_target_onset_frame_f1),
            "min_target_pitch_class_onset_f1": float(cfg.min_target_pitch_class_onset_f1),
            "min_target_note_count_ratio": float(cfg.min_target_note_count_ratio),
            "max_target_note_count_ratio": float(cfg.max_target_note_count_ratio),
        },
        "rows": rows[:20],
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def audit_source_manifest(cfg: SourceManifestAuditConfig) -> Dict[str, Any]:
    manifest = Path(cfg.source_manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest}")
    checkpoint = Path(cfg.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    df = pd.read_csv(manifest)
    source_col = _source_audio_column(df, manifest)
    if int(cfg.max_rows) > 0:
        df = df.head(int(cfg.max_rows)).copy()

    out_dir = Path(cfg.out_dir)
    outputs_dir = out_dir / "outputs"
    reports_dir = out_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    warning_counts: Dict[str, int] = {}
    for row_idx, rec in df.reset_index(drop=True).iterrows():
        source_audio = _resolve_manifest_path(rec[source_col], manifest)
        row_name = f"{int(row_idx):04d}_{_safe_stem(source_audio, f'row_{int(row_idx):04d}')}"
        out_stem = outputs_dir / row_name
        report_path = reports_dir / f"{row_name}.eval.json"
        section_report_path = reports_dir / f"{row_name}.sections.json"
        title = str(rec.get("title", "")) if "title" in df.columns else ""
        artist = str(rec.get("artist", "")) if "artist" in df.columns else ""
        row_out: Dict[str, Any] = {
            "row": int(row_idx),
            "source_audio": str(source_audio),
            "title": title,
            "artist": artist,
            "out_stem": str(out_stem),
            "eval_report": str(report_path),
            "section_report": str(section_report_path),
        }
        try:
            infer_kwargs = _source_infer_kwargs(cfg, source_audio, out_stem)
            if bool(cfg.chunked):
                infer_summary = infer_piano_arrangement_chunked(
                    PianoChunkedInferenceConfig(
                        **infer_kwargs,
                        chunk_seconds=float(cfg.chunk_seconds),
                        chunk_hop_seconds=float(cfg.chunk_hop_seconds),
                        section_profile=str(cfg.section_profile),
                    )
                )
            else:
                infer_summary = infer_piano_arrangement(PianoInferenceConfig(**infer_kwargs))

            arrangement_json = out_stem.with_suffix(".json")
            eval_report = evaluate_arrangement_file(
                PianoEvalConfig(
                    arrangement_json=arrangement_json,
                    report_path=report_path,
                    label=row_name,
                    source_audio=source_audio,
                    source_seconds=float(cfg.seconds),
                    frame_hz=float(cfg.frame_hz),
                    max_frames=int(cfg.eval_max_frames) if int(cfg.eval_max_frames) > 0 else int(cfg.max_frames),
                )
            )
            section_report = section_report_arrangement_file(
                PianoSectionReportConfig(
                    arrangement_json=arrangement_json,
                    report_path=section_report_path,
                    label=row_name,
                    section_seconds=float(cfg.section_seconds),
                )
            )
            metrics = dict(eval_report.get("metrics", {}))
            section_summary = dict(section_report.get("summary", {}))
            warnings = [str(w) for w in eval_report.get("warnings", [])]
            section_warnings = [str(w) for w in section_report.get("warnings", [])]
            warnings.extend(section_warnings)
            for warning in warnings:
                code = _warning_code(warning)
                warning_counts[code] = int(warning_counts.get(code, 0)) + 1
            eval_report["warnings"] = warnings
            eval_report["passed"] = bool(eval_report.get("passed", False)) and len(warnings) == 0
            _write_json(report_path, eval_report)

            wav_path = out_stem.with_suffix(".wav")
            wav_metrics = _wav_level_metrics(wav_path) if bool(cfg.render_wav) else {}
            row_out.update(
                {
                    "passed": bool(eval_report.get("passed", False)),
                    "warnings": warnings,
                    "section_warnings": section_warnings,
                    "metrics": metrics,
                    "section_summary": section_summary,
                    "notes": int(infer_summary.get("notes", 0)),
                    "post_chunk_section_bass_repairs": int(infer_summary.get("post_chunk_section_bass_repairs", 0)),
                    "post_chunk_section_diversity_repairs": int(infer_summary.get("post_chunk_section_diversity_repairs", 0)),
                    "section_diversity_guidance_chunks": int(infer_summary.get("section_diversity_guidance_chunks", 0)),
                    "section_diversity_reserved_notes": int(infer_summary.get("section_diversity_reserved_notes", 0)),
                    "arrangement_json": str(arrangement_json),
                    "midi": str(out_stem.with_suffix(".mid")),
                    "wav": str(wav_path) if bool(cfg.render_wav) else "",
                    **wav_metrics,
                    "min_section_notes": section_summary.get("min_section_notes"),
                    "min_section_unique_pitches": section_summary.get("min_section_unique_pitches"),
                    "min_section_bass_coverage_fraction": section_summary.get("min_section_bass_coverage_fraction"),
                    "min_section_chord_frame_fraction": section_summary.get("min_section_chord_frame_fraction"),
                    "min_section_fullness_score": section_summary.get("min_section_fullness_score"),
                    "mean_section_fullness_score": section_summary.get("mean_section_fullness_score"),
                }
            )
        except Exception as exc:
            warning_counts["source_audit_row_error"] = int(warning_counts.get("source_audit_row_error", 0)) + 1
            row_out.update(
                {
                    "passed": False,
                    "warnings": [f"source_audit_row_error: {exc}"],
                    "section_warnings": [],
                    "metrics": {},
                    "section_summary": {},
                    "notes": 0,
                    "arrangement_json": "",
                    "midi": "",
                    "wav": "",
                }
            )
        rows.append(row_out)

    rows_csv = out_dir / "rows.csv"
    csv_fields = [
        "row",
        "passed",
        "source_audio",
        "title",
        "artist",
        "notes",
        "warnings",
        "section_warnings",
        "source_global_chroma_cosine",
        "source_active_chroma_cosine",
        "source_onset_correlation",
        "source_onset_peak_alignment",
        "unique_pitches",
        "max_simultaneous_notes",
        "single_pitch_class_fraction",
        "bass_note_fraction",
        "mid_note_fraction",
        "high_note_fraction",
        "weighted_mean_velocity",
        "chord_frame_fraction",
        "bass_coverage_fraction",
        "melody_coverage_fraction",
        "mean_active_polyphony",
        "fullness_score",
        "min_section_notes",
        "min_section_unique_pitches",
        "min_section_bass_coverage_fraction",
        "min_section_chord_frame_fraction",
        "min_section_fullness_score",
        "mean_section_fullness_score",
        "post_chunk_section_bass_repairs",
        "post_chunk_section_diversity_repairs",
        "section_diversity_guidance_chunks",
        "section_diversity_reserved_notes",
        "wav_rms",
        "wav_peak",
        "arrangement_json",
        "midi",
        "wav",
        "eval_report",
        "section_report",
    ]
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row": row.get("row"),
                    "passed": row.get("passed"),
                    "source_audio": row.get("source_audio"),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "notes": row.get("notes"),
                    "warnings": ";".join(str(w) for w in row.get("warnings", [])),
                    "section_warnings": ";".join(str(w) for w in row.get("section_warnings", [])),
                    "source_global_chroma_cosine": _metric(row, "source_global_chroma_cosine"),
                    "source_active_chroma_cosine": _metric(row, "source_active_chroma_cosine"),
                    "source_onset_correlation": _metric(row, "source_onset_correlation"),
                    "source_onset_peak_alignment": _metric(row, "source_onset_peak_alignment"),
                    "unique_pitches": _metric(row, "unique_pitches"),
                    "max_simultaneous_notes": _metric(row, "max_simultaneous_notes"),
                    "single_pitch_class_fraction": _metric(row, "single_pitch_class_fraction"),
                    "bass_note_fraction": _metric(row, "bass_note_fraction"),
                    "mid_note_fraction": _metric(row, "mid_note_fraction"),
                    "high_note_fraction": _metric(row, "high_note_fraction"),
                    "weighted_mean_velocity": _metric(row, "weighted_mean_velocity"),
                    "chord_frame_fraction": _metric(row, "chord_frame_fraction"),
                    "bass_coverage_fraction": _metric(row, "bass_coverage_fraction"),
                    "melody_coverage_fraction": _metric(row, "melody_coverage_fraction"),
                    "mean_active_polyphony": _metric(row, "mean_active_polyphony"),
                    "fullness_score": _metric(row, "fullness_score"),
                    "min_section_notes": row.get("min_section_notes"),
                    "min_section_unique_pitches": row.get("min_section_unique_pitches"),
                    "min_section_bass_coverage_fraction": row.get("min_section_bass_coverage_fraction"),
                    "min_section_chord_frame_fraction": row.get("min_section_chord_frame_fraction"),
                    "min_section_fullness_score": row.get("min_section_fullness_score"),
                    "mean_section_fullness_score": row.get("mean_section_fullness_score"),
                    "post_chunk_section_bass_repairs": row.get("post_chunk_section_bass_repairs"),
                    "post_chunk_section_diversity_repairs": row.get("post_chunk_section_diversity_repairs"),
                    "section_diversity_guidance_chunks": row.get("section_diversity_guidance_chunks"),
                    "section_diversity_reserved_notes": row.get("section_diversity_reserved_notes"),
                    "wav_rms": row.get("wav_rms"),
                    "wav_peak": row.get("wav_peak"),
                    "arrangement_json": row.get("arrangement_json", ""),
                    "midi": row.get("midi", ""),
                    "wav": row.get("wav", ""),
                    "eval_report": row.get("eval_report", ""),
                    "section_report": row.get("section_report", ""),
                }
            )

    metric_keys = [
        "note_count",
        "unique_pitches",
        "max_simultaneous_notes",
        "bass_note_fraction",
        "mid_note_fraction",
        "high_note_fraction",
        "source_global_chroma_cosine",
        "source_active_chroma_cosine",
        "source_onset_correlation",
        "source_onset_peak_alignment",
        "single_pitch_class_fraction",
        "weighted_mean_velocity",
        "chord_frame_fraction",
        "bass_coverage_fraction",
        "melody_coverage_fraction",
        "mean_active_polyphony",
        "fullness_score",
    ]
    summary = {
        "checkpoint": str(checkpoint),
        "source_manifest": str(manifest),
        "source_column": str(source_col),
        "out_dir": str(out_dir),
        "rows_csv": str(rows_csv),
        "summary_json": str(out_dir / "summary.json"),
        "input_rows": int(len(df)),
        "processed_rows": int(len(rows)),
        "passed_rows": int(sum(1 for row in rows if bool(row.get("passed")))),
        "failed_rows": int(sum(1 for row in rows if not bool(row.get("passed")))),
        "warning_counts": warning_counts,
        "mean_metrics": {key: _mean_metric(rows, key) for key in metric_keys},
        "mean_row_values": {
            "min_section_notes": _mean_row_value(rows, "min_section_notes"),
            "min_section_unique_pitches": _mean_row_value(rows, "min_section_unique_pitches"),
            "min_section_bass_coverage_fraction": _mean_row_value(rows, "min_section_bass_coverage_fraction"),
            "min_section_chord_frame_fraction": _mean_row_value(rows, "min_section_chord_frame_fraction"),
            "min_section_fullness_score": _mean_row_value(rows, "min_section_fullness_score"),
            "mean_section_fullness_score": _mean_row_value(rows, "mean_section_fullness_score"),
            "wav_rms": _mean_row_value(rows, "wav_rms"),
            "wav_peak": _mean_row_value(rows, "wav_peak"),
            "section_diversity_guidance_chunks": _mean_row_value(rows, "section_diversity_guidance_chunks"),
            "section_diversity_reserved_notes": _mean_row_value(rows, "section_diversity_reserved_notes"),
        },
        "min_row_values": {
            "min_section_notes": _min_row_value(rows, "min_section_notes"),
            "min_section_unique_pitches": _min_row_value(rows, "min_section_unique_pitches"),
            "min_section_bass_coverage_fraction": _min_row_value(rows, "min_section_bass_coverage_fraction"),
            "min_section_chord_frame_fraction": _min_row_value(rows, "min_section_chord_frame_fraction"),
            "min_section_fullness_score": _min_row_value(rows, "min_section_fullness_score"),
            "wav_rms": _min_row_value(rows, "wav_rms"),
        },
        "decode": {
            "seconds": float(cfg.seconds),
            "max_frames": int(cfg.max_frames),
            "eval_max_frames": int(cfg.eval_max_frames),
            "frame_hz": float(cfg.frame_hz),
            "onset_threshold": float(cfg.onset_threshold),
            "frame_threshold": float(cfg.frame_threshold),
            "max_notes_per_second": float(cfg.max_notes_per_second),
            "max_simultaneous_notes": int(cfg.max_simultaneous_notes),
            "max_onsets_per_frame": int(cfg.max_onsets_per_frame),
            "max_pitch_fraction": float(cfg.max_pitch_fraction),
            "max_pitch_class_fraction": float(cfg.max_pitch_class_fraction),
            "min_selected_notes": int(cfg.min_selected_notes),
            "min_unique_pitches": int(cfg.min_unique_pitches),
            "section_bass_repair": bool(cfg.section_bass_repair),
            "section_bass_repair_min_coverage": float(cfg.section_bass_repair_min_coverage),
            "section_diversity_repair": bool(cfg.section_diversity_repair),
            "section_diversity_repair_min_unique_pitches": int(cfg.section_diversity_repair_min_unique_pitches),
            "section_diversity_repair_min_chord_frame": float(cfg.section_diversity_repair_min_chord_frame),
            "section_diversity_repair_max_notes": int(cfg.section_diversity_repair_max_notes),
            "source_onset_guidance_weight": float(cfg.source_onset_guidance_weight),
            "density_plan_guidance_weight": float(cfg.density_plan_guidance_weight),
            "event_plan_guidance_weight": float(cfg.event_plan_guidance_weight),
            "pc_onset_plan_guidance_weight": float(cfg.pc_onset_plan_guidance_weight),
            "pc_onset_plan_assign_threshold": float(cfg.pc_onset_plan_assign_threshold),
            "pc_onset_plan_assign_fraction": float(cfg.pc_onset_plan_assign_fraction),
            "pc_onset_plan_assign_window_frames": int(cfg.pc_onset_plan_assign_window_frames),
            "pc_onset_plan_assign_source_weight": float(cfg.pc_onset_plan_assign_source_weight),
            "pc_onset_plan_assign_event_weight": float(cfg.pc_onset_plan_assign_event_weight),
            "pc_onset_plan_assign_distance_penalty": float(cfg.pc_onset_plan_assign_distance_penalty),
            "source_chroma_guidance_weight": float(cfg.source_chroma_guidance_weight),
            "harmonic_plan_guidance_weight": float(cfg.harmonic_plan_guidance_weight),
            "chord_plan_guidance_weight": float(cfg.chord_plan_guidance_weight),
            "bass_plan_guidance_weight": float(cfg.bass_plan_guidance_weight),
            "voicing_plan_guidance_weight": float(cfg.voicing_plan_guidance_weight),
            "section_diversity_guidance_weight": float(cfg.section_diversity_guidance_weight),
            "section_diversity_reserve_fraction": float(cfg.section_diversity_reserve_fraction),
            "section_diversity_reserve_min_note_score": float(cfg.section_diversity_reserve_min_note_score),
            "section_diversity_unique_weight": float(cfg.section_diversity_unique_weight),
            "section_diversity_pc_weight": float(cfg.section_diversity_pc_weight),
            "section_diversity_range_weight": float(cfg.section_diversity_range_weight),
            "section_diversity_onset_weight": float(cfg.section_diversity_onset_weight),
            "section_diversity_section_seconds": float(cfg.section_diversity_section_seconds),
            "source_energy_velocity_weight": float(cfg.source_energy_velocity_weight),
            "density_plan_velocity_weight": float(cfg.density_plan_velocity_weight),
            "chunked": bool(cfg.chunked),
            "chunk_seconds": float(cfg.chunk_seconds),
            "chunk_hop_seconds": float(cfg.chunk_hop_seconds),
            "section_profile": str(cfg.section_profile),
            "section_seconds": float(cfg.section_seconds),
        },
        "rows": rows[:20],
    }
    _write_json(out_dir / "summary.json", summary)
    return summary
