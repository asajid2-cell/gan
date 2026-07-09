from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .piano_arranger_cache import DIFFUSION_SR, extract_source_condition, load_audio_for_cache, midi_to_arrangement


DEFAULT_PIANO_EVAL_REPORT_DIR = Path("saves2/piano_arranger/reports")


@dataclass(frozen=True)
class PianoEvalThresholds:
    min_notes: int = 8
    min_notes_per_second: float = 1.0
    max_notes_per_second: float = 35.0
    max_simultaneous_notes: int = 16
    min_unique_pitches: int = 8
    min_pitch_range: int = 12
    max_same_start_fraction: float = 0.35
    max_long_note_fraction: float = 0.45
    max_single_pitch_fraction: float = 0.25
    max_single_pitch_class_fraction: float = 0.35
    min_velocity_range: int = 12
    min_velocity_std: float = 4.0
    min_register_fraction: float = 0.05
    min_source_global_chroma_cosine: float = 0.20
    min_source_active_chroma_cosine: float = 0.20
    min_source_onset_correlation: float = 0.02
    min_bass_note_fraction: float = 0.05
    min_mid_note_fraction: float = 0.20
    min_high_note_fraction: float = 0.05
    min_weighted_mean_velocity: float = 60.0
    min_chord_frame_fraction: float = 0.05
    min_bass_coverage_fraction: float = 0.15
    min_melody_coverage_fraction: float = 0.10


@dataclass(frozen=True)
class PianoEvalConfig:
    arrangement_json: Path
    report_path: Path | None = None
    label: str = ""
    source_audio: Path | None = None
    source_seconds: float = 0.0
    frame_hz: float = 25.0
    max_frames: int = 0
    thresholds: PianoEvalThresholds = PianoEvalThresholds()


@dataclass(frozen=True)
class PianoSectionReportConfig:
    arrangement_json: Path
    report_path: Path | None = None
    label: str = ""
    section_seconds: float = 8.0


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _note_arrays(notes: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not notes:
        empty_f = np.zeros((0,), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int32)
        return {
            "starts": empty_f,
            "durations": empty_f,
            "ends": empty_f,
            "pitches": empty_i,
            "velocities": empty_i,
        }

    starts = np.asarray([float(n.get("start", 0.0)) for n in notes], dtype=np.float32)
    durations = np.asarray([max(0.0, float(n.get("duration", 0.0))) for n in notes], dtype=np.float32)
    pitches = np.asarray([int(round(float(n.get("pitch", 0)))) for n in notes], dtype=np.int32)
    velocities = np.asarray([int(round(float(n.get("velocity", 0)))) for n in notes], dtype=np.int32)
    return {
        "starts": starts,
        "durations": durations,
        "ends": starts + durations,
        "pitches": pitches,
        "velocities": velocities,
    }


def _max_polyphony(starts: np.ndarray, ends: np.ndarray) -> int:
    if starts.size == 0:
        return 0
    events: List[tuple[float, int]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        events.append((float(start), 1))
        events.append((float(max(start, end)), -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    best = 0
    for _time, delta in events:
        active += int(delta)
        best = max(best, active)
    return int(best)


def _register_fractions(pitches: np.ndarray) -> Dict[str, float]:
    if pitches.size == 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    return {
        "low": float(np.mean(pitches <= 52)),
        "mid": float(np.mean((pitches >= 53) & (pitches <= 76))),
        "high": float(np.mean(pitches >= 77)),
    }


def _role_fullness_metrics(
    starts: np.ndarray,
    ends: np.ndarray,
    pitches: np.ndarray,
    velocities: np.ndarray,
    *,
    duration: float,
    frame_hz: float = 25.0,
) -> Dict[str, float]:
    if pitches.size == 0:
        return {
            "bass_note_fraction": 0.0,
            "mid_note_fraction": 0.0,
            "high_note_fraction": 0.0,
            "weighted_mean_velocity": 0.0,
            "chord_frame_fraction": 0.0,
            "bass_coverage_fraction": 0.0,
            "melody_coverage_fraction": 0.0,
            "mean_active_polyphony": 0.0,
            "fullness_score": 0.0,
        }
    n_frames = max(1, int(np.ceil(max(1e-6, float(duration)) * float(frame_hz))))
    poly = np.zeros((n_frames,), dtype=np.float32)
    bass = np.zeros((n_frames,), dtype=np.float32)
    high = np.zeros((n_frames,), dtype=np.float32)
    for start, end, pitch in zip(starts.tolist(), ends.tolist(), pitches.tolist()):
        lo = max(0, min(n_frames - 1, int(np.floor(float(start) * float(frame_hz)))))
        hi = max(lo + 1, min(n_frames, int(np.ceil(float(end) * float(frame_hz)))))
        poly[lo:hi] += 1.0
        if int(pitch) <= 52:
            bass[lo:hi] = 1.0
        if int(pitch) >= 77:
            high[lo:hi] = 1.0
    active = poly > 0.0
    weighted_mean_velocity = float(np.average(velocities, weights=np.maximum(1e-3, velocities))) if velocities.size else 0.0
    bass_fraction = float(np.mean(pitches <= 52))
    mid_fraction = float(np.mean((pitches >= 53) & (pitches <= 76)))
    high_fraction = float(np.mean(pitches >= 77))
    chord_frame_fraction = float(np.mean(poly >= 3.0))
    bass_coverage = float(np.mean(bass[active])) if np.any(active) else 0.0
    melody_coverage = float(np.mean(high[active])) if np.any(active) else 0.0
    mean_active_polyphony = float(np.mean(poly[active])) if np.any(active) else 0.0
    fullness_score = float(
        np.clip(bass_coverage / 0.35, 0.0, 1.0) * 0.20
        + np.clip(chord_frame_fraction / 0.25, 0.0, 1.0) * 0.25
        + np.clip(melody_coverage / 0.30, 0.0, 1.0) * 0.20
        + np.clip(mean_active_polyphony / 4.0, 0.0, 1.0) * 0.20
        + np.clip((weighted_mean_velocity - 45.0) / 35.0, 0.0, 1.0) * 0.15
    )
    return {
        "bass_note_fraction": bass_fraction,
        "mid_note_fraction": mid_fraction,
        "high_note_fraction": high_fraction,
        "weighted_mean_velocity": weighted_mean_velocity,
        "chord_frame_fraction": chord_frame_fraction,
        "bass_coverage_fraction": bass_coverage,
        "melody_coverage_fraction": melody_coverage,
        "mean_active_polyphony": mean_active_polyphony,
        "fullness_score": fullness_score,
    }


def _dominant_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _unique, counts = np.unique(values, return_counts=True)
    return float(np.max(counts) / max(1, values.size))


def _same_start_fraction(starts: np.ndarray) -> float:
    if starts.size == 0:
        return 0.0
    rounded = np.round(starts.astype(np.float64), 3)
    return _dominant_fraction(rounded)


def _warn_if(condition: bool, warnings: List[str], code: str, message: str) -> None:
    if condition:
        warnings.append(f"{code}: {message}")


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(aa.size, bb.size)
    if n == 0:
        return 0.0
    aa = aa[:n]
    bb = bb[:n]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    n = min(aa.size, bb.size)
    if n < 2:
        return 0.0
    aa = aa[:n] - float(np.mean(aa[:n]))
    bb = bb[:n] - float(np.mean(bb[:n]))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _arrangement_features(
    notes: Sequence[Dict[str, Any]],
    *,
    frame_hz: float,
    max_frames: int,
) -> Dict[str, np.ndarray]:
    n_frames = max(1, int(max_frames))
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    onset = np.zeros((n_frames,), dtype=np.float32)
    frame_energy = np.zeros((n_frames,), dtype=np.float32)
    for note in notes:
        start_time = max(0.0, float(note.get("start", 0.0)))
        duration = max(0.0, float(note.get("duration", 0.0)))
        pitch = int(round(float(note.get("pitch", 0))))
        velocity = float(max(1, min(127, int(round(float(note.get("velocity", 64))))))) / 127.0
        start = max(0, min(n_frames - 1, int(round(start_time * float(frame_hz)))))
        end = max(start + 1, int(np.ceil((start_time + duration) * float(frame_hz))))
        end = max(start + 1, min(n_frames, end))
        pc = int(pitch % 12)
        onset[start] += velocity
        chroma[pc, start:end] += velocity
        frame_energy[start:end] += velocity
    if float(np.max(chroma)) > 1e-8:
        chroma = chroma / float(np.max(chroma))
    if float(np.max(onset)) > 1e-8:
        onset = onset / float(np.max(onset))
    if float(np.max(frame_energy)) > 1e-8:
        frame_energy = frame_energy / float(np.max(frame_energy))
    return {"chroma": chroma, "onset": onset, "frame_energy": frame_energy}


def _match_count_1d(pred: Sequence[int], target: Sequence[int], *, tolerance: int = 1) -> int:
    remaining = [int(x) for x in target]
    matched = 0
    for raw in sorted(int(x) for x in pred):
        best_idx = None
        best_dist = int(tolerance) + 1
        for i, target_idx in enumerate(remaining):
            dist = abs(int(raw) - int(target_idx))
            if dist <= int(tolerance) and dist < best_dist:
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            matched += 1
            remaining.pop(best_idx)
    return int(matched)


def _match_count_pitch_class(
    pred: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
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


def _prf(matches: int, pred_count: int, target_count: int) -> Dict[str, float]:
    precision = float(matches / pred_count) if int(pred_count) > 0 else 0.0
    recall = float(matches / target_count) if int(target_count) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall > 1e-8 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _source_alignment_metrics(
    arrangement: Dict[str, Any],
    *,
    source_audio: Path,
    source_seconds: float,
    frame_hz: float,
    max_frames: int,
) -> Dict[str, Any]:
    notes = list(arrangement.get("notes", []))
    declared_duration = max(0.0, float(arrangement.get("duration", 0.0)))
    seconds = float(source_seconds) if float(source_seconds) > 0 else declared_duration
    if seconds <= 0:
        seconds = 30.0
    frames = int(max_frames) if int(max_frames) > 0 else max(1, int(np.ceil(seconds * float(frame_hz))))
    y = load_audio_for_cache(Path(source_audio), seconds=seconds, sr=DIFFUSION_SR)
    source = extract_source_condition(y, sr=DIFFUSION_SR, frame_hz=float(frame_hz), max_frames=frames)
    src_chroma = np.asarray(source[:12], dtype=np.float32)
    src_onset = np.asarray(source[12], dtype=np.float32)
    arr = _arrangement_features(notes, frame_hz=float(frame_hz), max_frames=frames)
    arr_chroma = arr["chroma"]
    arr_onset = arr["onset"]

    active = np.where(np.sum(arr_chroma, axis=0) > 1e-5)[0]
    if active.size:
        frame_cosines = [_safe_cosine(src_chroma[:, i], arr_chroma[:, i]) for i in active.tolist()]
        active_chroma_cosine = float(np.mean(frame_cosines))
    else:
        active_chroma_cosine = 0.0

    source_peaks = np.where(src_onset >= max(0.35, float(np.percentile(src_onset, 75))))[0]
    arr_peaks = np.where(arr_onset > 1e-5)[0]
    if arr_peaks.size and source_peaks.size:
        aligned = 0
        source_peak_set = set(int(x) for x in source_peaks.tolist())
        for idx in arr_peaks.tolist():
            i = int(idx)
            if i in source_peak_set or i - 1 in source_peak_set or i + 1 in source_peak_set:
                aligned += 1
        onset_peak_alignment = float(aligned / max(1, arr_peaks.size))
    else:
        onset_peak_alignment = 0.0

    return {
        "source_audio": str(source_audio),
        "source_eval_seconds": float(len(y) / float(DIFFUSION_SR)),
        "source_eval_frames": int(frames),
        "source_global_chroma_cosine": _safe_cosine(np.sum(src_chroma, axis=1), np.sum(arr_chroma, axis=1)),
        "source_active_chroma_cosine": active_chroma_cosine,
        "source_onset_correlation": _safe_corr(src_onset, arr_onset),
        "source_onset_peak_alignment": onset_peak_alignment,
        "source_onset_peak_count": int(source_peaks.size),
        "arrangement_onset_frame_count": int(arr_peaks.size),
    }


def _notes_as_dicts(notes: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {
            "start": float(note.start),
            "duration": float(note.duration),
            "pitch": int(note.pitch),
            "velocity": int(note.velocity),
        }
        for note in notes
    ]


def target_midi_alignment_metrics(
    arrangement: Dict[str, Any],
    *,
    target_midi: Path,
    target_seconds: float,
    frame_hz: float,
    max_frames: int,
) -> Dict[str, Any]:
    seconds = float(target_seconds) if float(target_seconds) > 0 else float(arrangement.get("duration", 0.0) or 0.0)
    if seconds <= 0:
        seconds = 30.0
    frames = int(max_frames) if int(max_frames) > 0 else max(1, int(np.ceil(seconds * float(frame_hz))))
    target = midi_to_arrangement(Path(target_midi), seconds=seconds)
    target_notes = _notes_as_dicts(target.notes)
    pred_features = _arrangement_features(arrangement.get("notes", []), frame_hz=float(frame_hz), max_frames=frames)
    target_features = _arrangement_features(target_notes, frame_hz=float(frame_hz), max_frames=frames)
    pred_chroma = pred_features["chroma"]
    target_chroma = target_features["chroma"]
    active = np.where(np.sum(pred_chroma, axis=0) > 1e-5)[0]
    if active.size:
        frame_cosines = [_safe_cosine(target_chroma[:, i], pred_chroma[:, i]) for i in active.tolist()]
        active_chroma_cosine = float(np.mean(frame_cosines))
    else:
        active_chroma_cosine = 0.0
    target_pitches = np.asarray([int(note["pitch"]) for note in target_notes], dtype=np.int32)
    pred_notes = list(arrangement.get("notes", []))
    pred_onset_frames = sorted(
        {max(0, min(frames - 1, int(round(float(note.get("start", 0.0)) * float(frame_hz))))) for note in pred_notes}
    )
    target_onset_frames = sorted(
        {max(0, min(frames - 1, int(round(float(note.get("start", 0.0)) * float(frame_hz))))) for note in target_notes}
    )
    onset_matches = _match_count_1d(pred_onset_frames, target_onset_frames, tolerance=1)
    onset_prf = _prf(onset_matches, len(pred_onset_frames), len(target_onset_frames))

    pred_pc_onsets = [
        (
            int(round(float(note.get("pitch", 0)))) % 12,
            max(0, min(frames - 1, int(round(float(note.get("start", 0.0)) * float(frame_hz))))),
        )
        for note in pred_notes
    ]
    target_pc_onsets = [
        (
            int(round(float(note.get("pitch", 0)))) % 12,
            max(0, min(frames - 1, int(round(float(note.get("start", 0.0)) * float(frame_hz))))),
        )
        for note in target_notes
    ]
    pc_matches = _match_count_pitch_class(pred_pc_onsets, target_pc_onsets, tolerance=1)
    pc_prf = _prf(pc_matches, len(pred_pc_onsets), len(target_pc_onsets))
    return {
        "target_midi": str(target_midi),
        "target_eval_seconds": float(seconds),
        "target_eval_frames": int(frames),
        "target_note_count": int(len(target_notes)),
        "target_unique_pitches": int(np.unique(target_pitches).size) if target_pitches.size else 0,
        "target_global_chroma_cosine": _safe_cosine(np.sum(target_chroma, axis=1), np.sum(pred_chroma, axis=1)),
        "target_active_chroma_cosine": active_chroma_cosine,
        "target_onset_correlation": _safe_corr(target_features["onset"], pred_features["onset"]),
        "target_onset_frame_precision": onset_prf["precision"],
        "target_onset_frame_recall": onset_prf["recall"],
        "target_onset_frame_f1": onset_prf["f1"],
        "target_pitch_class_onset_precision": pc_prf["precision"],
        "target_pitch_class_onset_recall": pc_prf["recall"],
        "target_pitch_class_onset_f1": pc_prf["f1"],
        "target_note_count_ratio": float(len(pred_notes) / max(1, len(target_notes))),
    }


def target_midi_alignment_warnings(
    metrics: Dict[str, Any],
    *,
    min_target_global_chroma_cosine: float = 0.20,
    min_target_active_chroma_cosine: float = 0.20,
    min_target_onset_correlation: float = 0.02,
    min_target_onset_frame_f1: float = 0.0,
    min_target_pitch_class_onset_f1: float = 0.0,
    min_target_note_count_ratio: float = 0.0,
    max_target_note_count_ratio: float = 0.0,
) -> List[str]:
    warnings: List[str] = []
    if float(metrics.get("target_global_chroma_cosine", 0.0)) < float(min_target_global_chroma_cosine):
        warnings.append(
            "target_harmony_mismatch: predicted global pitch-class distribution is weakly aligned with target MIDI"
        )
    if float(metrics.get("target_active_chroma_cosine", 0.0)) < float(min_target_active_chroma_cosine):
        warnings.append(
            "target_active_harmony_mismatch: active predicted frames are weakly aligned with target MIDI chroma"
        )
    if float(metrics.get("target_onset_correlation", 0.0)) < float(min_target_onset_correlation):
        warnings.append("target_rhythm_mismatch: predicted onset curve is weakly aligned with target MIDI")
    if float(min_target_onset_frame_f1) > 0.0 and float(metrics.get("target_onset_frame_f1", 0.0)) < float(min_target_onset_frame_f1):
        warnings.append("target_onset_f1_mismatch: predicted onset frames weakly match target MIDI events")
    if float(min_target_pitch_class_onset_f1) > 0.0 and float(metrics.get("target_pitch_class_onset_f1", 0.0)) < float(min_target_pitch_class_onset_f1):
        warnings.append("target_pitch_class_onset_f1_mismatch: predicted pitch-class onsets weakly match target MIDI events")
    note_count_ratio = float(metrics.get("target_note_count_ratio", 0.0))
    if float(min_target_note_count_ratio) > 0.0 and note_count_ratio < float(min_target_note_count_ratio):
        warnings.append("target_note_count_underfill: predicted note count is too low relative to target MIDI")
    if float(max_target_note_count_ratio) > 0.0 and note_count_ratio > float(max_target_note_count_ratio):
        warnings.append("target_note_count_overfill: predicted note count is too high relative to target MIDI")
    return warnings


def evaluate_arrangement_dict(
    arrangement: Dict[str, Any],
    *,
    label: str = "",
    thresholds: PianoEvalThresholds = PianoEvalThresholds(),
    source_audio: Path | None = None,
    source_seconds: float = 0.0,
    frame_hz: float = 25.0,
    max_frames: int = 0,
) -> Dict[str, Any]:
    notes = list(arrangement.get("notes", []))
    arr = _note_arrays(notes)
    starts = arr["starts"]
    durations = arr["durations"]
    ends = arr["ends"]
    pitches = arr["pitches"]
    velocities = arr["velocities"]

    declared_duration = max(0.0, float(arrangement.get("duration", 0.0)))
    observed_duration = float(np.max(ends)) if ends.size else 0.0
    duration = max(declared_duration, observed_duration, 1e-6)
    notes_per_second = float(len(notes) / duration)
    pitch_min = int(np.min(pitches)) if pitches.size else 0
    pitch_max = int(np.max(pitches)) if pitches.size else 0
    pitch_range = int(pitch_max - pitch_min) if pitches.size else 0
    pitch_classes = np.mod(pitches, 12) if pitches.size else pitches
    register = _register_fractions(pitches)
    long_note_cutoff = max(1.5, duration * 0.45)
    long_note_fraction = float(np.mean(durations >= long_note_cutoff)) if durations.size else 0.0
    velocity_range = int(np.max(velocities) - np.min(velocities)) if velocities.size else 0
    velocity_std = float(np.std(velocities)) if velocities.size else 0.0
    role_metrics = _role_fullness_metrics(
        starts,
        ends,
        pitches,
        velocities,
        duration=float(duration),
        frame_hz=float(frame_hz),
    )

    metrics: Dict[str, Any] = {
        "label": label,
        "note_count": int(len(notes)),
        "duration_seconds": float(duration),
        "declared_duration_seconds": float(declared_duration),
        "observed_duration_seconds": float(observed_duration),
        "notes_per_second": notes_per_second,
        "max_simultaneous_notes": _max_polyphony(starts, ends),
        "unique_pitches": int(np.unique(pitches).size) if pitches.size else 0,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_range": pitch_range,
        "register_fraction_low": register["low"],
        "register_fraction_mid": register["mid"],
        "register_fraction_high": register["high"],
        "same_start_fraction": _same_start_fraction(starts),
        "long_note_fraction": long_note_fraction,
        "single_pitch_fraction": _dominant_fraction(pitches),
        "single_pitch_class_fraction": _dominant_fraction(pitch_classes),
        "velocity_min": int(np.min(velocities)) if velocities.size else 0,
        "velocity_max": int(np.max(velocities)) if velocities.size else 0,
        "velocity_range": velocity_range,
        "velocity_std": velocity_std,
        "sustain_event_count": int(len(arrangement.get("sustain", []))),
        **role_metrics,
    }

    warnings: List[str] = []
    th = thresholds
    _warn_if(metrics["note_count"] < th.min_notes, warnings, "empty_or_too_sparse", "too few notes")
    _warn_if(
        metrics["notes_per_second"] < th.min_notes_per_second,
        warnings,
        "too_sparse",
        "note density is below the minimum gate",
    )
    _warn_if(
        metrics["notes_per_second"] > th.max_notes_per_second,
        warnings,
        "overdense",
        "note density is above the maximum gate",
    )
    _warn_if(
        metrics["max_simultaneous_notes"] > th.max_simultaneous_notes,
        warnings,
        "overstacked",
        "too many simultaneous notes",
    )
    _warn_if(metrics["unique_pitches"] < th.min_unique_pitches, warnings, "pitch_collapse", "too few unique pitches")
    _warn_if(metrics["pitch_range"] < th.min_pitch_range, warnings, "register_collapse", "pitch range is too narrow")
    _warn_if(
        metrics["same_start_fraction"] > th.max_same_start_fraction,
        warnings,
        "rhythm_collapse",
        "too many notes share one start time",
    )
    _warn_if(
        metrics["long_note_fraction"] > th.max_long_note_fraction,
        warnings,
        "drone_collapse",
        "too many notes are long drones",
    )
    _warn_if(
        metrics["single_pitch_fraction"] > th.max_single_pitch_fraction,
        warnings,
        "single_pitch_collapse",
        "one pitch dominates the arrangement",
    )
    _warn_if(
        metrics["single_pitch_class_fraction"] > th.max_single_pitch_class_fraction,
        warnings,
        "harmony_collapse",
        "one pitch class dominates the arrangement",
    )
    _warn_if(metrics["velocity_range"] < th.min_velocity_range, warnings, "velocity_collapse", "velocity range is too small")
    _warn_if(metrics["velocity_std"] < th.min_velocity_std, warnings, "velocity_flat", "velocity variation is too small")
    _warn_if(
        any(metrics[f"register_fraction_{name}"] < th.min_register_fraction for name in ("low", "mid", "high")),
        warnings,
        "register_underuse",
        "one or more keyboard registers are barely used",
    )
    _warn_if(
        metrics["bass_note_fraction"] < th.min_bass_note_fraction,
        warnings,
        "bass_underuse",
        "bass register does not provide enough foundation",
    )
    _warn_if(
        metrics["mid_note_fraction"] < th.min_mid_note_fraction,
        warnings,
        "mid_harmony_underuse",
        "middle register harmony/body is underused",
    )
    _warn_if(
        metrics["high_note_fraction"] < th.min_high_note_fraction,
        warnings,
        "melody_register_underuse",
        "high register melody/topline is underused",
    )
    _warn_if(
        metrics["weighted_mean_velocity"] < th.min_weighted_mean_velocity,
        warnings,
        "weak_velocity_weight",
        "velocity distribution is too light for a full arrangement",
    )
    _warn_if(
        metrics["chord_frame_fraction"] < th.min_chord_frame_fraction,
        warnings,
        "thin_harmony_texture",
        "too few active frames contain chordal texture",
    )
    _warn_if(
        metrics["bass_coverage_fraction"] < th.min_bass_coverage_fraction,
        warnings,
        "bass_coverage_gap",
        "bass foundation is absent from too many active frames",
    )
    _warn_if(
        metrics["melody_coverage_fraction"] < th.min_melody_coverage_fraction,
        warnings,
        "melody_coverage_gap",
        "high-register/topline activity is absent from too many active frames",
    )
    if source_audio is not None:
        source_metrics = _source_alignment_metrics(
            arrangement,
            source_audio=Path(source_audio),
            source_seconds=float(source_seconds),
            frame_hz=float(frame_hz),
            max_frames=int(max_frames),
        )
        metrics.update(source_metrics)
        _warn_if(
            metrics["source_global_chroma_cosine"] < th.min_source_global_chroma_cosine,
            warnings,
            "source_harmony_mismatch",
            "global pitch-class distribution is weakly aligned with the source",
        )
        _warn_if(
            metrics["source_active_chroma_cosine"] < th.min_source_active_chroma_cosine,
            warnings,
            "source_active_harmony_mismatch",
            "active arrangement frames are weakly aligned with source chroma",
        )
        _warn_if(
            metrics["source_onset_correlation"] < th.min_source_onset_correlation,
            warnings,
            "source_rhythm_mismatch",
            "arrangement onset curve is weakly aligned with source onset strength",
        )

    return {
        "passed": len(warnings) == 0,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": asdict(thresholds),
    }


def evaluate_arrangement_file(cfg: PianoEvalConfig) -> Dict[str, Any]:
    arrangement_json = Path(cfg.arrangement_json)
    arrangement = _load_json(arrangement_json)
    label = cfg.label or arrangement_json.stem
    report = evaluate_arrangement_dict(
        arrangement,
        label=label,
        thresholds=cfg.thresholds,
        source_audio=cfg.source_audio,
        source_seconds=float(cfg.source_seconds),
        frame_hz=float(cfg.frame_hz),
        max_frames=int(cfg.max_frames),
    )
    report["arrangement_json"] = str(arrangement_json)

    report_path = cfg.report_path
    if report_path is None:
        report_path = DEFAULT_PIANO_EVAL_REPORT_DIR / f"{arrangement_json.stem}_eval.json"
    _write_json(Path(report_path), report)
    report["report_path"] = str(report_path)
    return report


def section_report_arrangement_dict(
    arrangement: Dict[str, Any],
    *,
    label: str = "",
    section_seconds: float = 8.0,
) -> Dict[str, Any]:
    notes = list(arrangement.get("notes", []))
    arr = _note_arrays(notes)
    duration = max(
        float(arrangement.get("duration", 0.0) or 0.0),
        float(np.max(arr["ends"])) if arr["ends"].size else 0.0,
        1e-6,
    )
    section_len = max(0.5, float(section_seconds))
    sections: List[Dict[str, Any]] = []
    starts = np.arange(0.0, duration, section_len).tolist()
    min_tail_seconds = min(0.5, 0.25 * section_len)
    if len(starts) > 1 and duration - float(starts[-1]) < min_tail_seconds:
        starts = starts[:-1]
    for section_idx, start in enumerate(starts):
        end = min(duration, float(start) + section_len)
        if section_idx == len(starts) - 1:
            end = duration
        section_notes = [
            note
            for note in notes
            if float(note.get("start", 0.0)) < end and float(note.get("start", 0.0)) + float(note.get("duration", 0.0)) > float(start)
        ]
        sec = _note_arrays(section_notes)
        pitches = sec["pitches"]
        velocities = sec["velocities"]
        register = _register_fractions(pitches)
        sec_duration = max(1e-6, end - float(start))
        local_starts = np.maximum(0.0, sec["starts"] - float(start)).astype(np.float32)
        local_ends = np.maximum(local_starts, sec["ends"] - float(start)).astype(np.float32)
        role = _role_fullness_metrics(
            local_starts,
            local_ends,
            pitches,
            velocities,
            duration=sec_duration,
            frame_hz=25.0,
        )
        sections.append(
            {
                "section": int(section_idx),
                "start": float(start),
                "end": float(end),
                "note_count": int(len(section_notes)),
                "notes_per_second": float(len(section_notes) / sec_duration),
                "max_simultaneous_notes": _max_polyphony(sec["starts"], sec["ends"]),
                "unique_pitches": int(np.unique(pitches).size) if pitches.size else 0,
                "pitch_min": int(np.min(pitches)) if pitches.size else 0,
                "pitch_max": int(np.max(pitches)) if pitches.size else 0,
                "pitch_range": int(np.max(pitches) - np.min(pitches)) if pitches.size else 0,
                "register_fraction_low": register["low"],
                "register_fraction_mid": register["mid"],
                "register_fraction_high": register["high"],
                "same_start_fraction": _same_start_fraction(sec["starts"]),
                "velocity_min": int(np.min(velocities)) if velocities.size else 0,
                "velocity_max": int(np.max(velocities)) if velocities.size else 0,
                "velocity_range": int(np.max(velocities) - np.min(velocities)) if velocities.size else 0,
                "velocity_std": float(np.std(velocities)) if velocities.size else 0.0,
                **role,
            }
        )

    note_counts = [int(s["note_count"]) for s in sections]
    unique_pitches = [int(s["unique_pitches"]) for s in sections]
    high_regs = [float(s["register_fraction_high"]) for s in sections]
    nps = [float(s["notes_per_second"]) for s in sections]
    fullness_scores = [float(s["fullness_score"]) for s in sections]
    bass_coverage = [float(s["bass_coverage_fraction"]) for s in sections]
    chord_frames = [float(s["chord_frame_fraction"]) for s in sections]
    summary = {
        "label": str(label),
        "duration_seconds": float(duration),
        "section_seconds": float(section_len),
        "section_count": int(len(sections)),
        "total_notes": int(len(notes)),
        "min_section_notes": int(min(note_counts)) if note_counts else 0,
        "max_section_notes": int(max(note_counts)) if note_counts else 0,
        "mean_section_notes": float(np.mean(note_counts)) if note_counts else 0.0,
        "min_section_unique_pitches": int(min(unique_pitches)) if unique_pitches else 0,
        "mean_section_unique_pitches": float(np.mean(unique_pitches)) if unique_pitches else 0.0,
        "min_section_notes_per_second": float(min(nps)) if nps else 0.0,
        "max_section_notes_per_second": float(max(nps)) if nps else 0.0,
        "min_section_high_register_fraction": float(min(high_regs)) if high_regs else 0.0,
        "min_section_bass_coverage_fraction": float(min(bass_coverage)) if bass_coverage else 0.0,
        "min_section_chord_frame_fraction": float(min(chord_frames)) if chord_frames else 0.0,
        "min_section_fullness_score": float(min(fullness_scores)) if fullness_scores else 0.0,
        "mean_section_fullness_score": float(np.mean(fullness_scores)) if fullness_scores else 0.0,
    }
    warnings: List[str] = []
    for section in sections:
        if int(section["note_count"]) == 0:
            warnings.append(f"empty_section:{section['section']}")
        if float(section["register_fraction_high"]) < 0.01:
            warnings.append(f"section_high_register_absent:{section['section']}")
        if int(section["note_count"]) >= 8 and int(section["unique_pitches"]) < 8:
            warnings.append(f"section_pitch_underuse:{section['section']}")
        if int(section["note_count"]) >= 8 and float(section["bass_coverage_fraction"]) < 0.05:
            warnings.append(f"section_bass_absent:{section['section']}")
        if int(section["note_count"]) >= 8 and float(section["chord_frame_fraction"]) < 0.05:
            warnings.append(f"section_thin_texture:{section['section']}")
    return {"summary": summary, "warnings": warnings, "sections": sections}


def section_report_arrangement_file(cfg: PianoSectionReportConfig) -> Dict[str, Any]:
    arrangement_json = Path(cfg.arrangement_json)
    arrangement = _load_json(arrangement_json)
    label = cfg.label or arrangement_json.stem
    report = section_report_arrangement_dict(arrangement, label=label, section_seconds=float(cfg.section_seconds))
    report["arrangement_json"] = str(arrangement_json)
    report_path = cfg.report_path
    if report_path is None:
        report_path = DEFAULT_PIANO_EVAL_REPORT_DIR / f"{arrangement_json.stem}_sections.json"
    _write_json(Path(report_path), report)
    report["report_path"] = str(report_path)
    return report


__all__ = [
    "DEFAULT_PIANO_EVAL_REPORT_DIR",
    "PianoEvalConfig",
    "PianoSectionReportConfig",
    "PianoEvalThresholds",
    "evaluate_arrangement_dict",
    "evaluate_arrangement_file",
    "section_report_arrangement_dict",
    "section_report_arrangement_file",
]
