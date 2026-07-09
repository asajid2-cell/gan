from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd

from .lab3_diffusion_data import DIFFUSION_SR, pad_or_trim
from .piano_arranger_baseline import HeuristicPianoConfig, arrange_audio_heuristic
from .piano_arranger_data import DEFAULT_MIDI_MANIFEST, DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST, DEFAULT_PIANO_MANIFEST
from .piano_arranger_render import PianoArrangement, PianoNote, SustainEvent, arrangement_to_source_preview_audio


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIANO_CACHE_DIR = REPO_ROOT / "saves2" / "piano_arranger" / "cache" / "bootstrap_pseudo_v1"
PIANO_MIN_MIDI = 21
PIANO_MAX_MIDI = 108
PIANO_N_KEYS = 88
SOURCE_FEATURE_NAMES = [
    *[f"chroma_{i:02d}" for i in range(12)],
    "onset",
    "rms",
    "beat",
    "spectral_centroid",
    "zero_crossing_rate",
]


@dataclass(frozen=True)
class PianoCacheConfig:
    manifest: Path = DEFAULT_PIANO_MANIFEST
    cache_dir: Path = DEFAULT_PIANO_CACHE_DIR
    seconds: float = 8.0
    max_frames: int = 256
    frame_hz: float = 25.0
    max_tracks: int = 0
    seed: int = 328
    fullness: float = 0.85
    melody_focus: float = 0.80
    rhythmic_drive: float = 0.65
    harmonic_adventure: float = 0.25
    register_width: float = 0.85
    pedal_amount: float = 0.70


@dataclass(frozen=True)
class MidiPianoCacheConfig:
    manifest: Path = DEFAULT_MIDI_MANIFEST
    cache_dir: Path = REPO_ROOT / "saves2" / "piano_arranger" / "cache" / "midi_targets_v1"
    seconds: float = 8.0
    max_frames: int = 256
    frame_hz: float = 25.0
    max_tracks: int = 0
    min_notes: int = 8
    preview_sample_rate: int = DIFFUSION_SR
    source_preview_mode: str = "piano"


@dataclass(frozen=True)
class PairedAudioMidiCacheConfig:
    manifest: Path = DEFAULT_PAIRED_AUDIO_MIDI_MANIFEST
    cache_dir: Path = REPO_ROOT / "saves2" / "piano_arranger" / "cache" / "paired_audio_midi_v1"
    seconds: float = 8.0
    max_frames: int = 256
    frame_hz: float = 25.0
    max_tracks: int = 0
    min_notes: int = 8


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _normalize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)
    lo = float(np.percentile(x, 5))
    hi = float(np.percentile(x, 95))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _resample_frames(arr: np.ndarray, target: int, axis: int = -1) -> np.ndarray:
    return pad_or_trim(np.asarray(arr, dtype=np.float32), int(target), axis=axis, pad_val=0.0).astype(np.float32)


def _resolve_manifest_path(raw: Any, manifest: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return Path(manifest).parent / path


def load_audio_for_cache(path: Path, seconds: float, sr: int = DIFFUSION_SR) -> np.ndarray:
    y, _ = librosa.load(
        str(path),
        sr=int(sr),
        mono=True,
        duration=float(seconds) if float(seconds) > 0 else None,
        dtype=np.float32,
        res_type="soxr_hq",
    )
    if y.size < int(0.25 * sr):
        raise ValueError(f"Audio is too short for cache: {path}")
    return librosa.util.normalize(y).astype(np.float32)


def extract_source_condition(
    y: np.ndarray,
    *,
    sr: int = DIFFUSION_SR,
    frame_hz: float = 25.0,
    max_frames: int = 256,
) -> np.ndarray:
    hop = max(128, int(round(float(sr) / max(1.0, float(frame_hz)))))
    chroma = librosa.feature.chroma_cqt(y=y, sr=int(sr), hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=int(sr), hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=int(sr), hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
    tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=int(sr), hop_length=hop)
    n_frames = int(max(chroma.shape[1], onset.shape[0], rms.shape[0], centroid.shape[0], zcr.shape[0]))
    beat = np.zeros((n_frames,), dtype=np.float32)
    for bf in np.asarray(beat_frames, dtype=np.int64).reshape(-1).tolist():
        if 0 <= int(bf) < n_frames:
            beat[int(bf)] = 1.0

    features = [
        _resample_frames(chroma, int(max_frames), axis=1),
        _resample_frames(_normalize_feature(onset)[None, :], int(max_frames), axis=1),
        _resample_frames(_normalize_feature(rms)[None, :], int(max_frames), axis=1),
        _resample_frames(beat[None, :], int(max_frames), axis=1),
        _resample_frames(_normalize_feature(centroid)[None, :], int(max_frames), axis=1),
        _resample_frames(_normalize_feature(zcr)[None, :], int(max_frames), axis=1),
    ]
    out = np.concatenate(features, axis=0).astype(np.float32)
    if out.shape[0] != len(SOURCE_FEATURE_NAMES):
        raise RuntimeError(f"Unexpected source feature channels: {out.shape}")
    return out


def piano_roll_from_arrangement(
    arrangement: PianoArrangement,
    *,
    max_frames: int,
    frame_hz: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    onset = np.zeros((PIANO_N_KEYS, int(max_frames)), dtype=np.float32)
    frame = np.zeros_like(onset)
    velocity = np.zeros_like(onset)
    pedal = np.zeros((int(max_frames),), dtype=np.float32)
    seconds_to_frame = float(frame_hz)

    for note in arrangement.notes:
        pitch = int(round(note.pitch))
        if pitch < PIANO_MIN_MIDI or pitch > PIANO_MAX_MIDI:
            continue
        key = pitch - PIANO_MIN_MIDI
        start = max(0, min(int(max_frames) - 1, int(round(float(note.start) * seconds_to_frame))))
        end = max(start + 1, int(np.ceil((float(note.start) + float(note.duration)) * seconds_to_frame)))
        end = max(start + 1, min(int(max_frames), end))
        vel = float(max(1, min(127, int(round(note.velocity))))) / 127.0
        onset[key, start] = max(onset[key, start], 1.0)
        frame[key, start:end] = 1.0
        velocity[key, start:end] = np.maximum(velocity[key, start:end], vel)

    sustain_events = sorted(arrangement.sustain, key=lambda ev: float(ev.time))
    current = 0.0
    cursor = 0
    for ev in sustain_events:
        event_frame = max(0, min(int(max_frames), int(round(float(ev.time) * seconds_to_frame))))
        if event_frame > cursor:
            pedal[cursor:event_frame] = current
        current = float(max(0, min(127, int(ev.value)))) / 127.0
        cursor = event_frame
    if cursor < int(max_frames):
        pedal[cursor:] = current
    return onset, frame, velocity, pedal


def hierarchy_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    density = np.stack(
        [
            np.clip(np.sum(onset, axis=0) / 8.0, 0.0, 1.0),
            np.clip(np.sum(frame, axis=0) / 16.0, 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)
    key_pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    register = np.stack(
        [
            np.clip(np.max(frame[key_pitches <= 52], axis=0), 0.0, 1.0),
            np.clip(np.max(frame[(key_pitches >= 53) & (key_pitches <= 76)], axis=0), 0.0, 1.0),
            np.clip(np.max(frame[key_pitches >= 77], axis=0), 0.0, 1.0),
        ],
        axis=0,
    ).astype(np.float32)
    return density, register


def musical_plan_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive chord, bass, and voicing planning targets from an 88-key piano roll."""

    del onset
    frame = np.asarray(frame, dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    pitch_classes = np.remainder(pitches, 12)
    n_frames = int(frame.shape[1])
    chord = np.zeros((13, n_frames), dtype=np.float32)
    bass = np.zeros((13, n_frames), dtype=np.float32)
    voicing = np.zeros((4, n_frames), dtype=np.float32)
    for t in range(n_frames):
        active_idx = np.flatnonzero(frame[:, t] > 0.1)
        if active_idx.size == 0:
            chord[12, t] = 1.0
            bass[12, t] = 1.0
            continue
        active_pitches = pitches[active_idx]
        active_pc = np.unique(pitch_classes[active_idx])
        chord[active_pc, t] = 1.0
        bass[int(active_pitches.min() % 12), t] = 1.0
        span = float(active_pitches.max() - active_pitches.min()) if active_pitches.size > 1 else 0.0
        voicing[:, t] = np.array(
            [
                np.clip(float(active_pitches.size) / 8.0, 0.0, 1.0),
                np.clip(span / 48.0, 0.0, 1.0),
                np.clip(float(active_pitches.mean() - PIANO_MIN_MIDI) / float(PIANO_MAX_MIDI - PIANO_MIN_MIDI), 0.0, 1.0),
                np.clip(float(np.mean(active_pitches >= 77)), 0.0, 1.0),
            ],
            dtype=np.float32,
        )
    return chord, bass, voicing


def event_plan_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Derive onset/off/change/chord-change rhythm targets from an 88-key piano roll."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    frame_prev = np.concatenate([np.zeros_like(frame[:, :1]), frame[:, :-1]], axis=1)
    onset_density = np.clip(np.sum(onset, axis=0) / 8.0, 0.0, 1.0)
    note_off_density = np.clip(np.sum(np.clip(frame_prev - frame, 0.0, 1.0), axis=0) / 8.0, 0.0, 1.0)
    frame_change = np.clip(np.sum(np.abs(frame - frame_prev), axis=0) / 16.0, 0.0, 1.0)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    pitch_classes = np.remainder(pitches, 12)
    chroma = np.zeros((12, n_frames), dtype=np.float32)
    chroma_index = np.broadcast_to(pitch_classes[:, None], frame.shape)
    np.add.at(chroma, (chroma_index, np.broadcast_to(np.arange(n_frames)[None, :], frame.shape)), frame)
    chroma = np.clip(chroma, 0.0, 1.0)
    chroma_prev = np.concatenate([np.zeros_like(chroma[:, :1]), chroma[:, :-1]], axis=1)
    chord_change = np.clip(np.sum(np.abs(chroma - chroma_prev), axis=0) / 6.0, 0.0, 1.0)
    return np.stack([onset_density, note_off_density, frame_change, chord_change], axis=0).astype(np.float32)


def pitch_class_onset_targets_from_roll(onset: np.ndarray) -> np.ndarray:
    """Collapse 88-key onsets into 12 pitch-class onset targets."""

    onset = np.asarray(onset, dtype=np.float32)
    n_frames = int(onset.shape[1])
    pc_onset = np.zeros((12, n_frames), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    pitch_classes = np.remainder(pitches, 12)
    chroma_index = np.broadcast_to(pitch_classes[:, None], onset.shape)
    frame_index = np.broadcast_to(np.arange(n_frames)[None, :], onset.shape)
    np.add.at(pc_onset, (chroma_index, frame_index), onset)
    return np.clip(pc_onset, 0.0, 1.0).astype(np.float32)


def role_fullness_targets_from_roll(
    frame: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Derive bass, chord, melody, polyphony, and velocity-weight targets."""

    frame = np.asarray(frame, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((5, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    active = frame > 0.1
    active_count = np.sum(active, axis=0).astype(np.float32)
    velocity_sum = np.sum(np.clip(velocity, 0.0, 1.0) * active.astype(np.float32), axis=0)
    active_velocity = np.divide(velocity_sum, active_count + 1e-6)
    role = np.stack(
        [
            np.max(frame[pitches <= 52], axis=0),
            (active_count >= 3.0).astype(np.float32),
            np.max(frame[pitches >= 77], axis=0),
            np.clip(active_count / 10.0, 0.0, 1.0),
            np.clip(active_velocity, 0.0, 1.0),
        ],
        axis=0,
    )
    return np.clip(role, 0.0, 1.0).astype(np.float32)


def melody_targets_from_roll(
    frame: np.ndarray,
    velocity: np.ndarray,
) -> np.ndarray:
    """Derive high-register/top-line planning targets from an 88-key roll."""

    frame = np.asarray(frame, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    high_activity = np.max(frame[pitches >= 77], axis=0)
    upper_activity = np.max(frame[pitches >= 72], axis=0)
    top_pitch = np.zeros((n_frames,), dtype=np.float32)
    top_velocity = np.zeros((n_frames,), dtype=np.float32)
    for t in range(n_frames):
        active_idx = np.flatnonzero(frame[:, t] > 0.1)
        if active_idx.size == 0:
            continue
        top_idx = int(active_idx[-1])
        top_pitch[t] = np.clip(float(pitches[top_idx] - PIANO_MIN_MIDI) / float(PIANO_MAX_MIDI - PIANO_MIN_MIDI), 0.0, 1.0)
        top_velocity[t] = np.clip(float(velocity[top_idx, t]), 0.0, 1.0)
    return np.stack([high_activity, upper_activity, top_pitch, top_velocity], axis=0).astype(np.float32)


def texture_role_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Derive joint bass/body/inner/top-line texture-role targets."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    low = pitches <= 52
    mid = (pitches >= 53) & (pitches <= 76)
    upper = pitches >= 72
    bass_floor = np.max(frame[low], axis=0)
    chord_body = np.clip(np.sum(frame[mid] > 0.1, axis=0).astype(np.float32) / 5.0, 0.0, 1.0)
    inner_motion = np.clip(np.sum(onset[mid] > 0.1, axis=0).astype(np.float32) / 3.0, 0.0, 1.0)
    top_line = np.max(frame[upper], axis=0)
    return np.stack([bass_floor, chord_body, inner_motion, top_line], axis=0).astype(np.float32)


def section_role_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
    *,
    frame_hz: float = 25.0,
    section_seconds: float = 4.0,
) -> np.ndarray:
    """Broadcast section-level role coverage targets across each local section."""

    del onset
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    low = pitches <= 52
    mid = (pitches >= 53) & (pitches <= 76)
    upper = pitches >= 72
    active = frame > 0.1
    active_count = np.sum(active, axis=0).astype(np.float32)
    bass_curve = np.max(frame[low], axis=0)
    body_curve = np.clip(np.sum(active[mid], axis=0).astype(np.float32) / 5.0, 0.0, 1.0)
    melody_curve = np.max(frame[upper], axis=0)
    fullness_curve = np.clip(active_count / 8.0, 0.0, 1.0)
    section_frames = max(1, int(round(float(section_seconds) * max(1.0, float(frame_hz)))))
    out = np.zeros((4, n_frames), dtype=np.float32)
    for start in range(0, n_frames, section_frames):
        end = min(n_frames, start + section_frames)
        if end <= start:
            continue
        active_section = active_count[start:end] > 0.1
        denom = max(1, int(np.sum(active_section)))
        out[0, start:end] = float(np.sum((bass_curve[start:end] > 0.1) & active_section) / denom)
        out[1, start:end] = float(np.mean(body_curve[start:end]))
        out[2, start:end] = float(np.sum((melody_curve[start:end] > 0.1) & active_section) / denom)
        out[3, start:end] = float(np.mean(fullness_curve[start:end]))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def arranger_state_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
    *,
    frame_hz: float = 25.0,
    section_seconds: float = 4.0,
) -> np.ndarray:
    """Derive an explicit role/section arranger state before note decoding."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((8, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    low = pitches <= 52
    mid = (pitches >= 53) & (pitches <= 76)
    upper = pitches >= 72
    bass_rhythm = np.max(onset[low], axis=0)
    bass_sustain = np.max(frame[low], axis=0)
    chord_body = np.clip(np.sum(frame[mid] > 0.1, axis=0).astype(np.float32) / 5.0, 0.0, 1.0)
    inner_motion = np.clip(np.sum(onset[mid] > 0.1, axis=0).astype(np.float32) / 3.0, 0.0, 1.0)
    top_line = np.max(frame[upper], axis=0)
    section_role = section_role_targets_from_roll(
        onset,
        frame,
        frame_hz=float(frame_hz),
        section_seconds=float(section_seconds),
    )
    section_transition = np.zeros((n_frames,), dtype=np.float32)
    section_frames = max(1, int(round(float(section_seconds) * max(1.0, float(frame_hz)))))
    transition_width = max(1, int(round(0.5 * max(1.0, float(frame_hz)))))
    for start in range(0, n_frames, section_frames):
        end = min(n_frames, start + transition_width)
        section_transition[start:end] = 1.0
    arranger_state = np.stack(
        [
            bass_rhythm,
            bass_sustain,
            chord_body,
            inner_motion,
            top_line,
            section_role[0],
            section_role[3],
            section_transition,
        ],
        axis=0,
    )
    return np.clip(arranger_state, 0.0, 1.0).astype(np.float32)


def bass_continuity_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
    *,
    frame_hz: float = 25.0,
    section_seconds: float = 4.0,
) -> np.ndarray:
    """Derive a focused left-hand continuity state separate from body/top-line planning."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    low = pitches <= 52
    bass_rhythm = np.max(onset[low], axis=0)
    bass_sustain = np.max(frame[low], axis=0)
    section_role = section_role_targets_from_roll(
        onset,
        frame,
        frame_hz=float(frame_hz),
        section_seconds=float(section_seconds),
    )
    section_transition = np.zeros((n_frames,), dtype=np.float32)
    section_frames = max(1, int(round(float(section_seconds) * max(1.0, float(frame_hz)))))
    transition_width = max(1, int(round(0.5 * max(1.0, float(frame_hz)))))
    for start in range(0, n_frames, section_frames):
        end = min(n_frames, start + transition_width)
        section_transition[start:end] = 1.0
    return np.stack([bass_rhythm, bass_sustain, section_role[0], section_transition], axis=0).astype(np.float32)


def body_melody_state_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
    *,
    frame_hz: float = 25.0,
    section_seconds: float = 4.0,
) -> np.ndarray:
    """Derive chord-body, inner-motion, and top-line state independent of bass continuity."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((6, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    mid = (pitches >= 53) & (pitches <= 76)
    high = pitches >= 77
    upper = pitches >= 72
    chord_body = np.clip(np.sum(frame[mid] > 0.1, axis=0).astype(np.float32) / 5.0, 0.0, 1.0)
    inner_motion = np.clip(np.sum(onset[mid] > 0.1, axis=0).astype(np.float32) / 3.0, 0.0, 1.0)
    top_line = np.max(frame[upper], axis=0)
    high_activity = np.max(frame[high], axis=0)
    section_role = section_role_targets_from_roll(
        onset,
        frame,
        frame_hz=float(frame_hz),
        section_seconds=float(section_seconds),
    )
    return np.stack(
        [chord_body, inner_motion, top_line, high_activity, section_role[1], section_role[2]],
        axis=0,
    ).astype(np.float32)


def section_diversity_targets_from_roll(
    onset: np.ndarray,
    frame: np.ndarray,
    *,
    frame_hz: float = 25.0,
    section_seconds: float = 4.0,
) -> np.ndarray:
    """Broadcast section-level pitch-diversity and onset-density targets."""

    onset = np.asarray(onset, dtype=np.float32)
    frame = np.asarray(frame, dtype=np.float32)
    n_frames = int(frame.shape[1])
    if n_frames <= 0:
        return np.zeros((4, 0), dtype=np.float32)
    pitches = np.arange(PIANO_MIN_MIDI, PIANO_MAX_MIDI + 1, dtype=np.int32)
    pitch_classes = np.remainder(pitches, 12)
    active = frame > 0.1
    section_frames = max(1, int(round(float(section_seconds) * max(1.0, float(frame_hz)))))
    out = np.zeros((4, n_frames), dtype=np.float32)
    for start in range(0, n_frames, section_frames):
        end = min(n_frames, start + section_frames)
        if end <= start:
            continue
        section_active = np.any(active[:, start:end], axis=1)
        active_pitches = pitches[section_active]
        active_pitch_classes = np.unique(pitch_classes[section_active]) if np.any(section_active) else np.zeros((0,), dtype=np.int32)
        unique_pitch_fraction = float(min(1.0, active_pitches.size / 16.0))
        unique_pc_fraction = float(min(1.0, active_pitch_classes.size / 12.0))
        pitch_range = float((active_pitches.max() - active_pitches.min()) / max(1, PIANO_MAX_MIDI - PIANO_MIN_MIDI)) if active_pitches.size else 0.0
        onset_frames = np.any(onset[:, start:end] > 0.1, axis=0)
        onset_density = float(min(1.0, np.sum(onset_frames) / max(1, end - start)))
        out[:, start:end] = np.asarray(
            [unique_pitch_fraction, unique_pc_fraction, pitch_range, onset_density],
            dtype=np.float32,
        )[:, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def midi_to_arrangement(
    path: Path,
    *,
    seconds: float = 0.0,
    min_note_duration: float = 0.03,
) -> PianoArrangement:
    import mido

    midi = mido.MidiFile(str(path))
    tempo = 500000
    now = 0.0
    active: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
    notes: List[PianoNote] = []
    sustain: List[SustainEvent] = []
    for msg in mido.merge_tracks(midi.tracks):
        now += float(mido.tick2second(int(msg.time), int(midi.ticks_per_beat), int(tempo)))
        if msg.type == "set_tempo":
            tempo = int(msg.tempo)
            continue
        if msg.type == "control_change" and int(getattr(msg, "control", -1)) == 64:
            sustain.append(SustainEvent(time=max(0.0, now), value=int(max(0, min(127, int(msg.value))))))
            continue
        if msg.type not in {"note_on", "note_off"}:
            continue
        channel = int(getattr(msg, "channel", 0))
        if channel == 9:
            continue
        pitch = int(getattr(msg, "note", 0))
        if pitch < PIANO_MIN_MIDI or pitch > PIANO_MAX_MIDI:
            continue
        velocity = int(getattr(msg, "velocity", 0))
        key = (channel, pitch)
        if msg.type == "note_on" and velocity > 0:
            active.setdefault(key, []).append((max(0.0, now), velocity))
            continue
        starts = active.get(key, [])
        if not starts:
            continue
        start, start_velocity = starts.pop(0)
        end = now
        duration = max(float(min_note_duration), float(end - start))
        notes.append(PianoNote(start=float(start), duration=duration, pitch=pitch, velocity=max(1, min(127, start_velocity))))

    end_cap = now
    for (_channel, pitch), starts in active.items():
        for start, start_velocity in starts:
            if end_cap <= start:
                continue
            notes.append(
                PianoNote(
                    start=float(start),
                    duration=max(float(min_note_duration), float(end_cap - start)),
                    pitch=int(pitch),
                    velocity=max(1, min(127, int(start_velocity))),
                )
            )
    notes = sorted(notes, key=lambda n: (float(n.start), int(n.pitch)))
    if float(seconds) > 0 and notes:
        origin = float(notes[0].start)
        window_end = origin + float(seconds)
        cropped: List[PianoNote] = []
        for note in notes:
            if float(note.start) >= window_end:
                break
            end = min(window_end, float(note.start) + float(note.duration))
            if end <= float(note.start):
                continue
            cropped.append(
                PianoNote(
                    start=float(note.start) - origin,
                    duration=max(float(min_note_duration), end - float(note.start)),
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                )
            )
        notes = cropped
        sustain = [
            SustainEvent(time=float(ev.time) - origin, value=int(ev.value))
            for ev in sustain
            if origin <= float(ev.time) <= window_end
        ]
        duration = float(seconds)
    else:
        duration = max(float(end_cap), max((n.start + n.duration for n in notes), default=0.0))
    return PianoArrangement(
        notes=sorted(notes, key=lambda n: (float(n.start), int(n.pitch))),
        tempo_bpm=120.0,
        duration=float(duration),
        sustain=sorted(sustain, key=lambda ev: float(ev.time)),
        metadata={"source": "midi_target", "midi_path": str(path), "ticks_per_beat": int(midi.ticks_per_beat)},
    )


def _write_cache_arrays(
    *,
    cache_dir: Path,
    src_list: List[np.ndarray],
    onset_list: List[np.ndarray],
    frame_list: List[np.ndarray],
    velocity_list: List[np.ndarray],
    pedal_list: List[np.ndarray],
    density_list: List[np.ndarray],
    register_list: List[np.ndarray],
    chord_list: List[np.ndarray],
    bass_list: List[np.ndarray],
    voicing_list: List[np.ndarray],
    event_list: List[np.ndarray],
    pc_onset_list: List[np.ndarray],
    role_list: List[np.ndarray],
    melody_list: List[np.ndarray],
    texture_role_list: List[np.ndarray],
    section_role_list: List[np.ndarray],
    arranger_state_list: List[np.ndarray],
    bass_continuity_list: List[np.ndarray],
    body_melody_state_list: List[np.ndarray],
    section_diversity_list: List[np.ndarray],
    rows: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    meta_extra: Dict[str, Any],
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "source_condition.npy", np.stack(src_list).astype(np.float32))
    np.save(cache_dir / "target_onset.npy", np.stack(onset_list).astype(np.float32))
    np.save(cache_dir / "target_frame.npy", np.stack(frame_list).astype(np.float32))
    np.save(cache_dir / "target_velocity.npy", np.stack(velocity_list).astype(np.float32))
    np.save(cache_dir / "target_pedal.npy", np.stack(pedal_list).astype(np.float32))
    np.save(cache_dir / "target_density.npy", np.stack(density_list).astype(np.float32))
    np.save(cache_dir / "target_register.npy", np.stack(register_list).astype(np.float32))
    np.save(cache_dir / "target_chord.npy", np.stack(chord_list).astype(np.float32))
    np.save(cache_dir / "target_bass.npy", np.stack(bass_list).astype(np.float32))
    np.save(cache_dir / "target_voicing.npy", np.stack(voicing_list).astype(np.float32))
    np.save(cache_dir / "target_event.npy", np.stack(event_list).astype(np.float32))
    np.save(cache_dir / "target_pc_onset.npy", np.stack(pc_onset_list).astype(np.float32))
    np.save(cache_dir / "target_role.npy", np.stack(role_list).astype(np.float32))
    np.save(cache_dir / "target_melody.npy", np.stack(melody_list).astype(np.float32))
    np.save(cache_dir / "target_texture_role.npy", np.stack(texture_role_list).astype(np.float32))
    np.save(cache_dir / "target_section_role.npy", np.stack(section_role_list).astype(np.float32))
    np.save(cache_dir / "target_arranger_state.npy", np.stack(arranger_state_list).astype(np.float32))
    np.save(cache_dir / "target_bass_continuity.npy", np.stack(bass_continuity_list).astype(np.float32))
    np.save(cache_dir / "target_body_melody_state.npy", np.stack(body_melody_state_list).astype(np.float32))
    np.save(cache_dir / "target_section_diversity.npy", np.stack(section_diversity_list).astype(np.float32))
    pd.DataFrame(rows).to_csv(cache_dir / "index.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(cache_dir / "errors.csv", index=False)
    meta = {
        "cache_dir": str(cache_dir),
        "n_samples": int(len(rows)),
        "n_errors": int(len(errors)),
        "source_features": SOURCE_FEATURE_NAMES,
        "source_condition_shape": list(np.stack(src_list[:1]).shape[1:]),
        "target_onset_shape": list(np.stack(onset_list[:1]).shape[1:]),
        "target_frame_shape": list(np.stack(frame_list[:1]).shape[1:]),
        "target_velocity_shape": list(np.stack(velocity_list[:1]).shape[1:]),
        "target_pedal_shape": list(np.stack(pedal_list[:1]).shape[1:]),
        "target_density_shape": list(np.stack(density_list[:1]).shape[1:]),
        "target_register_shape": list(np.stack(register_list[:1]).shape[1:]),
        "target_chord_shape": list(np.stack(chord_list[:1]).shape[1:]),
        "target_bass_shape": list(np.stack(bass_list[:1]).shape[1:]),
        "target_voicing_shape": list(np.stack(voicing_list[:1]).shape[1:]),
        "target_event_shape": list(np.stack(event_list[:1]).shape[1:]),
        "target_pc_onset_shape": list(np.stack(pc_onset_list[:1]).shape[1:]),
        "target_role_shape": list(np.stack(role_list[:1]).shape[1:]),
        "target_melody_shape": list(np.stack(melody_list[:1]).shape[1:]),
        "target_texture_role_shape": list(np.stack(texture_role_list[:1]).shape[1:]),
        "target_section_role_shape": list(np.stack(section_role_list[:1]).shape[1:]),
        "target_arranger_state_shape": list(np.stack(arranger_state_list[:1]).shape[1:]),
        "target_bass_continuity_shape": list(np.stack(bass_continuity_list[:1]).shape[1:]),
        "target_body_melody_state_shape": list(np.stack(body_melody_state_list[:1]).shape[1:]),
        "target_section_diversity_shape": list(np.stack(section_diversity_list[:1]).shape[1:]),
        "piano_min_midi": PIANO_MIN_MIDI,
        "piano_max_midi": PIANO_MAX_MIDI,
        **meta_extra,
    }
    _write_json(cache_dir / "meta.json", meta)
    return meta


def build_piano_arranger_cache(config: PianoCacheConfig = PianoCacheConfig()) -> Dict[str, Any]:
    manifest = Path(config.manifest)
    cache_dir = Path(config.cache_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing piano manifest: {manifest}")
    df = pd.read_csv(manifest)
    if "path" not in df.columns:
        raise ValueError(f"Manifest missing path column: {manifest}")
    if int(config.max_tracks) > 0:
        df = df.head(int(config.max_tracks)).copy()
    if len(df) == 0:
        raise ValueError(f"Manifest has no rows: {manifest}")

    src_list: List[np.ndarray] = []
    onset_list: List[np.ndarray] = []
    frame_list: List[np.ndarray] = []
    velocity_list: List[np.ndarray] = []
    pedal_list: List[np.ndarray] = []
    density_list: List[np.ndarray] = []
    register_list: List[np.ndarray] = []
    chord_list: List[np.ndarray] = []
    bass_list: List[np.ndarray] = []
    voicing_list: List[np.ndarray] = []
    event_list: List[np.ndarray] = []
    pc_onset_list: List[np.ndarray] = []
    role_list: List[np.ndarray] = []
    melody_list: List[np.ndarray] = []
    texture_role_list: List[np.ndarray] = []
    section_role_list: List[np.ndarray] = []
    arranger_state_list: List[np.ndarray] = []
    bass_continuity_list: List[np.ndarray] = []
    body_melody_state_list: List[np.ndarray] = []
    section_diversity_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    baseline_cfg = HeuristicPianoConfig(
        seconds=float(config.seconds),
        sample_rate=DIFFUSION_SR,
        hop_length=512,
        fullness=float(config.fullness),
        melody_focus=float(config.melody_focus),
        rhythmic_drive=float(config.rhythmic_drive),
        harmonic_adventure=float(config.harmonic_adventure),
        register_width=float(config.register_width),
        pedal_amount=float(config.pedal_amount),
        render_wav=False,
    )

    for i, rec in df.reset_index(drop=True).iterrows():
        path = Path(str(rec["path"]))
        try:
            y = load_audio_for_cache(path, seconds=float(config.seconds), sr=DIFFUSION_SR)
            source = extract_source_condition(
                y,
                sr=DIFFUSION_SR,
                frame_hz=float(config.frame_hz),
                max_frames=int(config.max_frames),
            )
            arrangement = arrange_audio_heuristic(path, baseline_cfg)
            onset, frame, velocity, pedal = piano_roll_from_arrangement(
                arrangement,
                max_frames=int(config.max_frames),
                frame_hz=float(config.frame_hz),
            )
            density, register = hierarchy_targets_from_roll(onset, frame)
            chord, bass, voicing = musical_plan_targets_from_roll(onset, frame)
            event = event_plan_targets_from_roll(onset, frame)
            pc_onset = pitch_class_onset_targets_from_roll(onset)
            role = role_fullness_targets_from_roll(frame, velocity)
            melody = melody_targets_from_roll(frame, velocity)
            texture_role = texture_role_targets_from_roll(onset, frame)
            section_role = section_role_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            arranger_state = arranger_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            bass_continuity = bass_continuity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            body_melody_state = body_melody_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            section_diversity = section_diversity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            src_list.append(source)
            onset_list.append(onset)
            frame_list.append(frame)
            velocity_list.append(velocity)
            pedal_list.append(pedal)
            density_list.append(density)
            register_list.append(register)
            chord_list.append(chord)
            bass_list.append(bass)
            voicing_list.append(voicing)
            event_list.append(event)
            pc_onset_list.append(pc_onset)
            role_list.append(role)
            melody_list.append(melody)
            texture_role_list.append(texture_role)
            section_role_list.append(section_role)
            arranger_state_list.append(arranger_state)
            bass_continuity_list.append(bass_continuity)
            body_melody_state_list.append(body_melody_state)
            section_diversity_list.append(section_diversity)
            rows.append(
                {
                    "cache_idx": len(rows),
                    "manifest_idx": int(i),
                    "path": str(path),
                    "title": str(rec.get("title", "")),
                    "artist": str(rec.get("artist", "")),
                    "album": str(rec.get("album", "")),
                    "genre": str(rec.get("genre", "")),
                    "duration_sec": float(rec.get("duration_sec", 0.0) or 0.0),
                    "piano_score": float(rec.get("piano_score", 0.0) or 0.0),
                    "notes": int(len(arrangement.notes)),
                    "source_seconds": float(len(y) / float(DIFFUSION_SR)),
                }
            )
        except Exception as exc:
            errors.append({"manifest_idx": int(i), "path": str(path), "error": str(exc)})

    if not src_list:
        raise RuntimeError(f"No cache rows could be built from {manifest}; first error: {errors[:1]}")

    return _write_cache_arrays(
        cache_dir=cache_dir,
        src_list=src_list,
        onset_list=onset_list,
        frame_list=frame_list,
        velocity_list=velocity_list,
        pedal_list=pedal_list,
        density_list=density_list,
        register_list=register_list,
        chord_list=chord_list,
        bass_list=bass_list,
        voicing_list=voicing_list,
        event_list=event_list,
        pc_onset_list=pc_onset_list,
        role_list=role_list,
        melody_list=melody_list,
        texture_role_list=texture_role_list,
        section_role_list=section_role_list,
        arranger_state_list=arranger_state_list,
        bass_continuity_list=bass_continuity_list,
        body_melody_state_list=body_melody_state_list,
        section_diversity_list=section_diversity_list,
        rows=rows,
        errors=errors,
        meta_extra={
        "manifest": str(manifest),
        "seconds": float(config.seconds),
        "frame_hz": float(config.frame_hz),
        "max_frames": int(config.max_frames),
        "target_source": "heuristic_pseudo_target_v1",
        "config": asdict(config),
        },
    )


def build_midi_piano_target_cache(config: MidiPianoCacheConfig = MidiPianoCacheConfig()) -> Dict[str, Any]:
    manifest = Path(config.manifest)
    cache_dir = Path(config.cache_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing MIDI manifest: {manifest}")
    df = pd.read_csv(manifest)
    if "path" not in df.columns:
        raise ValueError(f"Manifest missing path column: {manifest}")
    if int(config.max_tracks) > 0:
        df = df.head(int(config.max_tracks)).copy()
    if len(df) == 0:
        raise ValueError(f"Manifest has no rows: {manifest}")

    src_list: List[np.ndarray] = []
    onset_list: List[np.ndarray] = []
    frame_list: List[np.ndarray] = []
    velocity_list: List[np.ndarray] = []
    pedal_list: List[np.ndarray] = []
    density_list: List[np.ndarray] = []
    register_list: List[np.ndarray] = []
    chord_list: List[np.ndarray] = []
    bass_list: List[np.ndarray] = []
    voicing_list: List[np.ndarray] = []
    event_list: List[np.ndarray] = []
    pc_onset_list: List[np.ndarray] = []
    role_list: List[np.ndarray] = []
    melody_list: List[np.ndarray] = []
    texture_role_list: List[np.ndarray] = []
    section_role_list: List[np.ndarray] = []
    arranger_state_list: List[np.ndarray] = []
    bass_continuity_list: List[np.ndarray] = []
    body_melody_state_list: List[np.ndarray] = []
    section_diversity_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, rec in df.reset_index(drop=True).iterrows():
        path = Path(str(rec["path"]))
        try:
            arrangement = midi_to_arrangement(path, seconds=float(config.seconds))
            if len(arrangement.notes) < int(config.min_notes):
                raise ValueError(f"Too few piano notes after parsing: {len(arrangement.notes)}")
            audio = arrangement_to_source_preview_audio(
                arrangement,
                mode=str(config.source_preview_mode),
                sample_rate=int(config.preview_sample_rate),
            )
            source = extract_source_condition(
                audio,
                sr=int(config.preview_sample_rate),
                frame_hz=float(config.frame_hz),
                max_frames=int(config.max_frames),
            )
            onset, frame, velocity, pedal = piano_roll_from_arrangement(
                arrangement,
                max_frames=int(config.max_frames),
                frame_hz=float(config.frame_hz),
            )
            density, register = hierarchy_targets_from_roll(onset, frame)
            chord, bass, voicing = musical_plan_targets_from_roll(onset, frame)
            event = event_plan_targets_from_roll(onset, frame)
            pc_onset = pitch_class_onset_targets_from_roll(onset)
            role = role_fullness_targets_from_roll(frame, velocity)
            melody = melody_targets_from_roll(frame, velocity)
            texture_role = texture_role_targets_from_roll(onset, frame)
            section_role = section_role_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            arranger_state = arranger_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            bass_continuity = bass_continuity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            body_melody_state = body_melody_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            section_diversity = section_diversity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            src_list.append(source)
            onset_list.append(onset)
            frame_list.append(frame)
            velocity_list.append(velocity)
            pedal_list.append(pedal)
            density_list.append(density)
            register_list.append(register)
            chord_list.append(chord)
            bass_list.append(bass)
            voicing_list.append(voicing)
            event_list.append(event)
            pc_onset_list.append(pc_onset)
            role_list.append(role)
            melody_list.append(melody)
            texture_role_list.append(texture_role)
            section_role_list.append(section_role)
            arranger_state_list.append(arranger_state)
            bass_continuity_list.append(bass_continuity)
            body_melody_state_list.append(body_melody_state)
            section_diversity_list.append(section_diversity)
            rows.append(
                {
                    "cache_idx": len(rows),
                    "manifest_idx": int(i),
                    "path": str(path),
                    "title": str(rec.get("title", path.stem)),
                    "source": str(rec.get("source", "midi_discovery")),
                    "notes": int(len(arrangement.notes)),
                    "duration_sec": float(arrangement.duration),
                    "source_seconds": float(audio.shape[0] / float(config.preview_sample_rate)),
                    "target_source": "midi_symbolic_target_v1",
                    "source_preview_mode": str(config.source_preview_mode),
                }
            )
        except Exception as exc:
            errors.append({"manifest_idx": int(i), "path": str(path), "error": str(exc)})

    if not src_list:
        raise RuntimeError(f"No MIDI cache rows could be built from {manifest}; first error: {errors[:1]}")

    return _write_cache_arrays(
        cache_dir=cache_dir,
        src_list=src_list,
        onset_list=onset_list,
        frame_list=frame_list,
        velocity_list=velocity_list,
        pedal_list=pedal_list,
        density_list=density_list,
        register_list=register_list,
        chord_list=chord_list,
        bass_list=bass_list,
        voicing_list=voicing_list,
        event_list=event_list,
        pc_onset_list=pc_onset_list,
        role_list=role_list,
        melody_list=melody_list,
        texture_role_list=texture_role_list,
        section_role_list=section_role_list,
        arranger_state_list=arranger_state_list,
        bass_continuity_list=bass_continuity_list,
        body_melody_state_list=body_melody_state_list,
        section_diversity_list=section_diversity_list,
        rows=rows,
        errors=errors,
        meta_extra={
            "manifest": str(manifest),
            "seconds": float(config.seconds),
            "frame_hz": float(config.frame_hz),
            "max_frames": int(config.max_frames),
            "target_source": "midi_symbolic_target_v1",
            "conditioning_source": f"deterministic_{str(config.source_preview_mode)}_preview_audio_from_midi_target",
            "config": asdict(config),
        },
    )


def build_paired_audio_midi_target_cache(config: PairedAudioMidiCacheConfig = PairedAudioMidiCacheConfig()) -> Dict[str, Any]:
    manifest = Path(config.manifest)
    cache_dir = Path(config.cache_dir)
    if not manifest.exists():
        raise FileNotFoundError(f"Missing paired audio/MIDI manifest: {manifest}")
    df = pd.read_csv(manifest)
    required = {"source_audio", "target_midi"}
    missing = sorted(required.difference(set(df.columns)))
    if missing:
        raise ValueError(f"Paired manifest missing columns {missing}; expected source_audio,target_midi: {manifest}")
    if int(config.max_tracks) > 0:
        df = df.head(int(config.max_tracks)).copy()
    if len(df) == 0:
        raise ValueError(f"Manifest has no rows: {manifest}")

    src_list: List[np.ndarray] = []
    onset_list: List[np.ndarray] = []
    frame_list: List[np.ndarray] = []
    velocity_list: List[np.ndarray] = []
    pedal_list: List[np.ndarray] = []
    density_list: List[np.ndarray] = []
    register_list: List[np.ndarray] = []
    chord_list: List[np.ndarray] = []
    bass_list: List[np.ndarray] = []
    voicing_list: List[np.ndarray] = []
    event_list: List[np.ndarray] = []
    pc_onset_list: List[np.ndarray] = []
    role_list: List[np.ndarray] = []
    melody_list: List[np.ndarray] = []
    texture_role_list: List[np.ndarray] = []
    section_role_list: List[np.ndarray] = []
    arranger_state_list: List[np.ndarray] = []
    bass_continuity_list: List[np.ndarray] = []
    body_melody_state_list: List[np.ndarray] = []
    section_diversity_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for i, rec in df.reset_index(drop=True).iterrows():
        source_path = _resolve_manifest_path(rec["source_audio"], manifest)
        target_midi = _resolve_manifest_path(rec["target_midi"], manifest)
        try:
            y = load_audio_for_cache(source_path, seconds=float(config.seconds), sr=DIFFUSION_SR)
            source = extract_source_condition(
                y,
                sr=DIFFUSION_SR,
                frame_hz=float(config.frame_hz),
                max_frames=int(config.max_frames),
            )
            arrangement = midi_to_arrangement(target_midi, seconds=float(config.seconds))
            if len(arrangement.notes) < int(config.min_notes):
                raise ValueError(f"Too few piano notes after parsing target MIDI: {len(arrangement.notes)}")
            onset, frame, velocity, pedal = piano_roll_from_arrangement(
                arrangement,
                max_frames=int(config.max_frames),
                frame_hz=float(config.frame_hz),
            )
            density, register = hierarchy_targets_from_roll(onset, frame)
            chord, bass, voicing = musical_plan_targets_from_roll(onset, frame)
            event = event_plan_targets_from_roll(onset, frame)
            pc_onset = pitch_class_onset_targets_from_roll(onset)
            role = role_fullness_targets_from_roll(frame, velocity)
            melody = melody_targets_from_roll(frame, velocity)
            texture_role = texture_role_targets_from_roll(onset, frame)
            section_role = section_role_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            arranger_state = arranger_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            bass_continuity = bass_continuity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            body_melody_state = body_melody_state_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            section_diversity = section_diversity_targets_from_roll(onset, frame, frame_hz=float(config.frame_hz))
            src_list.append(source)
            onset_list.append(onset)
            frame_list.append(frame)
            velocity_list.append(velocity)
            pedal_list.append(pedal)
            density_list.append(density)
            register_list.append(register)
            chord_list.append(chord)
            bass_list.append(bass)
            voicing_list.append(voicing)
            event_list.append(event)
            pc_onset_list.append(pc_onset)
            role_list.append(role)
            melody_list.append(melody)
            texture_role_list.append(texture_role)
            section_role_list.append(section_role)
            arranger_state_list.append(arranger_state)
            bass_continuity_list.append(bass_continuity)
            body_melody_state_list.append(body_melody_state)
            section_diversity_list.append(section_diversity)
            rows.append(
                {
                    "cache_idx": len(rows),
                    "manifest_idx": int(i),
                    "path": str(source_path),
                    "source_audio": str(source_path),
                    "target_midi": str(target_midi),
                    "title": str(rec.get("title", source_path.stem)),
                    "artist": str(rec.get("artist", "")),
                    "source": str(rec.get("source", "paired_audio_midi_manifest")),
                    "notes": int(len(arrangement.notes)),
                    "duration_sec": float(arrangement.duration),
                    "source_seconds": float(len(y) / float(DIFFUSION_SR)),
                    "target_source": "paired_audio_midi_symbolic_target_v1",
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "manifest_idx": int(i),
                    "source_audio": str(source_path),
                    "target_midi": str(target_midi),
                    "error": str(exc),
                }
            )

    if not src_list:
        raise RuntimeError(f"No paired cache rows could be built from {manifest}; first error: {errors[:1]}")

    return _write_cache_arrays(
        cache_dir=cache_dir,
        src_list=src_list,
        onset_list=onset_list,
        frame_list=frame_list,
        velocity_list=velocity_list,
        pedal_list=pedal_list,
        density_list=density_list,
        register_list=register_list,
        chord_list=chord_list,
        bass_list=bass_list,
        voicing_list=voicing_list,
        event_list=event_list,
        pc_onset_list=pc_onset_list,
        role_list=role_list,
        melody_list=melody_list,
        texture_role_list=texture_role_list,
        section_role_list=section_role_list,
        arranger_state_list=arranger_state_list,
        bass_continuity_list=bass_continuity_list,
        body_melody_state_list=body_melody_state_list,
        section_diversity_list=section_diversity_list,
        rows=rows,
        errors=errors,
        meta_extra={
            "manifest": str(manifest),
            "seconds": float(config.seconds),
            "frame_hz": float(config.frame_hz),
            "max_frames": int(config.max_frames),
            "target_source": "paired_audio_midi_symbolic_target_v1",
            "conditioning_source": "manifest_source_audio",
            "manifest_contract": "source_audio,target_midi",
            "config": asdict(config),
        },
    )


__all__ = [
    "DEFAULT_PIANO_CACHE_DIR",
    "PIANO_MAX_MIDI",
    "PIANO_MIN_MIDI",
    "PIANO_N_KEYS",
    "MidiPianoCacheConfig",
    "PairedAudioMidiCacheConfig",
    "PianoCacheConfig",
    "SOURCE_FEATURE_NAMES",
    "build_midi_piano_target_cache",
    "build_paired_audio_midi_target_cache",
    "build_piano_arranger_cache",
    "arranger_state_targets_from_roll",
    "bass_continuity_targets_from_roll",
    "body_melody_state_targets_from_roll",
    "extract_source_condition",
    "event_plan_targets_from_roll",
    "hierarchy_targets_from_roll",
    "load_audio_for_cache",
    "melody_targets_from_roll",
    "musical_plan_targets_from_roll",
    "midi_to_arrangement",
    "pitch_class_onset_targets_from_roll",
    "piano_roll_from_arrangement",
    "role_fullness_targets_from_roll",
    "section_diversity_targets_from_roll",
    "section_role_targets_from_roll",
    "texture_role_targets_from_roll",
]
