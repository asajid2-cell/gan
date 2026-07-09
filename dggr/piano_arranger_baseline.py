from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import librosa
import numpy as np

from .lab3_diffusion_data import DIFFUSION_SR
from .piano_arranger_render import PianoArrangement, PianoNote, SustainEvent, write_arrangement_bundle


@dataclass(frozen=True)
class HeuristicPianoConfig:
    seconds: float = 30.0
    sample_rate: int = DIFFUSION_SR
    hop_length: int = 512
    fullness: float = 0.85
    melody_focus: float = 0.80
    rhythmic_drive: float = 0.65
    harmonic_adventure: float = 0.25
    register_width: float = 0.85
    pedal_amount: float = 0.70
    render_wav: bool = True


MAJOR_THIRD = 4
MINOR_THIRD = 3
FIFTH = 7
SEVENTH = 10
SIXTH = 9


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _clip01(v: float) -> float:
    return float(max(0.0, min(1.0, float(v))))


def _pc_to_midi(pc: int, base_octave: int) -> int:
    return int(base_octave * 12 + int(pc))


def _nearest_pitch_class(pc: int, center: int) -> int:
    candidates = [int(pc) + 12 * octave for octave in range(2, 9)]
    return int(min(candidates, key=lambda p: abs(p - int(center))))


def _safe_tempo(y: np.ndarray, sr: int, hop_length: int) -> Tuple[float, np.ndarray]:
    try:
        tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo = float(np.asarray(tempo_raw).reshape(-1)[0]) if np.asarray(tempo_raw).size else 120.0
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
    except Exception:
        tempo = 120.0
        beat_frames = np.zeros((0,), dtype=np.int64)
    if not np.isfinite(tempo) or tempo <= 30.0:
        tempo = 120.0
    return float(tempo), beat_frames


def _beat_times(y: np.ndarray, sr: int, hop_length: int) -> Tuple[float, np.ndarray]:
    tempo, beat_frames = _safe_tempo(y, sr, hop_length)
    if beat_frames.shape[0] >= 2:
        times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    else:
        step = 60.0 / tempo
        times = np.arange(0.0, max(step, len(y) / float(sr)), step, dtype=np.float32)
    duration = len(y) / float(sr)
    times = np.asarray(times, dtype=np.float32)
    times = times[(times >= 0.0) & (times < duration)]
    if times.size == 0 or times[0] > 0.05:
        times = np.concatenate([np.asarray([0.0], dtype=np.float32), times])
    return tempo, times


def _frame_slice(times: np.ndarray, frame_times: np.ndarray, i: int) -> np.ndarray:
    start = float(times[i])
    end = float(times[i + 1]) if i + 1 < len(times) else start + 0.5
    return np.where((frame_times >= start) & (frame_times < end))[0]


def _choose_chord(chroma_vec: np.ndarray, prev_root: int | None, adventure: float) -> Tuple[int, bool, List[int]]:
    chroma_vec = np.asarray(chroma_vec, dtype=np.float32)
    if chroma_vec.size != 12 or float(np.sum(chroma_vec)) <= 1e-6:
        root = int(prev_root if prev_root is not None else 0)
    else:
        weighted = chroma_vec.copy()
        if prev_root is not None:
            weighted[int(prev_root) % 12] *= 1.12
        root = int(np.argmax(weighted))
    major_score = float(chroma_vec[(root + MAJOR_THIRD) % 12] + chroma_vec[(root + FIFTH) % 12])
    minor_score = float(chroma_vec[(root + MINOR_THIRD) % 12] + chroma_vec[(root + FIFTH) % 12])
    is_minor = minor_score > major_score * 1.03
    third = MINOR_THIRD if is_minor else MAJOR_THIRD
    pcs = [root, (root + third) % 12, (root + FIFTH) % 12]
    if adventure > 0.45:
        pcs.append((root + SEVENTH) % 12)
    elif adventure > 0.25 and chroma_vec[(root + SIXTH) % 12] > chroma_vec.mean():
        pcs.append((root + SIXTH) % 12)
    return root, is_minor, pcs


def _velocity(energy: float, onset: float, base: int = 58, span: int = 48) -> int:
    val = float(0.72 * energy + 0.28 * onset)
    return int(max(28, min(124, round(base + span * val))))


def _normalize_feature(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 10)) if x.size else 0.0
    hi = float(np.percentile(x, 95)) if x.size else 1.0
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def arrange_audio_heuristic(source_audio: Path, config: HeuristicPianoConfig = HeuristicPianoConfig()) -> PianoArrangement:
    sr = int(config.sample_rate)
    y, _ = librosa.load(
        str(source_audio),
        sr=sr,
        mono=True,
        duration=float(config.seconds) if float(config.seconds) > 0 else None,
        dtype=np.float32,
        res_type="soxr_hq",
    )
    if y.size < int(0.25 * sr):
        raise ValueError(f"Audio is too short for arrangement: {source_audio}")
    y = librosa.util.normalize(y)
    duration = float(y.size / float(sr))
    hop = int(config.hop_length)
    tempo, beats = _beat_times(y, sr, hop)
    if beats.size < 2:
        beats = np.asarray([0.0, min(duration, 0.5)], dtype=np.float32)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    rms_n = _normalize_feature(rms)
    onset_n = _normalize_feature(onset)

    fullness = _clip01(config.fullness)
    melody_focus = _clip01(config.melody_focus)
    drive = _clip01(config.rhythmic_drive)
    adventure = _clip01(config.harmonic_adventure)
    width = _clip01(config.register_width)
    pedal_amount = _clip01(config.pedal_amount)

    notes: List[PianoNote] = []
    sustain: List[SustainEvent] = []
    prev_root: int | None = None
    chord_history: List[Dict[str, Any]] = []
    beat_step = float(np.median(np.diff(beats))) if beats.size > 2 else 60.0 / tempo
    beat_step = max(0.20, min(1.20, beat_step))
    bass_oct = 2 if width > 0.55 else 3
    chord_center = 61 + int(round(8 * width))
    melody_center = 72 + int(round(8 * width))

    if pedal_amount > 0.15:
        sustain.append(SustainEvent(time=0.0, value=int(45 + 55 * pedal_amount)))

    for i, start_raw in enumerate(beats.tolist()):
        start = float(start_raw)
        end = float(beats[i + 1]) if i + 1 < len(beats) else min(duration, start + beat_step)
        dur = max(0.12, end - start)
        idx = _frame_slice(beats, frame_times, i)
        if idx.size == 0:
            idx = np.asarray([min(chroma.shape[1] - 1, max(0, int(round(start / max(1e-6, duration) * chroma.shape[1]))))])
        ch = np.mean(chroma[:, idx], axis=1)
        energy = float(np.mean(rms_n[idx])) if idx.size else 0.5
        hit = float(np.mean(onset_n[idx])) if idx.size else 0.3
        root, is_minor, pcs = _choose_chord(ch, prev_root, adventure)
        prev_root = root
        vel = _velocity(energy, hit, base=52 + int(12 * fullness), span=42)
        chord_history.append({"time": start, "root_pc": int(root), "minor": bool(is_minor), "pcs": [int(p) for p in pcs]})

        # Left hand: root/fifth/octave gives the arrangement weight.
        bass_root = _pc_to_midi(root, bass_oct)
        if bass_root < 28:
            bass_root += 12
        notes.append(PianoNote(start=start, duration=min(dur * 1.30, dur + 0.55), pitch=bass_root, velocity=max(34, vel - 12)))
        if fullness > 0.35:
            notes.append(PianoNote(start=start + dur * 0.48, duration=min(dur * 0.72, 0.65), pitch=bass_root + 12, velocity=max(30, vel - 20)))
        if fullness > 0.65 and i % 2 == 0:
            notes.append(PianoNote(start=start + dur * 0.24, duration=min(dur * 0.65, 0.60), pitch=bass_root + 7, velocity=max(28, vel - 24)))

        # Right hand chord voicing.
        chord_pitches = sorted({_nearest_pitch_class(pc, chord_center) for pc in pcs})
        if fullness > 0.55:
            chord_pitches.append(chord_pitches[0] + 12)
        chord_start = start + dur * (0.02 if hit > 0.45 else 0.10)
        chord_dur = min(dur * (1.05 + 0.30 * pedal_amount), dur + 0.55)
        for j, pitch in enumerate(chord_pitches):
            notes.append(PianoNote(start=chord_start + 0.006 * j, duration=chord_dur, pitch=pitch, velocity=max(30, vel - 8 + j * 2)))

        # Inner rhythmic motion: arpeggio/repeated tones without losing the chord.
        if drive > 0.20:
            subdivisions = 2 + int(round(2 * drive))
            arp_pool = chord_pitches + [p + 12 for p in chord_pitches[:2]]
            for s in range(1, subdivisions):
                t = start + dur * (s / subdivisions)
                pitch = arp_pool[(i + s) % len(arp_pool)]
                notes.append(PianoNote(start=t, duration=min(dur / subdivisions * 0.78, 0.34), pitch=pitch, velocity=max(24, vel - 22 + int(12 * drive))))

        # Melody proxy: strongest pitch class in the source chroma, placed in upper register.
        if melody_focus > 0.10:
            melody_pc = int(np.argmax(ch))
            melody_pitch = _nearest_pitch_class(melody_pc, melody_center)
            if melody_pitch <= max(chord_pitches):
                melody_pitch += 12
            notes.append(PianoNote(start=start + dur * 0.06, duration=min(dur * 0.82, 0.90), pitch=melody_pitch, velocity=min(124, vel + int(18 * melody_focus))))
            if fullness > 0.75 and width > 0.55:
                notes.append(PianoNote(start=start + dur * 0.06, duration=min(dur * 0.74, 0.82), pitch=melody_pitch + 12, velocity=max(24, vel - 18)))

    if pedal_amount > 0.15:
        sustain.append(SustainEvent(time=max(0.0, duration - 0.05), value=0))

    notes = [n for n in notes if 21 <= int(n.pitch) <= 108 and n.start < duration + 0.1]
    metadata = {
        "source_audio": str(source_audio),
        "config": asdict(config),
        "heuristic": "source_chroma_onset_beat_to_full_piano_v1",
        "beat_count": int(len(beats)),
        "chord_history": chord_history[:256],
    }
    return PianoArrangement(notes=notes, tempo_bpm=float(tempo), duration=duration, sustain=sustain, metadata=metadata)


def render_heuristic_baseline(
    *,
    source_audio: Path,
    out_stem: Path,
    config: HeuristicPianoConfig = HeuristicPianoConfig(),
) -> Dict[str, Any]:
    arrangement = arrange_audio_heuristic(source_audio, config)
    bundle = write_arrangement_bundle(arrangement, out_stem=out_stem, render_wav=bool(config.render_wav))
    summary = {
        "source_audio": str(source_audio),
        "out_stem": str(out_stem),
        "config": asdict(config),
        "bundle": bundle,
        "notes": int(len(arrangement.notes)),
        "tempo_bpm": float(arrangement.tempo_bpm),
        "duration": float(arrangement.duration),
        "render_type": "heuristic_piano_baseline",
    }
    summary_path = Path(out_stem).with_name(Path(out_stem).name + ".summary.json")
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary


__all__ = [
    "HeuristicPianoConfig",
    "arrange_audio_heuristic",
    "render_heuristic_baseline",
]
