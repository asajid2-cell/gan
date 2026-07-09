from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class PianoNote:
    start: float
    duration: float
    pitch: int
    velocity: int = 80


@dataclass(frozen=True)
class SustainEvent:
    time: float
    value: int


@dataclass
class PianoArrangement:
    notes: List[PianoNote]
    tempo_bpm: float
    duration: float
    sustain: List[SustainEvent]
    metadata: Dict[str, Any]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _vlq(value: int) -> bytes:
    value = max(0, int(value))
    parts = [value & 0x7F]
    value >>= 7
    while value:
        parts.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(parts))


def _clamp_midi_note(pitch: int) -> int:
    return int(max(21, min(108, int(round(pitch)))))


def _clamp_velocity(velocity: int) -> int:
    return int(max(1, min(127, int(round(velocity)))))


def write_midi(
    arrangement: PianoArrangement,
    out_midi: Path,
    *,
    ticks_per_beat: int = 480,
    channel: int = 0,
    program: int = 0,
) -> Dict[str, Any]:
    """Write a type-0 standard MIDI file without external MIDI dependencies."""

    out_midi = Path(out_midi)
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    tempo_bpm = float(arrangement.tempo_bpm if arrangement.tempo_bpm > 0 else 120.0)
    seconds_to_ticks = float(ticks_per_beat) * tempo_bpm / 60.0
    tempo_us = int(round(60_000_000.0 / tempo_bpm))

    events: List[Tuple[int, int, bytes]] = []
    events.append((0, 0, bytes([0xFF, 0x51, 0x03]) + tempo_us.to_bytes(3, "big")))
    events.append((0, 1, bytes([0xC0 | int(channel), int(program)])))

    for ev in arrangement.sustain:
        tick = int(round(float(ev.time) * seconds_to_ticks))
        events.append((tick, 2, bytes([0xB0 | int(channel), 64, int(max(0, min(127, ev.value)))])))

    for note in arrangement.notes:
        start = max(0.0, float(note.start))
        dur = max(0.03, float(note.duration))
        end = max(start + 0.03, start + dur)
        pitch = _clamp_midi_note(note.pitch)
        vel = _clamp_velocity(note.velocity)
        start_tick = int(round(start * seconds_to_ticks))
        end_tick = int(round(end * seconds_to_ticks))
        events.append((start_tick, 3, bytes([0x90 | int(channel), pitch, vel])))
        events.append((end_tick, 2, bytes([0x80 | int(channel), pitch, 0])))

    events.sort(key=lambda item: (item[0], item[1]))
    track = bytearray()
    last_tick = 0
    for tick, _priority, data in events:
        delta = max(0, int(tick) - int(last_tick))
        track.extend(_vlq(delta))
        track.extend(data)
        last_tick = int(tick)
    track.extend(_vlq(0))
    track.extend(bytes([0xFF, 0x2F, 0x00]))

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, int(ticks_per_beat))
    body = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    out_midi.write_bytes(header + body)
    return {
        "out_midi": str(out_midi),
        "tempo_bpm": tempo_bpm,
        "ticks_per_beat": int(ticks_per_beat),
        "notes": int(len(arrangement.notes)),
        "sustain_events": int(len(arrangement.sustain)),
    }


def _note_envelope(n: int, sr: int, duration: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    attack = max(0.004, min(0.030, 0.010 + 0.010 * (1.0 - min(1.0, duration))))
    decay = np.exp(-2.5 * t / max(0.15, float(duration) + 0.35))
    env = np.minimum(1.0, t / attack) * decay
    release_len = min(n, max(1, int(0.12 * sr)))
    if release_len > 1:
        env[-release_len:] *= np.linspace(1.0, 0.0, release_len, dtype=np.float32)
    return env.astype(np.float32)


def _synth_note(note: PianoNote, sr: int) -> np.ndarray:
    dur = max(0.05, float(note.duration) + 0.22)
    n = max(1, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    freq = 440.0 * (2.0 ** ((_clamp_midi_note(note.pitch) - 69) / 12.0))
    vel = float(_clamp_velocity(note.velocity)) / 127.0
    env = _note_envelope(n, sr, float(note.duration))
    # Simple bright piano preview: inharmonic partials plus fast-decaying hammer noise.
    sig = (
        1.00 * np.sin(2.0 * np.pi * freq * t)
        + 0.38 * np.sin(2.0 * np.pi * freq * 2.01 * t)
        + 0.18 * np.sin(2.0 * np.pi * freq * 3.02 * t)
        + 0.08 * np.sin(2.0 * np.pi * freq * 4.04 * t)
    )
    hammer = np.sin(2.0 * np.pi * min(7000.0, freq * 9.0) * t) * np.exp(-80.0 * t)
    return ((sig * env + 0.05 * hammer) * (0.12 + 0.88 * vel) * 0.16).astype(np.float32)


def _source_note_envelope(n: int, sr: int, duration: float, role: str) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / float(sr)
    if role == "pad":
        attack = min(0.18, max(0.03, float(duration) * 0.10))
        release = min(n, max(1, int(0.22 * sr)))
        env = np.minimum(1.0, t / max(1e-4, attack)) * np.exp(-0.65 * t / max(0.25, float(duration) + 0.25))
    else:
        attack = 0.006 if role == "bass" else 0.010
        env = np.minimum(1.0, t / attack) * np.exp((-2.2 if role == "bass" else -3.0) * t / max(0.12, float(duration) + 0.20))
        release = min(n, max(1, int((0.10 if role == "bass" else 0.08) * sr)))
    if release > 1:
        env[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return env.astype(np.float32)


def _synth_source_note(note: PianoNote, sr: int) -> np.ndarray:
    pitch = _clamp_midi_note(note.pitch)
    if pitch <= 52:
        role = "bass"
        octave_shift = -12 if pitch > 40 else 0
        gain = 0.24
        tail = 0.18
    elif pitch <= 76:
        role = "pad"
        octave_shift = 0
        gain = 0.12
        tail = 0.35
    else:
        role = "lead"
        octave_shift = -12
        gain = 0.10
        tail = 0.15
    dur = max(0.05, float(note.duration) + tail)
    n = max(1, int(round(dur * sr)))
    t = np.arange(n, dtype=np.float32) / float(sr)
    freq = 440.0 * (2.0 ** ((_clamp_midi_note(pitch + octave_shift) - 69) / 12.0))
    vel = float(_clamp_velocity(note.velocity)) / 127.0
    env = _source_note_envelope(n, sr, float(note.duration), role)
    if role == "bass":
        sig = 0.85 * np.sin(2.0 * np.pi * freq * t) + 0.30 * np.sin(2.0 * np.pi * freq * 2.0 * t)
        click = np.sin(2.0 * np.pi * min(2400.0, freq * 12.0) * t) * np.exp(-90.0 * t)
        sig = sig + 0.04 * click
    elif role == "pad":
        sig = (
            0.55 * np.sin(2.0 * np.pi * freq * t)
            + 0.36 * np.sin(2.0 * np.pi * freq * 1.005 * t + 0.5)
            + 0.18 * np.sin(2.0 * np.pi * freq * 2.0 * t)
            + 0.08 * np.sin(2.0 * np.pi * freq * 3.0 * t)
        )
    else:
        sig = (
            0.72 * np.sin(2.0 * np.pi * freq * t)
            + 0.24 * np.sin(2.0 * np.pi * freq * 2.0 * t)
            + 0.12 * np.sin(2.0 * np.pi * freq * 5.0 * t)
        )
    return (sig * env * (0.18 + 0.82 * vel) * gain).astype(np.float32)


def render_preview_wav(
    arrangement: PianoArrangement,
    out_wav: Path,
    *,
    sample_rate: int = 22050,
    normalize_peak: float = 0.95,
) -> Dict[str, Any]:
    sr = int(sample_rate)
    out_wav = Path(out_wav)
    audio = arrangement_to_preview_audio(arrangement, sample_rate=sr, normalize_peak=normalize_peak)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio, sr)
    return {
        "out_wav": str(out_wav),
        "sample_rate": sr,
        "duration_seconds": float(audio.shape[0] / float(sr)),
        "preview_renderer": "dggr_numpy_piano_preview_v1",
    }


def arrangement_to_preview_audio(
    arrangement: PianoArrangement,
    *,
    sample_rate: int = 22050,
    normalize_peak: float = 0.95,
) -> np.ndarray:
    """Render a deterministic piano-like waveform from arrangement notes."""

    sr = int(sample_rate)
    total_duration = max(float(arrangement.duration), max((n.start + n.duration for n in arrangement.notes), default=0.0)) + 0.6
    audio = np.zeros(max(1, int(math.ceil(total_duration * sr))), dtype=np.float32)

    for note in arrangement.notes:
        start = max(0, int(round(float(note.start) * sr)))
        sig = _synth_note(note, sr)
        end = min(audio.shape[0], start + sig.shape[0])
        if end > start:
            audio[start:end] += sig[: end - start]

    if audio.size:
        # Soft compression and level normalization for the "oomph" preview.
        audio = np.tanh(audio * 1.35).astype(np.float32)
        peak = float(np.max(np.abs(audio)) + 1e-8)
        audio = audio / peak * float(max(0.1, min(0.99, normalize_peak)))
    return audio.astype(np.float32)


def arrangement_to_source_preview_audio(
    arrangement: PianoArrangement,
    *,
    mode: str = "piano",
    sample_rate: int = 22050,
    normalize_peak: float = 0.95,
) -> np.ndarray:
    """Render deterministic conditioning audio from symbolic targets.

    ``piano`` preserves the historical cache behavior. ``ensemble`` keeps the
    MIDI note content but changes the timbre and register roles, which creates
    a synthetic non-piano source paired with the same piano-roll target.
    """

    raw_mode = str(mode or "piano").lower()
    if raw_mode in {"piano", "preview", "piano_preview"}:
        return arrangement_to_preview_audio(arrangement, sample_rate=int(sample_rate), normalize_peak=float(normalize_peak))
    if raw_mode != "ensemble":
        raise ValueError(f"Unknown source preview mode: {mode}")

    sr = int(sample_rate)
    total_duration = max(float(arrangement.duration), max((n.start + n.duration for n in arrangement.notes), default=0.0)) + 0.8
    audio = np.zeros(max(1, int(math.ceil(total_duration * sr))), dtype=np.float32)
    for note in arrangement.notes:
        start = max(0, int(round(float(note.start) * sr)))
        sig = _synth_source_note(note, sr)
        end = min(audio.shape[0], start + sig.shape[0])
        if end > start:
            audio[start:end] += sig[: end - start]
    if audio.size:
        audio = np.tanh(audio * 1.20).astype(np.float32)
        peak = float(np.max(np.abs(audio)) + 1e-8)
        audio = audio / peak * float(max(0.1, min(0.99, normalize_peak)))
    return audio.astype(np.float32)


def arrangement_to_dict(arrangement: PianoArrangement) -> Dict[str, Any]:
    return {
        "tempo_bpm": float(arrangement.tempo_bpm),
        "duration": float(arrangement.duration),
        "notes": [asdict(n) for n in arrangement.notes],
        "sustain": [asdict(e) for e in arrangement.sustain],
        "metadata": arrangement.metadata,
    }


def write_arrangement_bundle(
    arrangement: PianoArrangement,
    *,
    out_stem: Path,
    render_wav: bool = True,
) -> Dict[str, Any]:
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    midi_meta = write_midi(arrangement, out_stem.with_suffix(".mid"))
    json_path = out_stem.with_suffix(".json")
    _write_json(json_path, arrangement_to_dict(arrangement))
    wav_meta: Dict[str, Any] = {}
    if render_wav:
        wav_meta = render_preview_wav(arrangement, out_stem.with_suffix(".wav"))
    return {
        "out_stem": str(out_stem),
        "json": str(json_path),
        "midi": midi_meta,
        "wav": wav_meta,
        "notes": int(len(arrangement.notes)),
    }


__all__ = [
    "PianoArrangement",
    "PianoNote",
    "SustainEvent",
    "arrangement_to_dict",
    "arrangement_to_preview_audio",
    "arrangement_to_source_preview_audio",
    "render_preview_wav",
    "write_arrangement_bundle",
    "write_midi",
]
