from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .piano_arranger_cache import DIFFUSION_SR, extract_source_condition, load_audio_for_cache
from .piano_arranger_models import PianoRollGenerator, PianoRollModelConfig
from .piano_arranger_render import PianoArrangement, PianoNote, SustainEvent, write_arrangement_bundle
from .piano_arranger_train import PianoDecodeConfig, prediction_to_arrangement


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIANO_INFER_OUTPUT_DIR = REPO_ROOT / "saves2" / "piano_arranger" / "outputs" / "model_infer"


@dataclass(frozen=True)
class PianoInferenceConfig:
    checkpoint: Path
    source_audio: Path
    out_stem: Path = Path("")
    seconds: float = 30.0
    max_frames: int = 256
    frame_hz: float = 25.0
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
    device: str = "auto"
    render_wav: bool = True


@dataclass(frozen=True)
class PianoChunkedInferenceConfig(PianoInferenceConfig):
    chunk_seconds: float = 12.0
    chunk_hop_seconds: float = 0.0
    section_profile: str = "flat"


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _device(raw: str) -> torch.device:
    if str(raw).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(raw))


def _default_out_stem(source_audio: Path) -> Path:
    return DEFAULT_PIANO_INFER_OUTPUT_DIR / f"{Path(source_audio).stem}__model_piano"


def _load_model(checkpoint: Path, device: torch.device) -> tuple[PianoRollGenerator, Dict[str, Any], PianoRollModelConfig]:
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model_cfg_raw = payload.get("model_cfg", {})
    model_cfg = PianoRollModelConfig(
        in_channels=int(model_cfg_raw.get("in_channels", 17)),
        hidden_channels=int(model_cfg_raw.get("hidden_channels", 96)),
        n_keys=int(model_cfg_raw.get("n_keys", 88)),
        n_blocks=int(model_cfg_raw.get("n_blocks", 6)),
        dropout=float(model_cfg_raw.get("dropout", 0.0)),
        architecture=str(model_cfg_raw.get("architecture", "conv1d")),
        key_embed_dim=int(model_cfg_raw.get("key_embed_dim", 32)),
    )
    model = PianoRollGenerator(model_cfg).to(device)
    load_info = model.load_state_dict(payload["model"], strict=False)
    payload["_load_missing_keys"] = [str(k) for k in load_info.missing_keys]
    payload["_load_unexpected_keys"] = [str(k) for k in load_info.unexpected_keys]
    compat_prefixes = (
        "role_head",
        "role_key_proj",
        "melody_head",
        "melody_key_proj",
        "texture_role_head",
        "texture_role_key_proj",
        "section_role_head",
        "section_role_key_proj",
        "arranger_state_head",
        "arranger_state_key_proj",
        "bass_continuity_head",
        "bass_continuity_key_proj",
        "body_melody_state_head",
        "body_melody_state_key_proj",
        "section_diversity_head",
        "section_diversity_key_proj",
    )
    if any(str(k).startswith(compat_prefixes) for k in load_info.missing_keys):
        missing_modules = {
            str(k).rsplit(".", 1)[0]
            for k in load_info.missing_keys
            if str(k).startswith(compat_prefixes)
        }
        for name, module in model.named_modules():
            if name in missing_modules and hasattr(module, "weight"):
                torch.nn.init.zeros_(module.weight)
                if getattr(module, "bias", None) is not None:
                    torch.nn.init.zeros_(module.bias)
    model.eval()
    return model, payload, model_cfg


def _decode_config_from_inference_cfg(cfg: PianoInferenceConfig, *, min_selected_notes: int | None = None, min_unique_pitches: int | None = None) -> PianoDecodeConfig:
    return PianoDecodeConfig(
        onset_threshold=float(cfg.onset_threshold),
        frame_threshold=float(cfg.frame_threshold),
        max_notes_per_second=float(cfg.max_notes_per_second),
        max_simultaneous_notes=int(cfg.max_simultaneous_notes),
        max_onsets_per_frame=int(cfg.max_onsets_per_frame),
        max_pitch_fraction=float(cfg.max_pitch_fraction),
        max_pitch_class_fraction=float(cfg.max_pitch_class_fraction),
        min_note_duration=float(cfg.min_note_duration),
        max_note_duration=float(cfg.max_note_duration),
        bass_min_note_duration=float(cfg.bass_min_note_duration),
        min_selected_notes=int(cfg.min_selected_notes if min_selected_notes is None else min_selected_notes),
        min_unique_pitches=int(cfg.min_unique_pitches if min_unique_pitches is None else min_unique_pitches),
        require_register_coverage=bool(cfg.require_register_coverage),
        register_coverage_chunk_seconds=float(cfg.register_coverage_chunk_seconds),
        section_bass_repair=bool(cfg.section_bass_repair),
        section_bass_repair_min_coverage=float(cfg.section_bass_repair_min_coverage),
        diversity_fallback_threshold=float(cfg.diversity_fallback_threshold),
        source_onset_guidance_weight=float(cfg.source_onset_guidance_weight),
        source_onset_snap_frames=int(cfg.source_onset_snap_frames),
        source_onset_peak_threshold=float(cfg.source_onset_peak_threshold),
        density_plan_guidance_weight=float(cfg.density_plan_guidance_weight),
        density_plan_snap_frames=int(cfg.density_plan_snap_frames),
        density_plan_peak_threshold=float(cfg.density_plan_peak_threshold),
        event_plan_guidance_weight=float(cfg.event_plan_guidance_weight),
        event_plan_snap_frames=int(cfg.event_plan_snap_frames),
        event_plan_peak_threshold=float(cfg.event_plan_peak_threshold),
        pc_onset_plan_guidance_weight=float(cfg.pc_onset_plan_guidance_weight),
        pc_onset_plan_reserve_threshold=float(cfg.pc_onset_plan_reserve_threshold),
        pc_onset_plan_reserve_max_per_frame=int(cfg.pc_onset_plan_reserve_max_per_frame),
        pc_onset_plan_reserve_min_note_score=float(cfg.pc_onset_plan_reserve_min_note_score),
        pc_onset_plan_select_reserve_fraction=float(cfg.pc_onset_plan_select_reserve_fraction),
        pc_onset_plan_assign_threshold=float(cfg.pc_onset_plan_assign_threshold),
        pc_onset_plan_assign_fraction=float(cfg.pc_onset_plan_assign_fraction),
        pc_onset_plan_assign_window_frames=int(cfg.pc_onset_plan_assign_window_frames),
        pc_onset_plan_assign_min_note_score=float(cfg.pc_onset_plan_assign_min_note_score),
        pc_onset_plan_assign_source_weight=float(cfg.pc_onset_plan_assign_source_weight),
        pc_onset_plan_assign_event_weight=float(cfg.pc_onset_plan_assign_event_weight),
        pc_onset_plan_assign_distance_penalty=float(cfg.pc_onset_plan_assign_distance_penalty),
        source_chroma_guidance_weight=float(cfg.source_chroma_guidance_weight),
        harmonic_plan_guidance_weight=float(cfg.harmonic_plan_guidance_weight),
        chord_plan_guidance_weight=float(cfg.chord_plan_guidance_weight),
        bass_plan_guidance_weight=float(cfg.bass_plan_guidance_weight),
        voicing_plan_guidance_weight=float(cfg.voicing_plan_guidance_weight),
        section_diversity_guidance_weight=float(cfg.section_diversity_guidance_weight),
        section_diversity_reserve_fraction=float(cfg.section_diversity_reserve_fraction),
        section_diversity_reserve_min_note_score=float(cfg.section_diversity_reserve_min_note_score),
        section_diversity_unique_weight=float(cfg.section_diversity_unique_weight),
        section_diversity_pc_weight=float(cfg.section_diversity_pc_weight),
        section_diversity_range_weight=float(cfg.section_diversity_range_weight),
        section_diversity_onset_weight=float(cfg.section_diversity_onset_weight),
        section_diversity_section_seconds=float(cfg.section_diversity_section_seconds),
        source_energy_velocity_weight=float(cfg.source_energy_velocity_weight),
        density_plan_velocity_weight=float(cfg.density_plan_velocity_weight),
    )


def _resolved_frames_and_rate(cfg: PianoInferenceConfig, payload: Dict[str, Any]) -> tuple[int, float]:
    max_frames = int(cfg.max_frames)
    frame_hz = float(cfg.frame_hz)
    cache_meta = payload.get("cache_meta", {})
    if int(max_frames) <= 0:
        max_frames = int(cache_meta.get("max_frames", 256))
    if frame_hz <= 0:
        frame_hz = float(cache_meta.get("frame_hz", 25.0))
    return int(max_frames), float(frame_hz)


def _section_profile_controls(profile: str, chunk_idx: int, n_chunks: int) -> Dict[str, float]:
    raw = str(profile or "flat").lower()
    if raw in {"", "none", "flat"}:
        return {
            "density_multiplier": 1.0,
            "unique_multiplier": 1.0,
            "velocity_multiplier": 1.0,
            "register_chunk_multiplier": 1.0,
        }
    if raw != "arc":
        raise ValueError(f"Unknown section profile: {profile}")
    if int(n_chunks) <= 1:
        center = 1.0
    else:
        pos = float(chunk_idx) / float(max(1, int(n_chunks) - 1))
        center = 1.0 - min(1.0, abs(pos - 0.5) * 2.0)
    return {
        "density_multiplier": 0.80 + 0.45 * center,
        "unique_multiplier": 1.00 + 0.25 * center,
        "velocity_multiplier": 0.85 + 0.35 * center,
        "register_chunk_multiplier": 1.25 - 0.30 * center,
    }


def _repair_combined_section_bass(
    notes: list[PianoNote],
    *,
    duration: float,
    section_seconds: float,
    frame_hz: float,
    min_coverage: float,
) -> tuple[list[PianoNote], int]:
    if not notes:
        return notes, 0
    section_len = max(0.5, float(section_seconds))
    min_cov = max(0.0, float(min_coverage))
    repaired = list(notes)
    repairs = 0
    for section_start in np.arange(0.0, max(0.0, float(duration)), section_len):
        section_end = min(float(duration), float(section_start) + section_len)
        if section_end <= section_start:
            continue
        active_notes = [
            note
            for note in repaired
            if float(note.start) < section_end and float(note.start) + float(note.duration) > section_start
        ]
        if not active_notes:
            continue
        repair_frame_hz = max(1.0, float(frame_hz))
        active_span = np.zeros((max(1, int(round((section_end - section_start) * repair_frame_hz))),), dtype=np.bool_)
        bass_span = np.zeros_like(active_span)
        for note in active_notes:
            lo = max(0, int(round((float(note.start) - section_start) * repair_frame_hz)))
            hi = min(active_span.size, int(round((float(note.start) + float(note.duration) - section_start) * repair_frame_hz)))
            if hi <= lo:
                hi = min(active_span.size, lo + 1)
            active_span[lo:hi] = True
            if int(note.pitch) <= 52:
                bass_span[lo:hi] = True
        if bool(np.any(active_span)) and float(np.mean(bass_span[active_span])) >= min_cov:
            continue
        pitch_class_counts = np.zeros((12,), dtype=np.float32)
        for note in active_notes:
            weight = max(0.05, min(float(note.duration), section_end - section_start))
            pitch_class_counts[int(note.pitch) % 12] += float(weight)
        pc = int(np.argmax(pitch_class_counts))
        bass_pitch = 36 + pc
        while bass_pitch > 48:
            bass_pitch -= 12
        while bass_pitch < 28:
            bass_pitch += 12
        section_velocity = int(np.median([int(note.velocity) for note in active_notes]))
        repair_duration = max(0.45, min(section_end - section_start, section_len * max(min_cov, 0.12)))
        repaired.append(
            PianoNote(
                start=float(section_start),
                duration=float(repair_duration),
                pitch=int(bass_pitch),
                velocity=max(55, min(105, section_velocity - 4)),
            )
        )
        repairs += 1
    return sorted(repaired, key=lambda n: (float(n.start), int(n.pitch))), repairs


def _repair_combined_section_diversity(
    notes: list[PianoNote],
    *,
    duration: float,
    section_seconds: float,
    frame_hz: float,
    min_unique_pitches: int,
    min_chord_frame: float,
    max_notes: int,
) -> tuple[list[PianoNote], int]:
    if not notes:
        return notes, 0
    section_len = max(0.5, float(section_seconds))
    repair_frame_hz = max(1.0, float(frame_hz))
    min_unique = max(0, int(min_unique_pitches))
    min_chord = max(0.0, float(min_chord_frame))
    max_add = max(0, int(max_notes))
    if max_add <= 0 or (min_unique <= 0 and min_chord <= 0.0):
        return notes, 0

    repaired = list(notes)
    repairs = 0
    for section_start in np.arange(0.0, max(0.0, float(duration)), section_len):
        section_end = min(float(duration), float(section_start) + section_len)
        if section_end <= section_start:
            continue
        active_notes = [
            note
            for note in repaired
            if float(note.start) < section_end and float(note.start) + float(note.duration) > section_start
        ]
        if len(active_notes) < 8:
            continue
        pitches = np.asarray([int(note.pitch) for note in active_notes], dtype=np.int32)
        unique_pitches = int(np.unique(pitches).size) if pitches.size else 0
        n_frames = max(1, int(round((section_end - section_start) * repair_frame_hz)))
        poly = np.zeros((n_frames,), dtype=np.float32)
        for note in active_notes:
            lo = max(0, int(round((float(note.start) - section_start) * repair_frame_hz)))
            hi = min(n_frames, int(round((float(note.start) + float(note.duration) - section_start) * repair_frame_hz)))
            if hi <= lo:
                hi = min(n_frames, lo + 1)
            poly[lo:hi] += 1.0
        chord_frame = float(np.mean(poly >= 3.0)) if poly.size else 0.0
        if unique_pitches >= min_unique and chord_frame >= min_chord:
            continue

        pc_counts = np.zeros((12,), dtype=np.float32)
        existing_pitches = {int(p) for p in pitches.tolist()}
        for note in active_notes:
            weight = max(0.05, min(float(note.duration), section_end - section_start))
            pc_counts[int(note.pitch) % 12] += float(weight)
        root_pc = int(np.argmax(pc_counts))
        chord_pcs = [root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12, (root_pc + 2) % 12, (root_pc + 9) % 12]
        median_velocity = int(np.median([int(note.velocity) for note in active_notes]))
        unique_gap = max(0, min_unique - unique_pitches)
        chord_gap_notes = 3 if chord_frame < min_chord else 0
        add_count = min(max_add, max(1, unique_gap, chord_gap_notes))
        added = 0
        for idx, pc in enumerate(chord_pcs):
            if added >= add_count:
                break
            base = 60 if idx < 3 else 72
            pitch = base + int(pc)
            while pitch > 84:
                pitch -= 12
            while pitch < 48:
                pitch += 12
            if pitch in existing_pitches:
                pitch += 12 if pitch <= 72 else -12
            if pitch in existing_pitches or pitch < 36 or pitch > 96:
                continue
            start = float(section_start) + min(float(section_end - section_start) * (0.12 + 0.18 * added), 1.1)
            dur = max(0.45, min(1.25, float(section_end) - start))
            if dur <= 0.05:
                continue
            repaired.append(
                PianoNote(
                    start=float(start),
                    duration=float(dur),
                    pitch=int(pitch),
                    velocity=max(58, min(104, median_velocity - 2 + 3 * added)),
                )
            )
            existing_pitches.add(int(pitch))
            added += 1
        if added:
            repairs += int(added)
    return sorted(repaired, key=lambda n: (float(n.start), int(n.pitch))), repairs


def infer_piano_arrangement(cfg: PianoInferenceConfig) -> Dict[str, Any]:
    device = _device(str(cfg.device))
    checkpoint = Path(cfg.checkpoint)
    model, payload, _model_cfg = _load_model(checkpoint, device)
    max_frames, frame_hz = _resolved_frames_and_rate(cfg, payload)

    y = load_audio_for_cache(Path(cfg.source_audio), seconds=float(cfg.seconds), sr=DIFFUSION_SR)
    source = extract_source_condition(y, sr=DIFFUSION_SR, frame_hz=frame_hz, max_frames=max_frames)
    with torch.no_grad():
        pred = model(torch.from_numpy(np.asarray(source, dtype=np.float32))[None, :, :].to(device))
    duration = float(len(y) / float(DIFFUSION_SR))
    arrangement = prediction_to_arrangement(
        pred,
        frame_hz=frame_hz,
        duration=duration,
        onset_threshold=float(cfg.onset_threshold),
        frame_threshold=float(cfg.frame_threshold),
        decode_config=_decode_config_from_inference_cfg(cfg),
        metadata={
            "source_audio": str(cfg.source_audio),
            "checkpoint": str(checkpoint),
            "source": "model_inference",
            "checkpoint_missing_keys": payload.get("_load_missing_keys", []),
            "checkpoint_unexpected_keys": payload.get("_load_unexpected_keys", []),
        },
        source_condition=source,
    )
    out_stem = Path(cfg.out_stem) if str(cfg.out_stem) else _default_out_stem(Path(cfg.source_audio))
    bundle = write_arrangement_bundle(arrangement, out_stem=out_stem, render_wav=bool(cfg.render_wav))
    summary = {
        "checkpoint": str(checkpoint),
        "source_audio": str(cfg.source_audio),
        "out_stem": str(out_stem),
        "notes": int(len(arrangement.notes)),
        "duration": float(duration),
        "frame_hz": frame_hz,
        "max_frames": int(max_frames),
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
        "device": str(device),
        "bundle": bundle,
    }
    _write_json(out_stem.with_name(out_stem.name + ".summary.json"), summary)
    return summary


def infer_piano_arrangement_chunked(cfg: PianoChunkedInferenceConfig) -> Dict[str, Any]:
    device = _device(str(cfg.device))
    checkpoint = Path(cfg.checkpoint)
    model, payload, _model_cfg = _load_model(checkpoint, device)
    max_frames, frame_hz = _resolved_frames_and_rate(cfg, payload)
    chunk_seconds = max(0.5, float(cfg.chunk_seconds))
    hop_seconds = float(cfg.chunk_hop_seconds) if float(cfg.chunk_hop_seconds) > 0 else chunk_seconds
    hop_seconds = max(0.25, hop_seconds)

    y = load_audio_for_cache(Path(cfg.source_audio), seconds=float(cfg.seconds), sr=DIFFUSION_SR)
    total_duration = float(len(y) / float(DIFFUSION_SR))
    starts: List[float] = []
    cursor = 0.0
    while cursor < total_duration - 1e-6:
        starts.append(float(cursor))
        cursor += hop_seconds
    n_chunks = int(len(starts))
    notes: List[PianoNote] = []
    sustain: List[SustainEvent] = []
    chunk_rows: List[Dict[str, Any]] = []
    total_min_selected = max(0, int(cfg.min_selected_notes))
    total_min_unique = max(0, int(cfg.min_unique_pitches))

    for chunk_idx, start_sec in enumerate(starts):
        controls = _section_profile_controls(str(cfg.section_profile), int(chunk_idx), n_chunks)
        start_sample = int(round(float(start_sec) * float(DIFFUSION_SR)))
        end_sample = min(len(y), start_sample + int(round(chunk_seconds * float(DIFFUSION_SR))))
        if end_sample <= start_sample + int(0.25 * DIFFUSION_SR):
            continue
        chunk_y = y[start_sample:end_sample].astype(np.float32, copy=False)
        chunk_duration = float(len(chunk_y) / float(DIFFUSION_SR))
        chunk_frames = max(1, int(round(chunk_duration * frame_hz)))
        use_frames = max(1, min(int(max_frames), chunk_frames))
        source = extract_source_condition(chunk_y, sr=DIFFUSION_SR, frame_hz=frame_hz, max_frames=use_frames)
        with torch.no_grad():
            pred = model(torch.from_numpy(np.asarray(source, dtype=np.float32))[None, :, :].to(device))
        scale = chunk_duration / max(1e-6, total_duration)
        chunk_min_selected = max(8, int(round(total_min_selected * scale * float(controls["density_multiplier"])))) if total_min_selected > 0 else 0
        chunk_min_unique = max(8, int(round(total_min_unique * scale * float(controls["unique_multiplier"])))) if total_min_unique > 0 else 0
        chunk_cfg = PianoChunkedInferenceConfig(
            **{
                **cfg.__dict__,
                "source_energy_velocity_weight": float(cfg.source_energy_velocity_weight) * float(controls["velocity_multiplier"]),
                "density_plan_velocity_weight": float(cfg.density_plan_velocity_weight) * float(controls["velocity_multiplier"]),
                "register_coverage_chunk_seconds": float(cfg.register_coverage_chunk_seconds) * float(controls["register_chunk_multiplier"]),
            }
        )
        arrangement = prediction_to_arrangement(
            pred,
            frame_hz=frame_hz,
            duration=chunk_duration,
            onset_threshold=float(cfg.onset_threshold),
            frame_threshold=float(cfg.frame_threshold),
            decode_config=_decode_config_from_inference_cfg(
                chunk_cfg,
                min_selected_notes=chunk_min_selected,
                min_unique_pitches=chunk_min_unique,
            ),
            metadata={
                "source_audio": str(cfg.source_audio),
                "checkpoint": str(checkpoint),
                "source": "model_chunked_inference",
                "chunk_index": int(chunk_idx),
                "chunk_start": float(start_sec),
                "chunk_duration": float(chunk_duration),
                "section_profile": str(cfg.section_profile),
                "section_profile_controls": controls,
                "checkpoint_missing_keys": payload.get("_load_missing_keys", []),
                "checkpoint_unexpected_keys": payload.get("_load_unexpected_keys", []),
            },
            source_condition=source,
        )
        keep_until = hop_seconds if chunk_idx < len(starts) - 1 else chunk_duration + 1e-6
        kept = 0
        for note in arrangement.notes:
            if float(note.start) >= keep_until:
                continue
            notes.append(
                PianoNote(
                    start=float(note.start) + float(start_sec),
                    duration=float(note.duration),
                    pitch=int(note.pitch),
                    velocity=int(note.velocity),
                )
            )
            kept += 1
        for ev in arrangement.sustain:
            if 0.0 <= float(ev.time) < keep_until:
                sustain.append(SustainEvent(time=float(ev.time) + float(start_sec), value=int(ev.value)))
        chunk_rows.append(
            {
                "chunk_index": int(chunk_idx),
                "start": float(start_sec),
                "duration": float(chunk_duration),
                "frames": int(use_frames),
                "decoded_notes": int(len(arrangement.notes)),
                "kept_notes": int(kept),
                "min_selected_notes": int(chunk_min_selected),
                "min_unique_pitches": int(chunk_min_unique),
                "section_profile_controls": controls,
                "metadata": arrangement.metadata,
            }
        )

    section_diversity_guidance_chunks = 0
    section_diversity_reserved_notes = 0
    for row in chunk_rows:
        guidance = dict(row.get("metadata", {})).get("decode_section_diversity_guidance", {})
        if bool(guidance.get("available", False)):
            section_diversity_guidance_chunks += 1
        section_diversity_reserved_notes += int(guidance.get("reserved_notes", 0) or 0)

    post_chunk_section_bass_repairs = 0
    post_chunk_section_diversity_repairs = 0
    combined_notes = sorted(notes, key=lambda n: (float(n.start), int(n.pitch)))
    if bool(cfg.section_bass_repair):
        combined_notes, post_chunk_section_bass_repairs = _repair_combined_section_bass(
            combined_notes,
            duration=float(total_duration),
            section_seconds=float(chunk_seconds),
            frame_hz=float(frame_hz),
            min_coverage=float(cfg.section_bass_repair_min_coverage),
        )
    if bool(cfg.section_diversity_repair):
        combined_notes, post_chunk_section_diversity_repairs = _repair_combined_section_diversity(
            combined_notes,
            duration=float(total_duration),
            section_seconds=float(chunk_seconds),
            frame_hz=float(frame_hz),
            min_unique_pitches=int(cfg.section_diversity_repair_min_unique_pitches),
            min_chord_frame=float(cfg.section_diversity_repair_min_chord_frame),
            max_notes=int(cfg.section_diversity_repair_max_notes),
        )
    combined = PianoArrangement(
        notes=combined_notes,
        tempo_bpm=120.0,
        duration=float(total_duration),
        sustain=sorted(sustain, key=lambda ev: float(ev.time)),
        metadata={
            "source_audio": str(cfg.source_audio),
            "checkpoint": str(checkpoint),
            "source": "model_chunked_inference",
            "chunk_seconds": float(chunk_seconds),
            "chunk_hop_seconds": float(hop_seconds),
            "section_profile": str(cfg.section_profile),
            "post_chunk_section_bass_repairs": int(post_chunk_section_bass_repairs),
            "post_chunk_section_diversity_repairs": int(post_chunk_section_diversity_repairs),
            "section_diversity_guidance_chunks": int(section_diversity_guidance_chunks),
            "section_diversity_reserved_notes": int(section_diversity_reserved_notes),
            "chunks": chunk_rows,
        },
    )
    out_stem = Path(cfg.out_stem) if str(cfg.out_stem) else _default_out_stem(Path(cfg.source_audio)).with_name(f"{Path(cfg.source_audio).stem}__model_piano_chunked")
    bundle = write_arrangement_bundle(combined, out_stem=out_stem, render_wav=bool(cfg.render_wav))
    summary = {
        "checkpoint": str(checkpoint),
        "source_audio": str(cfg.source_audio),
        "out_stem": str(out_stem),
        "notes": int(len(combined.notes)),
        "duration": float(total_duration),
        "frame_hz": float(frame_hz),
        "max_frames": int(max_frames),
        "chunk_seconds": float(chunk_seconds),
        "chunk_hop_seconds": float(hop_seconds),
        "section_profile": str(cfg.section_profile),
        "post_chunk_section_bass_repairs": int(post_chunk_section_bass_repairs),
        "post_chunk_section_diversity_repairs": int(post_chunk_section_diversity_repairs),
        "section_diversity_guidance_chunks": int(section_diversity_guidance_chunks),
        "section_diversity_reserved_notes": int(section_diversity_reserved_notes),
        "section_diversity_repair": bool(cfg.section_diversity_repair),
        "section_diversity_repair_min_unique_pitches": int(cfg.section_diversity_repair_min_unique_pitches),
        "section_diversity_repair_min_chord_frame": float(cfg.section_diversity_repair_min_chord_frame),
        "section_diversity_repair_max_notes": int(cfg.section_diversity_repair_max_notes),
        "section_diversity_guidance_weight": float(cfg.section_diversity_guidance_weight),
        "section_diversity_reserve_fraction": float(cfg.section_diversity_reserve_fraction),
        "section_diversity_reserve_min_note_score": float(cfg.section_diversity_reserve_min_note_score),
        "section_diversity_unique_weight": float(cfg.section_diversity_unique_weight),
        "section_diversity_pc_weight": float(cfg.section_diversity_pc_weight),
        "section_diversity_range_weight": float(cfg.section_diversity_range_weight),
        "section_diversity_onset_weight": float(cfg.section_diversity_onset_weight),
        "section_diversity_section_seconds": float(cfg.section_diversity_section_seconds),
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
        "chunks": chunk_rows,
        "device": str(device),
        "bundle": bundle,
    }
    _write_json(out_stem.with_name(out_stem.name + ".summary.json"), summary)
    return summary


__all__ = [
    "DEFAULT_PIANO_INFER_OUTPUT_DIR",
    "PianoChunkedInferenceConfig",
    "PianoInferenceConfig",
    "infer_piano_arrangement",
    "infer_piano_arrangement_chunked",
]
