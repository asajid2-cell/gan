from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .piano_arranger_cache import (
    DEFAULT_PIANO_CACHE_DIR,
    PIANO_MIN_MIDI,
    SOURCE_FEATURE_NAMES,
    arranger_state_targets_from_roll,
    bass_continuity_targets_from_roll,
    body_melody_state_targets_from_roll,
    event_plan_targets_from_roll,
    hierarchy_targets_from_roll,
    melody_targets_from_roll,
    musical_plan_targets_from_roll,
    pitch_class_onset_targets_from_roll,
    role_fullness_targets_from_roll,
    section_diversity_targets_from_roll,
    section_role_targets_from_roll,
    texture_role_targets_from_roll,
)
from .piano_arranger_eval import (
    evaluate_arrangement_dict,
    section_report_arrangement_dict,
    target_midi_alignment_metrics,
    target_midi_alignment_warnings,
)
from .piano_arranger_models import PianoRollGenerator, PianoRollModelConfig, piano_roll_loss
from .piano_arranger_render import PianoArrangement, PianoNote, SustainEvent, arrangement_to_dict, write_arrangement_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIANO_RUN_ROOT = REPO_ROOT / "saves2" / "piano_arranger" / "runs"
AUDIO_SOURCE_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".aiff", ".aif"}


@dataclass(frozen=True)
class PianoDecodeConfig:
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
    diversity_fallback_threshold: float = 0.05
    fallback_top_events: int = 24
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


@dataclass(frozen=True)
class PianoTrainConfig:
    cache_dir: Path = DEFAULT_PIANO_CACHE_DIR
    out_root: Path = DEFAULT_PIANO_RUN_ROOT
    run_name: str = ""
    warm_start_checkpoint: Path | None = None
    epochs: int = 2
    batch_size: int = 2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    density_loss_weight: float = 0.35
    chroma_loss_weight: float = 0.35
    pitch_usage_loss_weight: float = 0.35
    hierarchy_loss_weight: float = 0.25
    musical_plan_loss_weight: float = 0.0
    event_plan_loss_weight: float = 0.0
    pc_onset_plan_loss_weight: float = 0.0
    pc_onset_f1_loss_weight: float = 0.0
    pc_onset_alignment_loss_weight: float = 0.0
    role_plan_loss_weight: float = 0.0
    texture_balance_loss_weight: float = 0.0
    melody_plan_loss_weight: float = 0.0
    melody_balance_loss_weight: float = 0.0
    texture_role_plan_loss_weight: float = 0.0
    texture_role_balance_loss_weight: float = 0.0
    section_role_plan_loss_weight: float = 0.0
    section_role_balance_loss_weight: float = 0.0
    arranger_state_plan_loss_weight: float = 0.0
    bass_continuity_plan_loss_weight: float = 0.0
    body_melody_state_plan_loss_weight: float = 0.0
    body_melody_state_balance_loss_weight: float = 0.0
    section_diversity_plan_loss_weight: float = 0.0
    section_diversity_balance_loss_weight: float = 0.0
    anti_collapse_loss_weight: float = 0.0
    source_onset_loss_weight: float = 0.0
    source_chroma_loss_weight: float = 0.0
    harmonic_plan_loss_weight: float = 0.0
    hidden_channels: int = 96
    n_blocks: int = 6
    dropout: float = 0.05
    model_architecture: str = "conv1d"
    key_embed_dim: int = 32
    max_batches_per_epoch: int = 0
    sample_every: int = 1
    sample_count: int = 1
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
    sample_eval: bool = True
    sample_source_eval: bool = True
    sample_target_eval: bool = True
    sample_section_eval: bool = True
    sample_section_seconds: float = 4.0
    min_target_global_chroma_cosine: float = 0.20
    min_target_active_chroma_cosine: float = 0.20
    min_target_onset_correlation: float = 0.02
    min_target_onset_frame_f1: float = 0.0
    min_target_pitch_class_onset_f1: float = 0.0
    min_target_note_count_ratio: float = 0.0
    max_target_note_count_ratio: float = 0.0
    sample_score_pass_weight: float = 1000.0
    sample_score_warning_penalty: float = 10.0
    sample_score_source_active_weight: float = 1.0
    sample_score_source_onset_weight: float = 0.25
    sample_score_target_active_weight: float = 1.0
    sample_score_target_onset_weight: float = 0.25
    sample_score_role_balance_weight: float = 25.0
    sample_score_chord_frame_target: float = 0.80
    sample_score_melody_coverage_target: float = 0.30
    sample_score_bass_coverage_min: float = 0.35
    sample_score_bass_coverage_max: float = 0.85
    sample_score_polyphony_target: float = 5.5
    sample_score_rms_target: float = 0.18
    sample_score_quality_penalty_weight: float = 25.0
    sample_score_min_notes_per_second: float = 6.0
    sample_score_min_section_notes: float = 12.0
    sample_score_min_section_unique_pitches: float = 8.0
    sample_score_min_section_chord_frame: float = 0.40
    sample_score_min_section_fullness: float = 0.70
    sample_score_max_single_pitch_class_fraction: float = 0.32
    sample_score_min_mid_note_fraction: float = 0.20
    sample_score_max_high_note_fraction: float = 0.75
    device: str = "auto"
    seed: int = 328


class PianoRollCacheDataset(Dataset):
    def __init__(self, cache_dir: Path) -> None:
        cache = Path(cache_dir)
        self.cache_dir = cache
        self.source = np.load(cache / "source_condition.npy", mmap_mode="r")
        self.onset = np.load(cache / "target_onset.npy", mmap_mode="r")
        self.frame = np.load(cache / "target_frame.npy", mmap_mode="r")
        self.velocity = np.load(cache / "target_velocity.npy", mmap_mode="r")
        self.pedal = np.load(cache / "target_pedal.npy", mmap_mode="r")
        self.density = np.load(cache / "target_density.npy", mmap_mode="r") if (cache / "target_density.npy").exists() else None
        self.register = np.load(cache / "target_register.npy", mmap_mode="r") if (cache / "target_register.npy").exists() else None
        self.chord = np.load(cache / "target_chord.npy", mmap_mode="r") if (cache / "target_chord.npy").exists() else None
        self.bass = np.load(cache / "target_bass.npy", mmap_mode="r") if (cache / "target_bass.npy").exists() else None
        self.voicing = np.load(cache / "target_voicing.npy", mmap_mode="r") if (cache / "target_voicing.npy").exists() else None
        self.event = np.load(cache / "target_event.npy", mmap_mode="r") if (cache / "target_event.npy").exists() else None
        self.pc_onset = np.load(cache / "target_pc_onset.npy", mmap_mode="r") if (cache / "target_pc_onset.npy").exists() else None
        self.role = np.load(cache / "target_role.npy", mmap_mode="r") if (cache / "target_role.npy").exists() else None
        self.melody = np.load(cache / "target_melody.npy", mmap_mode="r") if (cache / "target_melody.npy").exists() else None
        self.texture_role = (
            np.load(cache / "target_texture_role.npy", mmap_mode="r")
            if (cache / "target_texture_role.npy").exists()
            else None
        )
        self.section_role = (
            np.load(cache / "target_section_role.npy", mmap_mode="r")
            if (cache / "target_section_role.npy").exists()
            else None
        )
        self.arranger_state = (
            np.load(cache / "target_arranger_state.npy", mmap_mode="r")
            if (cache / "target_arranger_state.npy").exists()
            else None
        )
        self.bass_continuity = (
            np.load(cache / "target_bass_continuity.npy", mmap_mode="r")
            if (cache / "target_bass_continuity.npy").exists()
            else None
        )
        self.body_melody_state = (
            np.load(cache / "target_body_melody_state.npy", mmap_mode="r")
            if (cache / "target_body_melody_state.npy").exists()
            else None
        )
        self.section_diversity = (
            np.load(cache / "target_section_diversity.npy", mmap_mode="r")
            if (cache / "target_section_diversity.npy").exists()
            else None
        )
        self.frame_hz = 25.0
        if (cache / "meta.json").exists():
            try:
                meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
                self.frame_hz = float(meta.get("frame_hz", self.frame_hz))
            except Exception:
                self.frame_hz = 25.0
        self.index = pd.read_csv(cache / "index.csv") if (cache / "index.csv").exists() else pd.DataFrame()

    def __len__(self) -> int:
        return int(self.source.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(idx)
        onset = np.array(self.onset[i], dtype=np.float32, copy=True)
        frame = np.array(self.frame[i], dtype=np.float32, copy=True)
        if self.density is not None and self.register is not None:
            density = np.array(self.density[i], dtype=np.float32, copy=True)
            register = np.array(self.register[i], dtype=np.float32, copy=True)
        else:
            density, register = hierarchy_targets_from_roll(onset, frame)
        if self.chord is not None and self.bass is not None and self.voicing is not None:
            chord = np.array(self.chord[i], dtype=np.float32, copy=True)
            bass = np.array(self.bass[i], dtype=np.float32, copy=True)
            voicing = np.array(self.voicing[i], dtype=np.float32, copy=True)
        else:
            chord, bass, voicing = musical_plan_targets_from_roll(onset, frame)
        if self.event is not None:
            event = np.array(self.event[i], dtype=np.float32, copy=True)
        else:
            event = event_plan_targets_from_roll(onset, frame)
        if self.pc_onset is not None:
            pc_onset = np.array(self.pc_onset[i], dtype=np.float32, copy=True)
        else:
            pc_onset = pitch_class_onset_targets_from_roll(onset)
        if self.role is not None:
            role = np.array(self.role[i], dtype=np.float32, copy=True)
        else:
            role = role_fullness_targets_from_roll(frame, np.array(self.velocity[i], dtype=np.float32, copy=True))
        if self.melody is not None:
            melody = np.array(self.melody[i], dtype=np.float32, copy=True)
        else:
            melody = melody_targets_from_roll(frame, np.array(self.velocity[i], dtype=np.float32, copy=True))
        if self.texture_role is not None:
            texture_role = np.array(self.texture_role[i], dtype=np.float32, copy=True)
        else:
            texture_role = texture_role_targets_from_roll(onset, frame)
        if self.section_role is not None:
            section_role = np.array(self.section_role[i], dtype=np.float32, copy=True)
        else:
            section_role = section_role_targets_from_roll(onset, frame, frame_hz=float(self.frame_hz))
        if self.arranger_state is not None:
            arranger_state = np.array(self.arranger_state[i], dtype=np.float32, copy=True)
        else:
            arranger_state = arranger_state_targets_from_roll(onset, frame, frame_hz=float(self.frame_hz))
        if self.bass_continuity is not None:
            bass_continuity = np.array(self.bass_continuity[i], dtype=np.float32, copy=True)
        else:
            bass_continuity = bass_continuity_targets_from_roll(onset, frame, frame_hz=float(self.frame_hz))
        if self.body_melody_state is not None:
            body_melody_state = np.array(self.body_melody_state[i], dtype=np.float32, copy=True)
        else:
            body_melody_state = body_melody_state_targets_from_roll(onset, frame, frame_hz=float(self.frame_hz))
        if self.section_diversity is not None:
            section_diversity = np.array(self.section_diversity[i], dtype=np.float32, copy=True)
        else:
            section_diversity = section_diversity_targets_from_roll(onset, frame, frame_hz=float(self.frame_hz))
        return {
            "source": torch.from_numpy(np.array(self.source[i], dtype=np.float32, copy=True)),
            "onset": torch.from_numpy(onset),
            "frame": torch.from_numpy(frame),
            "velocity": torch.from_numpy(np.array(self.velocity[i], dtype=np.float32, copy=True)),
            "pedal": torch.from_numpy(np.array(self.pedal[i], dtype=np.float32, copy=True)),
            "density": torch.from_numpy(density),
            "register": torch.from_numpy(register),
            "chord": torch.from_numpy(chord),
            "bass": torch.from_numpy(bass),
            "voicing": torch.from_numpy(voicing),
            "event": torch.from_numpy(event),
            "pc_onset": torch.from_numpy(pc_onset),
            "role": torch.from_numpy(role),
            "melody": torch.from_numpy(melody),
            "texture_role": torch.from_numpy(texture_role),
            "section_role": torch.from_numpy(section_role),
            "arranger_state": torch.from_numpy(arranger_state),
            "bass_continuity": torch.from_numpy(bass_continuity),
            "body_melody_state": torch.from_numpy(body_melody_state),
            "section_diversity": torch.from_numpy(section_diversity),
            "idx": torch.tensor(i, dtype=torch.long),
        }


def _load_warm_start_checkpoint(
    model: PianoRollGenerator,
    checkpoint: Path,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    raw_state = payload.get("model", payload)
    if not isinstance(raw_state, dict):
        raise ValueError(f"Warm-start checkpoint has no model state: {checkpoint}")
    current_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    shape_mismatches: List[Dict[str, Any]] = []
    unexpected: List[str] = []
    for key, value in raw_state.items():
        key = str(key)
        if key not in current_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(current_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(current_state[key].shape),
                }
            )
            continue
        compatible[key] = value
    load_info = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": int(len(compatible)),
        "missing_keys": [str(k) for k in load_info.missing_keys],
        "unexpected_keys": unexpected + [str(k) for k in load_info.unexpected_keys],
        "shape_mismatches": shape_mismatches,
        "source_epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        "source_best_sample_eval_epoch": payload.get("best_sample_eval_epoch") if isinstance(payload, dict) else None,
        "source_best_sample_eval_score": payload.get("best_sample_eval_score") if isinstance(payload, dict) else None,
    }


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _device(raw: str) -> torch.device:
    if str(raw).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(raw))


def _source_audio_for_sample(ds: PianoRollCacheDataset, idx: int) -> Path | None:
    if ds.index.empty or "path" not in ds.index.columns or int(idx) >= len(ds.index):
        return None
    path = Path(str(ds.index.iloc[int(idx)]["path"]))
    if path.suffix.lower() not in AUDIO_SOURCE_SUFFIXES:
        return None
    return path


def _target_midi_for_sample(ds: PianoRollCacheDataset, idx: int) -> Path | None:
    if ds.index.empty or "target_midi" not in ds.index.columns or int(idx) >= len(ds.index):
        return None
    raw = str(ds.index.iloc[int(idx)]["target_midi"])
    if raw == "" or raw.lower() == "nan":
        return None
    path = Path(raw)
    if not path.exists():
        return None
    if path.suffix.lower() not in {".mid", ".midi", ".kar"}:
        return None
    return path


def _new_run_dir(cfg: PianoTrainConfig) -> Path:
    if cfg.run_name:
        return Path(cfg.out_root) / str(cfg.run_name)
    stamp = datetime.now().strftime("piano_roll_%Y%m%d_%H%M%S")
    return Path(cfg.out_root) / stamp


def _prune_pitch_dominance(
    selected: List[Dict[str, Any]],
    *,
    max_pitch_fraction: float,
    max_pitch_class_fraction: float,
    min_keep: int,
) -> List[Dict[str, Any]]:
    kept = list(selected)
    while len(kept) > max(0, int(min_keep)):
        n = len(kept)
        max_pitch = max(1, int(np.floor(n * float(max_pitch_fraction))))
        max_pitch_class = max(1, int(np.floor(n * float(max_pitch_class_fraction))))
        pitch_counts: Dict[int, int] = {}
        pitch_class_counts: Dict[int, int] = {}
        for cand in kept:
            key = int(cand["key"])
            pitch_counts[key] = pitch_counts.get(key, 0) + 1
            pc = int((PIANO_MIN_MIDI + key) % 12)
            pitch_class_counts[pc] = pitch_class_counts.get(pc, 0) + 1
        bad_pitches = {k for k, v in pitch_counts.items() if v > max_pitch}
        bad_pitch_classes = {k for k, v in pitch_class_counts.items() if v > max_pitch_class}
        if not bad_pitches and not bad_pitch_classes:
            return kept
        remove_idx = None
        remove_score = float("inf")
        for i, cand in enumerate(kept):
            if bool(cand.get("protected", False)):
                continue
            key = int(cand["key"])
            pc = int((PIANO_MIN_MIDI + key) % 12)
            if key not in bad_pitches and pc not in bad_pitch_classes:
                continue
            score = float(cand.get("score", 0.0))
            if score < remove_score:
                remove_idx = i
                remove_score = score
        if remove_idx is None:
            for i, cand in enumerate(kept):
                key = int(cand["key"])
                pc = int((PIANO_MIN_MIDI + key) % 12)
                if key not in bad_pitches and pc not in bad_pitch_classes:
                    continue
                score = float(cand.get("score", 0.0))
                if score < remove_score:
                    remove_idx = i
                    remove_score = score
        if remove_idx is None:
            return kept
        kept.pop(remove_idx)
    return kept


def _key_register(key: int) -> str:
    pitch = PIANO_MIN_MIDI + int(key)
    if pitch <= 52:
        return "low"
    if pitch <= 76:
        return "mid"
    return "high"


def _source_onset_from_condition(
    source_condition: np.ndarray | torch.Tensor | None,
    *,
    n_frames: int,
) -> np.ndarray | None:
    if source_condition is None:
        return None
    if isinstance(source_condition, torch.Tensor):
        arr = source_condition.detach().cpu().numpy()
    else:
        arr = np.asarray(source_condition)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        if arr.shape[0] > 12:
            arr = arr[12]
        else:
            arr = np.max(arr, axis=0)
    if arr.ndim != 1 or arr.size == 0:
        return None
    out = np.zeros((int(n_frames),), dtype=np.float32)
    take = min(int(n_frames), int(arr.shape[0]))
    out[:take] = arr[:take]
    lo = float(np.min(out))
    hi = float(np.max(out))
    if hi > lo and (lo < 0.0 or hi > 1.0):
        out = (out - lo) / max(1e-6, hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _source_energy_from_condition(
    source_condition: np.ndarray | torch.Tensor | None,
    *,
    n_frames: int,
) -> np.ndarray | None:
    if source_condition is None:
        return None
    if isinstance(source_condition, torch.Tensor):
        arr = source_condition.detach().cpu().numpy()
    else:
        arr = np.asarray(source_condition)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 2:
        if arr.shape[0] > 13:
            arr = arr[13]
        else:
            arr = np.max(arr, axis=0)
    if arr.ndim != 1 or arr.size == 0:
        return None
    out = np.zeros((int(n_frames),), dtype=np.float32)
    take = min(int(n_frames), int(arr.shape[0]))
    out[:take] = arr[:take]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _source_chroma_from_condition(
    source_condition: np.ndarray | torch.Tensor | None,
    *,
    n_frames: int,
) -> np.ndarray | None:
    if source_condition is None:
        return None
    if isinstance(source_condition, torch.Tensor):
        arr = source_condition.detach().cpu().numpy()
    else:
        arr = np.asarray(source_condition)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 12:
        return None
    raw = np.clip(arr[:12, :], 0.0, 1.0)
    out = np.zeros((12, int(n_frames)), dtype=np.float32)
    take = min(int(n_frames), int(raw.shape[1]))
    out[:, :take] = raw[:, :take]
    out = out / (np.sum(out, axis=0, keepdims=True) + 1e-6)
    return out.astype(np.float32)


def prediction_to_arrangement(
    pred: Dict[str, torch.Tensor],
    *,
    frame_hz: float,
    duration: float,
    onset_threshold: float,
    frame_threshold: float,
    metadata: Dict[str, Any],
    decode_config: PianoDecodeConfig | None = None,
    source_condition: np.ndarray | torch.Tensor | None = None,
) -> PianoArrangement:
    decode = decode_config or PianoDecodeConfig(onset_threshold=float(onset_threshold), frame_threshold=float(frame_threshold))
    onset = torch.sigmoid(pred["onset_logits"])[0].detach().cpu().numpy()
    frame = torch.sigmoid(pred["frame_logits"])[0].detach().cpu().numpy()
    velocity = pred["velocity"][0].detach().cpu().numpy()
    pedal = pred["pedal"][0].detach().cpu().numpy()
    n_keys, n_frames = onset.shape
    source_onset = _source_onset_from_condition(source_condition, n_frames=n_frames)
    source_energy = _source_energy_from_condition(source_condition, n_frames=n_frames)
    source_chroma = _source_chroma_from_condition(source_condition, n_frames=n_frames)
    density_plan = None
    if "density" in pred:
        raw_density = pred["density"][0, 0].detach().cpu().numpy()
        density_plan = np.clip(np.asarray(raw_density, dtype=np.float32)[:n_frames], 0.0, 1.0)
    harmonic_plan = None
    if "harmony" in pred:
        raw_harmony = pred["harmony"][0].detach().cpu().numpy()
        raw_harmony = np.clip(np.asarray(raw_harmony, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
        harmonic_plan = raw_harmony / (np.sum(raw_harmony, axis=0, keepdims=True) + 1e-6)
    checkpoint_missing_keys = [str(k) for k in dict(metadata).get("checkpoint_missing_keys", [])]

    def checkpoint_loaded_head(prefix: str) -> bool:
        return not any(str(k).startswith(str(prefix)) for k in checkpoint_missing_keys)

    chord_plan = None
    if "chord_logits" in pred and checkpoint_loaded_head("chord_head"):
        raw_chord = torch.sigmoid(pred["chord_logits"])[0].detach().cpu().numpy()
        chord_plan = np.clip(np.asarray(raw_chord, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
    bass_plan = None
    if "bass_logits" in pred and checkpoint_loaded_head("bass_head"):
        raw_bass = torch.softmax(pred["bass_logits"][0], dim=0).detach().cpu().numpy()
        bass_plan = np.clip(np.asarray(raw_bass, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
    voicing_plan = None
    if "voicing" in pred and checkpoint_loaded_head("voicing_head"):
        raw_voicing = pred["voicing"][0].detach().cpu().numpy()
        voicing_plan = np.clip(np.asarray(raw_voicing, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
    event_plan = None
    if "event" in pred and checkpoint_loaded_head("event_head"):
        raw_event = pred["event"][0].detach().cpu().numpy()
        event_plan = np.clip(np.asarray(raw_event, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
    pc_onset_plan = None
    if "pc_onset" in pred and checkpoint_loaded_head("pc_onset_head"):
        raw_pc_onset = pred["pc_onset"][0].detach().cpu().numpy()
        pc_onset_plan = np.clip(np.asarray(raw_pc_onset, dtype=np.float32)[:, :n_frames], 0.0, 1.0)
    section_diversity_plan = None
    if "section_diversity" in pred and checkpoint_loaded_head("section_diversity_head"):
        raw_section_diversity = pred["section_diversity"][0].detach().cpu().numpy()
        raw_section_diversity = np.asarray(raw_section_diversity, dtype=np.float32)
        if raw_section_diversity.ndim == 2 and raw_section_diversity.shape[0] >= 4:
            section_diversity_plan = np.clip(raw_section_diversity[:4, :n_frames], 0.0, 1.0)
    event_onset_plan = event_plan[0] if event_plan is not None and event_plan.shape[0] > 0 else None
    event_source_curve = event_onset_plan
    if event_onset_plan is not None and source_onset is not None:
        event_source_curve = np.clip(0.5 * event_onset_plan + 0.5 * source_onset, 0.0, 1.0).astype(np.float32)
    guidance_weight = max(0.0, float(decode.source_onset_guidance_weight))
    snap_frames = max(0, int(decode.source_onset_snap_frames))
    snap_threshold = max(0.0, float(decode.source_onset_peak_threshold))
    density_guidance_weight = max(0.0, float(decode.density_plan_guidance_weight))
    density_snap_frames = max(0, int(decode.density_plan_snap_frames))
    density_snap_threshold = max(0.0, float(decode.density_plan_peak_threshold))
    event_guidance_weight = max(0.0, float(decode.event_plan_guidance_weight))
    event_snap_frames = max(0, int(decode.event_plan_snap_frames))
    event_snap_threshold = max(0.0, float(decode.event_plan_peak_threshold))
    pc_onset_guidance_weight = max(0.0, float(decode.pc_onset_plan_guidance_weight))
    source_chroma_guidance_weight = max(0.0, float(decode.source_chroma_guidance_weight))
    harmonic_guidance_weight = max(0.0, float(decode.harmonic_plan_guidance_weight))
    chord_guidance_weight = max(0.0, float(decode.chord_plan_guidance_weight))
    bass_guidance_weight = max(0.0, float(decode.bass_plan_guidance_weight))
    voicing_guidance_weight = max(0.0, float(decode.voicing_plan_guidance_weight))
    section_diversity_guidance_weight = max(0.0, float(decode.section_diversity_guidance_weight))
    section_diversity_reserve_fraction = max(0.0, float(decode.section_diversity_reserve_fraction))
    section_diversity_reserve_min_note_score = max(0.0, float(decode.section_diversity_reserve_min_note_score))
    section_diversity_unique_weight = max(0.0, float(decode.section_diversity_unique_weight))
    section_diversity_pc_weight = max(0.0, float(decode.section_diversity_pc_weight))
    section_diversity_range_weight = max(0.0, float(decode.section_diversity_range_weight))
    section_diversity_onset_weight = max(0.0, float(decode.section_diversity_onset_weight))
    section_diversity_section_frames = max(1, int(round(float(decode.section_diversity_section_seconds) * float(frame_hz))))
    source_energy_velocity_weight = max(0.0, float(decode.source_energy_velocity_weight))
    density_plan_velocity_weight = max(0.0, float(decode.density_plan_velocity_weight))
    pc_onset_reserve_threshold = max(0.0, float(decode.pc_onset_plan_reserve_threshold))
    pc_onset_reserve_max_per_frame = max(0, int(decode.pc_onset_plan_reserve_max_per_frame))
    pc_onset_reserve_min_note_score = max(0.0, float(decode.pc_onset_plan_reserve_min_note_score))
    pc_onset_select_reserve_fraction = max(0.0, float(decode.pc_onset_plan_select_reserve_fraction))
    pc_onset_assign_threshold = max(0.0, float(decode.pc_onset_plan_assign_threshold))
    pc_onset_assign_fraction = max(0.0, float(decode.pc_onset_plan_assign_fraction))
    pc_onset_assign_window_frames = max(0, int(decode.pc_onset_plan_assign_window_frames))
    pc_onset_assign_min_note_score = max(0.0, float(decode.pc_onset_plan_assign_min_note_score))
    pc_onset_assign_source_weight = max(0.0, float(decode.pc_onset_plan_assign_source_weight))
    pc_onset_assign_event_weight = max(0.0, float(decode.pc_onset_plan_assign_event_weight))
    pc_onset_assign_distance_penalty = max(0.0, float(decode.pc_onset_plan_assign_distance_penalty))
    source_energy_center = float(np.mean(source_energy)) if source_energy is not None else 0.5
    density_plan_center = float(np.mean(density_plan)) if density_plan is not None else 0.5
    source_snapped_candidates = 0
    event_snapped_candidates = 0
    density_snapped_candidates = 0
    pc_onset_reserved_candidates = 0
    pc_onset_selected_reservations = 0
    pc_onset_assignment_events = 0
    pc_onset_assigned_notes = 0
    section_diversity_reserved_notes = 0
    section_diversity_center = 0.5
    if section_diversity_plan is not None:
        div_weight_sum = (
            section_diversity_unique_weight
            + section_diversity_pc_weight
            + section_diversity_range_weight
            + section_diversity_onset_weight
        )
        if div_weight_sum > 0.0:
            section_diversity_center = float(
                np.mean(
                    (
                        section_diversity_unique_weight * section_diversity_plan[0]
                        + section_diversity_pc_weight * section_diversity_plan[1]
                        + section_diversity_range_weight * section_diversity_plan[2]
                        + section_diversity_onset_weight * section_diversity_plan[3]
                    )
                    / div_weight_sum
                )
            )

    def snap_to_curve(start: int, curve: np.ndarray | None, frames: int, threshold: float) -> tuple[int, bool]:
        base = max(0, min(n_frames - 1, int(start)))
        if curve is None or int(frames) <= 0:
            return base, False
        lo = max(0, base - int(frames))
        hi = min(n_frames, base + int(frames) + 1)
        if hi <= lo:
            return base, False
        local = curve[lo:hi]
        best = int(lo + int(np.argmax(local)))
        if float(curve[best]) >= float(threshold) and best != base:
            return best, True
        return base, False

    def guided_start(start: int) -> int:
        nonlocal source_snapped_candidates, event_snapped_candidates, density_snapped_candidates
        base = max(0, min(n_frames - 1, int(start)))
        best, snapped = snap_to_curve(base, source_onset, snap_frames, snap_threshold)
        if snapped:
            source_snapped_candidates += 1
            return best
        best, snapped = snap_to_curve(base, event_source_curve, event_snap_frames, event_snap_threshold)
        if snapped:
            event_snapped_candidates += 1
            return best
        best, snapped = snap_to_curve(base, density_plan, density_snap_frames, density_snap_threshold)
        if snapped:
            density_snapped_candidates += 1
            return best
        return base

    def make_candidate(key: int, start: int, end: int, base_score: float, velocity_value: float) -> Dict[str, Any]:
        guided = guided_start(int(start))
        dur_frames = max(1, int(end) - int(start))
        out_end = min(n_frames, guided + dur_frames)
        source_score = float(source_onset[guided]) if source_onset is not None else 0.0
        source_energy_score = float(source_energy[guided]) if source_energy is not None else 0.0
        density_score = float(density_plan[guided]) if density_plan is not None else 0.0
        event_score = float(event_source_curve[guided]) if event_source_curve is not None else 0.0
        pitch_class = int((PIANO_MIN_MIDI + int(key)) % 12)
        source_chroma_score = float(source_chroma[pitch_class, guided] * 12.0) if source_chroma is not None else 1.0
        source_chroma_multiplier = max(0.05, 1.0 + source_chroma_guidance_weight * (source_chroma_score - 1.0))
        harmonic_score = float(harmonic_plan[pitch_class, guided] * 12.0) if harmonic_plan is not None else 1.0
        harmonic_multiplier = max(0.05, 1.0 + harmonic_guidance_weight * (harmonic_score - 1.0))
        pc_onset_score = float(pc_onset_plan[pitch_class, guided] * 12.0) if pc_onset_plan is not None else 1.0
        pc_onset_multiplier = max(0.05, 1.0 + pc_onset_guidance_weight * (pc_onset_score - 1.0))
        chord_score = float(chord_plan[pitch_class, guided] * 2.0) if chord_plan is not None else 1.0
        chord_multiplier = max(0.05, 1.0 + chord_guidance_weight * (chord_score - 1.0))
        key_register = _key_register(int(key))
        if bass_plan is not None and key_register == "low":
            bass_score = float(bass_plan[pitch_class, guided] * 13.0)
        else:
            bass_score = 1.0
        bass_multiplier = max(0.05, 1.0 + bass_guidance_weight * (bass_score - 1.0))
        voicing_score = 1.0
        if voicing_plan is not None:
            key_norm = float(int(key)) / max(1.0, float(n_keys - 1))
            span = max(0.08, float(voicing_plan[1, guided]) * 0.5 + 0.08)
            center = float(voicing_plan[2, guided])
            high_fraction = float(voicing_plan[3, guided])
            center_affinity = max(0.0, 1.0 - abs(key_norm - center) / span)
            high_affinity = high_fraction if key_register == "high" else (1.0 - high_fraction)
            voicing_score = 2.0 * (0.65 * center_affinity + 0.35 * high_affinity)
        voicing_multiplier = max(0.05, 1.0 + voicing_guidance_weight * (voicing_score - 1.0))
        section_diversity_score = 1.0
        if section_diversity_plan is not None:
            section_unique = float(section_diversity_plan[0, guided])
            section_pc = float(section_diversity_plan[1, guided])
            section_range = float(section_diversity_plan[2, guided])
            section_onset = float(section_diversity_plan[3, guided])
            range_affinity = 1.0 if key_register in {"low", "high"} else 0.55
            weight_sum = (
                section_diversity_unique_weight
                + section_diversity_pc_weight
                + section_diversity_range_weight
                + section_diversity_onset_weight
            )
            if weight_sum > 0.0:
                section_diversity_score = float(
                    (
                        section_diversity_unique_weight * section_unique
                        + section_diversity_pc_weight * section_pc
                        + section_diversity_range_weight * section_range * range_affinity
                        + section_diversity_onset_weight * section_onset
                    )
                    / weight_sum
                )
        section_diversity_multiplier = max(
            0.05,
            1.0 + section_diversity_guidance_weight * (section_diversity_score - section_diversity_center),
        )
        score = (
            float(base_score)
            * (1.0 + guidance_weight * source_score + density_guidance_weight * density_score + event_guidance_weight * event_score)
            * source_chroma_multiplier
            * harmonic_multiplier
            * pc_onset_multiplier
            * chord_multiplier
            * bass_multiplier
            * voicing_multiplier
            * section_diversity_multiplier
        )
        velocity_delta = (
            36.0 * source_energy_velocity_weight * (source_energy_score - source_energy_center)
            + 24.0 * density_plan_velocity_weight * (density_score - density_plan_center)
        )
        note_velocity = int(round(35 + 90 * float(velocity_value) + velocity_delta))
        return {
            "key": int(key),
            "start": int(guided),
            "end": int(out_end),
            "score": score,
            "model_score": float(base_score),
            "source_onset_score": source_score,
            "source_energy_score": source_energy_score,
            "density_plan_score": density_score,
            "event_plan_score": event_score,
            "source_chroma_score": source_chroma_score,
            "source_chroma_multiplier": source_chroma_multiplier,
            "harmonic_plan_score": harmonic_score,
            "harmonic_plan_multiplier": harmonic_multiplier,
            "pc_onset_plan_score": pc_onset_score,
            "pc_onset_plan_multiplier": pc_onset_multiplier,
            "chord_plan_score": chord_score,
            "chord_plan_multiplier": chord_multiplier,
            "bass_plan_score": bass_score,
            "bass_plan_multiplier": bass_multiplier,
            "voicing_plan_score": voicing_score,
            "voicing_plan_multiplier": voicing_multiplier,
            "section_diversity_score": section_diversity_score,
            "section_diversity_multiplier": section_diversity_multiplier,
            "velocity": max(1, min(127, note_velocity)),
        }

    candidates: List[Dict[str, Any]] = []
    for key in range(n_keys):
        starts = np.where(onset[key] >= float(decode.onset_threshold))[0].tolist()
        for start in starts:
            end = int(start) + 1
            while end < n_frames and frame[key, end] >= float(decode.frame_threshold):
                end += 1
            candidates.append(make_candidate(key, int(start), int(end), float(onset[key, int(start)]), float(np.max(velocity[key, int(start) : end]))))
    # If the model is still too cold, keep the sample inspectable by taking the top events.
    if not candidates:
        flat = onset.reshape(-1)
        for flat_idx in np.argsort(flat)[-max(1, int(decode.fallback_top_events)) :][::-1]:
            key = int(flat_idx // n_frames)
            start = int(flat_idx % n_frames)
            if float(flat[flat_idx]) <= 0.01:
                continue
            end = min(n_frames, start + max(1, int(round(float(decode.min_note_duration) * float(frame_hz)))))
            candidates.append(make_candidate(key, start, end, float(flat[flat_idx]), (64.0 - 35.0) / 90.0))

    existing_candidate_keys = {int(c["key"]) for c in candidates}
    needs_more_keys = len(existing_candidate_keys) < max(0, int(decode.min_unique_pitches))
    existing_registers = {_key_register(k) for k in existing_candidate_keys}
    needs_more_registers = bool(decode.require_register_coverage) and len(existing_registers) < 3
    if needs_more_keys or needs_more_registers:
        for key in range(n_keys):
            if key in existing_candidate_keys and not needs_more_registers:
                continue
            best_start = int(np.argmax(onset[key]))
            best_score = float(onset[key, best_start])
            if best_score < float(decode.diversity_fallback_threshold):
                continue
            end = min(n_frames, best_start + max(1, int(round(float(decode.min_note_duration) * float(frame_hz)))))
            candidates.append(make_candidate(int(key), int(best_start), int(end), best_score, float(velocity[key, best_start])))

    if pc_onset_plan is not None and pc_onset_reserve_threshold > 0.0 and pc_onset_reserve_max_per_frame > 0:
        candidate_sigs = {(int(c["key"]), int(c["start"])) for c in candidates}
        key_pitch_classes = np.asarray([(PIANO_MIN_MIDI + int(key)) % 12 for key in range(n_keys)], dtype=np.int32)
        reserve_frames = np.zeros((n_frames,), dtype=np.int32)
        for frame_idx in range(n_frames):
            pc_values = np.asarray(pc_onset_plan[:, frame_idx], dtype=np.float32)
            if not np.any(pc_values >= pc_onset_reserve_threshold):
                continue
            for pc in np.argsort(pc_values)[::-1]:
                pc = int(pc)
                plan_value = float(pc_values[pc])
                if plan_value < pc_onset_reserve_threshold:
                    break
                left = float(pc_onset_plan[pc, frame_idx - 1]) if frame_idx > 0 else -1.0
                right = float(pc_onset_plan[pc, frame_idx + 1]) if frame_idx + 1 < n_frames else -1.0
                if plan_value < max(left, right):
                    continue
                pc_keys = np.where(key_pitch_classes == pc)[0]
                if pc_keys.size == 0:
                    continue
                key_scores = onset[pc_keys, frame_idx] + 0.35 * frame[pc_keys, frame_idx]
                best_local = int(np.argmax(key_scores))
                key = int(pc_keys[best_local])
                note_score = float(key_scores[best_local])
                if note_score < pc_onset_reserve_min_note_score:
                    continue
                if (key, int(frame_idx)) in candidate_sigs:
                    continue
                end = min(n_frames, int(frame_idx) + max(1, int(round(float(decode.min_note_duration) * float(frame_hz)))))
                cand = make_candidate(key, int(frame_idx), int(end), max(note_score, plan_value), float(velocity[key, frame_idx]))
                cand["pc_onset_reserved"] = True
                candidates.append(cand)
                candidate_sigs.add((int(cand["key"]), int(cand["start"])))
                pc_onset_reserved_candidates += 1
                reserve_frames[frame_idx] += 1
                if reserve_frames[frame_idx] >= pc_onset_reserve_max_per_frame:
                    break

    per_frame_selected = np.zeros((n_frames,), dtype=np.int32)
    capped_candidates: List[Dict[str, Any]] = []
    for cand in sorted(candidates, key=lambda c: (int(c["start"]), -float(c["score"]), int(c["key"]))):
        start = int(cand["start"])
        if start < 0 or start >= n_frames:
            continue
        if per_frame_selected[start] >= max(1, int(decode.max_onsets_per_frame)):
            continue
        per_frame_selected[start] += 1
        capped_candidates.append(cand)

    max_notes = max(1, int(round(max(float(duration), float(n_frames) / float(frame_hz)) * float(decode.max_notes_per_second))))
    polyphony = np.zeros((n_frames,), dtype=np.int32)
    min_dur_frames = max(1, int(round(float(decode.min_note_duration) * float(frame_hz))))
    max_per_pitch = max(1, int(np.floor(max_notes * float(decode.max_pitch_fraction))))
    max_per_pitch_class = max(1, int(np.floor(max_notes * float(decode.max_pitch_class_fraction))))
    pitch_counts = np.zeros((n_keys,), dtype=np.int32)
    pitch_class_counts = np.zeros((12,), dtype=np.int32)
    start_counts = np.zeros((n_frames,), dtype=np.int32)
    selected: List[Dict[str, Any]] = []
    chunk_register_reservations = 0
    register_rebalance_replacements = 0
    section_bass_repairs = 0

    def try_add_candidate(
        cand: Dict[str, Any],
        *,
        enforce_pitch_caps: bool = True,
        force_short: bool = False,
        protected: bool = False,
    ) -> bool:
        if len(selected) >= max_notes:
            return False
        key = max(0, min(n_keys - 1, int(cand["key"])))
        pitch_class = int((PIANO_MIN_MIDI + key) % 12)
        if enforce_pitch_caps:
            if pitch_counts[key] >= max_per_pitch:
                return False
            if pitch_class_counts[pitch_class] >= max_per_pitch_class:
                return False
        start = max(0, min(n_frames - 1, int(cand["start"])))
        if bool(force_short):
            end = min(n_frames, start + min_dur_frames)
        else:
            end = max(start + min_dur_frames, min(n_frames, int(cand["end"])))
        if key <= 31 and float(decode.bass_min_note_duration) > 0:
            bass_min_dur_frames = max(min_dur_frames, int(round(float(decode.bass_min_note_duration) * float(frame_hz))))
            end = max(end, min(n_frames, start + bass_min_dur_frames))
        if float(decode.max_note_duration) > 0:
            max_dur_frames = max(min_dur_frames, int(round(float(decode.max_note_duration) * float(frame_hz))))
            end = min(end, start + max_dur_frames, n_frames)
        if start_counts[start] >= max(1, int(decode.max_onsets_per_frame)):
            return False
        if int(np.max(polyphony[start:end])) >= max(1, int(decode.max_simultaneous_notes)):
            return False
        polyphony[start:end] += 1
        start_counts[start] += 1
        pitch_counts[key] += 1
        pitch_class_counts[pitch_class] += 1
        selected.append({**cand, "key": key, "start": start, "end": end, "protected": bool(protected)})
        return True

    diversity_candidates = sorted(candidates, key=lambda c: (-float(c["score"]), int(c["start"]), int(c["key"])))
    sorted_candidates = sorted(capped_candidates, key=lambda c: (-float(c["score"]), int(c["start"]), int(c["key"])))
    min_unique = max(0, min(n_keys, int(decode.min_unique_pitches)))
    selected_keys: set[int] = set()
    selected_registers: set[str] = set()
    if section_diversity_plan is not None and section_diversity_guidance_weight > 0.0 and section_diversity_reserve_fraction > 0.0:
        reserve_limit = min(
            max_notes,
            max(0, int(round(max(1, int(decode.min_selected_notes)) * section_diversity_reserve_fraction))),
        )
        section_keys: Dict[int, set[int]] = {}
        section_pcs: Dict[int, set[int]] = {}
        section_registers: Dict[int, set[str]] = {}

        def section_idx(frame_idx: int) -> int:
            return int(max(0, min(n_frames - 1, int(frame_idx))) // section_diversity_section_frames)

        for cand in sorted(
            diversity_candidates,
            key=lambda c: (
                -float(c.get("section_diversity_score", 0.0)),
                -float(c.get("score", 0.0)),
                int(c.get("start", 0)),
                int(c.get("key", 0)),
            ),
        ):
            if section_diversity_reserved_notes >= reserve_limit:
                break
            if float(cand.get("model_score", cand.get("score", 0.0))) < section_diversity_reserve_min_note_score:
                continue
            key = max(0, min(n_keys - 1, int(cand["key"])))
            start = max(0, min(n_frames - 1, int(cand["start"])))
            sec = section_idx(start)
            sec_keys = section_keys.setdefault(sec, set())
            sec_pcs = section_pcs.setdefault(sec, set())
            sec_registers = section_registers.setdefault(sec, set())
            pc = int((PIANO_MIN_MIDI + key) % 12)
            register = _key_register(key)
            target_unique = int(np.ceil(float(section_diversity_plan[0, start]) * 16.0))
            target_pc = int(np.ceil(float(section_diversity_plan[1, start]) * 12.0))
            target_range = float(section_diversity_plan[2, start])
            needs_key = len(sec_keys) < max(1, target_unique) and key not in sec_keys
            needs_pc = len(sec_pcs) < max(1, target_pc) and pc not in sec_pcs
            needs_range = target_range >= 0.35 and register in {"low", "high"} and register not in sec_registers
            if not (needs_key or needs_pc or needs_range):
                continue
            reserve = {**cand, "section_diversity_reserved": True}
            if try_add_candidate(reserve, force_short=True, protected=True):
                sec_keys.add(key)
                sec_pcs.add(pc)
                sec_registers.add(register)
                selected_keys.add(key)
                selected_registers.add(register)
                section_diversity_reserved_notes += 1
    if pc_onset_plan is not None and pc_onset_assign_threshold > 0.0 and pc_onset_assign_fraction > 0.0:
        key_pitch_classes = np.asarray([(PIANO_MIN_MIDI + int(key)) % 12 for key in range(n_keys)], dtype=np.int32)
        assignment_limit = min(
            max_notes,
            max(0, int(round(max(1, int(decode.min_selected_notes)) * pc_onset_assign_fraction))),
        )
        plan_events: List[tuple[float, int, int]] = []
        for frame_idx in range(n_frames):
            pc_values = np.asarray(pc_onset_plan[:, frame_idx], dtype=np.float32)
            if not np.any(pc_values >= pc_onset_assign_threshold):
                continue
            for pc in np.argsort(pc_values)[::-1]:
                pc = int(pc)
                plan_value = float(pc_values[pc])
                if plan_value < pc_onset_assign_threshold:
                    break
                left = float(pc_onset_plan[pc, frame_idx - 1]) if frame_idx > 0 else -1.0
                right = float(pc_onset_plan[pc, frame_idx + 1]) if frame_idx + 1 < n_frames else -1.0
                if plan_value < max(left, right):
                    continue
                plan_events.append((plan_value, int(frame_idx), pc))
        pc_onset_assignment_events = int(len(plan_events))
        assigned_sigs = {(int(c["key"]), int(c["start"])) for c in selected}
        def assignment_event_score(plan_value: float, frame_idx: int) -> float:
            source_value = float(source_onset[frame_idx]) if source_onset is not None else 0.0
            event_value = float(event_source_curve[frame_idx]) if event_source_curve is not None else 0.0
            return float(plan_value) * (
                1.0
                + pc_onset_assign_source_weight * source_value
                + pc_onset_assign_event_weight * event_value
            )

        for plan_value, frame_idx, pc in sorted(
            plan_events,
            key=lambda item: (
                -assignment_event_score(float(item[0]), int(item[1])),
                -float(item[0]),
                int(item[1]),
                int(item[2]),
            ),
        ):
            if pc_onset_assigned_notes >= assignment_limit:
                break
            lo = max(0, int(frame_idx) - pc_onset_assign_window_frames)
            hi = min(n_frames - 1, int(frame_idx) + pc_onset_assign_window_frames)
            local_candidates = []
            for cand in diversity_candidates:
                key = max(0, min(n_keys - 1, int(cand["key"])))
                if int((PIANO_MIN_MIDI + key) % 12) != pc:
                    continue
                start = max(0, min(n_frames - 1, int(cand["start"])))
                if start < lo or start > hi:
                    continue
                if (key, start) in assigned_sigs:
                    continue
                distance = abs(start - int(frame_idx))
                if pc_onset_assign_source_weight > 0.0 or pc_onset_assign_event_weight > 0.0:
                    assign_score = (
                        np.log1p(max(0.0, float(cand.get("score", 0.0))))
                        + pc_onset_assign_source_weight * float(cand.get("source_onset_score", 0.0))
                        + pc_onset_assign_event_weight * float(cand.get("event_plan_score", 0.0))
                        + 0.05 * float(cand.get("pc_onset_plan_score", 0.0))
                        - pc_onset_assign_distance_penalty * float(distance)
                    )
                    local_candidates.append((-float(assign_score), distance, key, start, cand))
                else:
                    local_candidates.append(
                        (
                            distance,
                            -float(cand.get("score", 0.0)),
                            -float(cand.get("pc_onset_plan_score", 0.0)),
                            key,
                            start,
                            cand,
                        )
                    )
            if local_candidates:
                cand = dict(sorted(local_candidates, key=lambda item: item[:-1])[0][-1])
            else:
                pc_keys = np.where(key_pitch_classes == pc)[0]
                if pc_keys.size == 0:
                    continue
                best_key = None
                best_score = -1.0
                best_start = int(frame_idx)
                for test_frame in range(lo, hi + 1):
                    key_scores = onset[pc_keys, test_frame] + 0.35 * frame[pc_keys, test_frame]
                    if pc_onset_assign_source_weight > 0.0 and source_onset is not None:
                        key_scores = key_scores * (1.0 + pc_onset_assign_source_weight * float(source_onset[test_frame]))
                    if pc_onset_assign_event_weight > 0.0 and event_source_curve is not None:
                        key_scores = key_scores * (1.0 + pc_onset_assign_event_weight * float(event_source_curve[test_frame]))
                    local_idx = int(np.argmax(key_scores))
                    score = float(key_scores[local_idx])
                    if score > best_score:
                        best_score = score
                        best_key = int(pc_keys[local_idx])
                        best_start = int(test_frame)
                if best_key is None or best_score < pc_onset_assign_min_note_score:
                    continue
                end = min(n_frames, best_start + min_dur_frames)
                cand = make_candidate(best_key, best_start, end, max(best_score, float(plan_value)), float(velocity[best_key, best_start]))
            key = max(0, min(n_keys - 1, int(cand["key"])))
            start = max(0, min(n_frames - 1, int(cand["start"])))
            if (key, start) in assigned_sigs:
                continue
            cand["pc_onset_assigned"] = True
            cand["pc_onset_assignment_plan_value"] = float(plan_value)
            cand["pc_onset_assignment_pc"] = int(pc)
            cand["pc_onset_assignment_frame"] = int(frame_idx)
            if try_add_candidate(cand, force_short=True, protected=True):
                assigned_sigs.add((key, start))
                selected_keys.add(key)
                selected_registers.add(_key_register(key))
                pc_onset_assigned_notes += 1
    if pc_onset_select_reserve_fraction > 0.0:
        reserve_limit = min(
            max_notes,
            max(0, int(round(max(1, int(decode.min_selected_notes)) * pc_onset_select_reserve_fraction))),
        )
        reserve_candidates = [
            cand for cand in diversity_candidates if bool(cand.get("pc_onset_reserved", False))
        ]
        reserve_candidates = sorted(
            reserve_candidates,
            key=lambda c: (
                -float(c.get("pc_onset_plan_score", 0.0)),
                -float(c.get("score", 0.0)),
                int(c.get("start", 0)),
                int(c.get("key", 0)),
            ),
        )
        for cand in reserve_candidates:
            if pc_onset_selected_reservations >= reserve_limit:
                break
            if try_add_candidate(cand, force_short=True, protected=True):
                key = max(0, min(n_keys - 1, int(cand["key"])))
                selected_keys.add(key)
                selected_registers.add(_key_register(key))
                pc_onset_selected_reservations += 1
    if bool(decode.require_register_coverage):
        for wanted_register in ("low", "mid", "high"):
            for cand in diversity_candidates:
                key = max(0, min(n_keys - 1, int(cand["key"])))
                if _key_register(key) != wanted_register:
                    continue
                if try_add_candidate(cand, force_short=True, protected=True):
                    selected_keys.add(key)
                    selected_registers.add(wanted_register)
                    break

    chunk_seconds = float(decode.register_coverage_chunk_seconds)
    if bool(decode.require_register_coverage) and chunk_seconds > 0.0:
        chunk_frames = max(1, int(round(chunk_seconds * float(frame_hz))))
        for chunk_start in range(0, n_frames, chunk_frames):
            chunk_end = min(n_frames, chunk_start + chunk_frames)
            chunk_registers: set[str] = set()
            for cand in selected:
                start = int(cand["start"])
                if chunk_start <= start < chunk_end:
                    chunk_registers.add(_key_register(int(cand["key"])))
            for wanted_register in ("low", "mid", "high"):
                if wanted_register in chunk_registers:
                    continue
                for cand in diversity_candidates:
                    start = int(cand["start"])
                    if not (chunk_start <= start < chunk_end):
                        continue
                    key = max(0, min(n_keys - 1, int(cand["key"])))
                    if _key_register(key) != wanted_register:
                        continue
                    if try_add_candidate(cand, force_short=True, protected=True):
                        selected_keys.add(key)
                        selected_registers.add(wanted_register)
                        chunk_registers.add(wanted_register)
                        chunk_register_reservations += 1
                        break

    for cand in diversity_candidates:
        key = max(0, min(n_keys - 1, int(cand["key"])))
        register = _key_register(key)
        needs_key = len(selected_keys) < min_unique and key not in selected_keys
        needs_register = bool(decode.require_register_coverage) and register not in selected_registers
        if not needs_key and not needs_register:
            continue
        if try_add_candidate(cand, force_short=True, protected=True):
            selected_keys.add(key)
            selected_registers.add(register)
        if len(selected_keys) >= min_unique and (not bool(decode.require_register_coverage) or len(selected_registers) >= 3):
            break

    for cand in sorted_candidates:
        if len(selected) >= max_notes:
            break
        try_add_candidate(cand)
    selected = _prune_pitch_dominance(
        selected,
        max_pitch_fraction=float(decode.max_pitch_fraction),
        max_pitch_class_fraction=float(decode.max_pitch_class_fraction),
        min_keep=min(max_notes, max(0, int(decode.min_selected_notes))),
    )
    if selected:
        selected_sigs = {(int(c["key"]), int(c["start"])) for c in selected}
        for _ in range(64):
            n_selected = int(len(selected))
            if n_selected <= 0:
                break
            pc_cap = max(1, int(np.floor(n_selected * float(decode.max_pitch_class_fraction))))
            pc_counts = np.zeros((12,), dtype=np.int32)
            for cand in selected:
                pc_counts[int((PIANO_MIN_MIDI + int(cand["key"])) % 12)] += 1
            over_pcs = np.where(pc_counts > pc_cap)[0]
            if over_pcs.size == 0:
                break
            over_pc = int(over_pcs[np.argmax(pc_counts[over_pcs])])
            remove_idx = None
            remove_score = float("inf")
            for i, cand in enumerate(selected):
                pc = int((PIANO_MIN_MIDI + int(cand["key"])) % 12)
                if pc != over_pc:
                    continue
                score = float(cand.get("score", 0.0))
                if score < remove_score:
                    remove_idx = i
                    remove_score = score
            if remove_idx is None:
                break
            replacement = None
            selected_without = [c for i, c in enumerate(selected) if i != remove_idx]
            poly = np.zeros((n_frames,), dtype=np.int32)
            starts = np.zeros((n_frames,), dtype=np.int32)
            for cand in selected_without:
                start = max(0, min(n_frames - 1, int(cand["start"])))
                end = max(start + 1, min(n_frames, int(cand["end"])))
                poly[start:end] += 1
                starts[start] += 1
            pc_counts_after = np.zeros((12,), dtype=np.int32)
            for cand in selected_without:
                pc_counts_after[int((PIANO_MIN_MIDI + int(cand["key"])) % 12)] += 1
            for cand in diversity_candidates:
                key = max(0, min(n_keys - 1, int(cand["key"])))
                start = max(0, min(n_frames - 1, int(cand["start"])))
                sig = (key, start)
                if sig in selected_sigs:
                    continue
                pc = int((PIANO_MIN_MIDI + key) % 12)
                if pc == over_pc or pc_counts_after[pc] >= pc_cap:
                    continue
                end = max(start + min_dur_frames, min(n_frames, int(cand["end"])))
                if starts[start] >= max(1, int(decode.max_onsets_per_frame)):
                    continue
                if int(np.max(poly[start:end])) >= max(1, int(decode.max_simultaneous_notes)):
                    continue
                replacement = {**cand, "key": key, "start": start, "end": end, "protected": False}
                break
            if replacement is None:
                break
            removed = selected.pop(remove_idx)
            selected_sigs.discard((int(removed["key"]), int(removed["start"])))
            selected.append(replacement)
            selected_sigs.add((int(replacement["key"]), int(replacement["start"])))
    if selected and bool(decode.require_register_coverage):
        selected_sigs = {(int(c["key"]), int(c["start"])) for c in selected}
        for _ in range(16):
            n_selected = int(len(selected))
            min_register_notes = max(1, int(np.ceil(0.05 * float(n_selected))))
            register_counts = {reg: 0 for reg in ("low", "mid", "high")}
            for cand in selected:
                register_counts[_key_register(int(cand["key"]))] += 1
            under = [reg for reg, count in register_counts.items() if count < min_register_notes]
            if not under:
                break
            wanted_register = min(under, key=lambda reg: register_counts[reg])
            donor_registers = [
                reg for reg, count in sorted(register_counts.items(), key=lambda item: -int(item[1]))
                if reg != wanted_register and count > min_register_notes
            ]
            if not donor_registers:
                break
            donor_register = donor_registers[0]
            remove_idx = None
            remove_score = float("inf")
            for i, cand in enumerate(selected):
                if _key_register(int(cand["key"])) != donor_register:
                    continue
                score = float(cand.get("score", 0.0))
                if score < remove_score:
                    remove_idx = i
                    remove_score = score
            if remove_idx is None:
                break
            selected_without = [c for i, c in enumerate(selected) if i != remove_idx]
            poly = np.zeros((n_frames,), dtype=np.int32)
            starts = np.zeros((n_frames,), dtype=np.int32)
            for cand in selected_without:
                start = max(0, min(n_frames - 1, int(cand["start"])))
                end = max(start + 1, min(n_frames, int(cand["end"])))
                poly[start:end] += 1
                starts[start] += 1
            pc_cap = max(1, int(np.floor(max(1, len(selected_without) + 1) * float(decode.max_pitch_class_fraction))))
            pc_counts_after = np.zeros((12,), dtype=np.int32)
            for cand in selected_without:
                pc_counts_after[int((PIANO_MIN_MIDI + int(cand["key"])) % 12)] += 1
            replacement = None
            for cand in diversity_candidates:
                key = max(0, min(n_keys - 1, int(cand["key"])))
                if _key_register(key) != wanted_register:
                    continue
                start = max(0, min(n_frames - 1, int(cand["start"])))
                sig = (key, start)
                if sig in selected_sigs:
                    continue
                pc = int((PIANO_MIN_MIDI + key) % 12)
                if pc_counts_after[pc] >= pc_cap:
                    continue
                end = max(start + min_dur_frames, min(n_frames, int(cand["end"])))
                if starts[start] >= max(1, int(decode.max_onsets_per_frame)):
                    continue
                if int(np.max(poly[start:end])) >= max(1, int(decode.max_simultaneous_notes)):
                    continue
                replacement = {**cand, "key": key, "start": start, "end": end, "protected": True}
                break
            if replacement is None:
                break
            removed = selected.pop(remove_idx)
            selected_sigs.discard((int(removed["key"]), int(removed["start"])))
            selected.append(replacement)
            selected_sigs.add((int(replacement["key"]), int(replacement["start"])))
            register_rebalance_replacements += 1

    if (
        selected
        and bool(decode.require_register_coverage)
        and bool(decode.section_bass_repair)
        and float(decode.register_coverage_chunk_seconds) > 0.0
    ):
        selected_sigs = {(int(c["key"]), int(c["start"])) for c in selected}
        repair_chunk_frames = max(1, int(round(float(decode.register_coverage_chunk_seconds) * float(frame_hz))))
        bass_min_dur_frames = max(
            min_dur_frames,
            int(round(max(float(decode.bass_min_note_duration), 0.35) * float(frame_hz))),
        )

        def selected_bass_coverage(chunk_start: int, chunk_end: int) -> float:
            active = np.zeros((chunk_end - chunk_start,), dtype=np.bool_)
            bass = np.zeros_like(active)
            for cand in selected:
                start = max(chunk_start, int(cand["start"]))
                end = min(chunk_end, int(cand["end"]))
                if end <= start:
                    continue
                active[start - chunk_start : end - chunk_start] = True
                if int(cand["key"]) <= 31:
                    bass[start - chunk_start : end - chunk_start] = True
            if not bool(np.any(active)):
                return 1.0
            return float(np.mean(bass[active]))

        low_candidates = [
            cand for cand in diversity_candidates if max(0, min(n_keys - 1, int(cand["key"]))) <= 31
        ]
        for chunk_start in range(0, n_frames, repair_chunk_frames):
            chunk_end = min(n_frames, chunk_start + repair_chunk_frames)
            if selected_bass_coverage(chunk_start, chunk_end) >= float(decode.section_bass_repair_min_coverage):
                continue
            in_chunk = [
                cand
                for cand in low_candidates
                if chunk_start <= int(cand["start"]) < chunk_end
            ]
            repair_pool = in_chunk if in_chunk else low_candidates
            for cand in repair_pool:
                key = max(0, min(n_keys - 1, int(cand["key"])))
                start = int(cand["start"])
                if start < chunk_start or start >= chunk_end:
                    start = chunk_start
                start = max(chunk_start, min(start, max(chunk_start, chunk_end - bass_min_dur_frames)))
                end = min(n_frames, max(start + bass_min_dur_frames, int(cand["end"])))
                sig = (key, start)
                if sig in selected_sigs:
                    continue
                repair = {**cand, "key": key, "start": start, "end": end, "protected": True}
                if try_add_candidate(repair, enforce_pitch_caps=True, protected=True):
                    selected_sigs.add(sig)
                    section_bass_repairs += 1
                    break

    notes: List[PianoNote] = []
    for cand in selected:
        start = int(cand["start"])
        end = int(cand["end"])
        dur_frames = max(1, end - start)
        notes.append(
            PianoNote(
                start=float(start) / float(frame_hz),
                duration=float(dur_frames) / float(frame_hz),
                pitch=int(PIANO_MIN_MIDI + int(cand["key"])),
                velocity=max(1, min(127, int(cand["velocity"]))),
            )
        )

    sustain: List[SustainEvent] = []
    if pedal.size:
        if float(np.mean(pedal)) > 0.05:
            pedal_value = int(max(0, min(127, round(float(np.mean(pedal)) * 127))))
            sustain.append(SustainEvent(time=0.0, value=pedal_value))
            sustain.append(SustainEvent(time=max(0.0, float(duration) - 0.05), value=0))
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))[:max_notes]
    metadata = {
        **metadata,
        "decode_config": asdict(decode),
        "decode_candidates": int(len(candidates)),
        "decode_selected": int(len(notes)),
        "decode_chunk_register_reservations": int(chunk_register_reservations),
        "decode_pc_onset_reserved_candidates": int(pc_onset_reserved_candidates),
        "decode_pc_onset_selected_reservations": int(pc_onset_selected_reservations),
        "decode_pc_onset_assignment_events": int(pc_onset_assignment_events),
        "decode_pc_onset_assigned_notes": int(pc_onset_assigned_notes),
        "decode_register_rebalance_replacements": int(register_rebalance_replacements),
        "decode_section_bass_repairs": int(section_bass_repairs),
        "decode_source_onset_guidance": {
            "available": bool(source_onset is not None),
            "weight": float(guidance_weight),
            "snap_frames": int(snap_frames),
            "snap_threshold": float(snap_threshold),
            "snapped_candidates": int(source_snapped_candidates),
        },
        "decode_density_plan_guidance": {
            "available": bool(density_plan is not None),
            "weight": float(density_guidance_weight),
            "snap_frames": int(density_snap_frames),
            "snap_threshold": float(density_snap_threshold),
            "snapped_candidates": int(density_snapped_candidates),
        },
        "decode_event_plan_guidance": {
            "available": bool(event_onset_plan is not None),
            "source_blended": bool(event_onset_plan is not None and source_onset is not None),
            "weight": float(event_guidance_weight),
            "snap_frames": int(event_snap_frames),
            "snap_threshold": float(event_snap_threshold),
            "snapped_candidates": int(event_snapped_candidates),
        },
        "decode_harmonic_plan_guidance": {
            "available": bool(harmonic_plan is not None),
            "weight": float(harmonic_guidance_weight),
        },
        "decode_musical_plan_guidance": {
            "chord_available": bool(chord_plan is not None),
            "bass_available": bool(bass_plan is not None),
            "voicing_available": bool(voicing_plan is not None),
            "event_available": bool(event_onset_plan is not None),
            "pc_onset_available": bool(pc_onset_plan is not None),
            "chord_weight": float(chord_guidance_weight),
            "bass_weight": float(bass_guidance_weight),
            "voicing_weight": float(voicing_guidance_weight),
            "section_diversity_available": bool(section_diversity_plan is not None),
            "section_diversity_weight": float(section_diversity_guidance_weight),
            "section_diversity_reserve_fraction": float(section_diversity_reserve_fraction),
            "section_diversity_reserved_notes": int(section_diversity_reserved_notes),
            "pc_onset_weight": float(pc_onset_guidance_weight),
            "pc_onset_reserve_threshold": float(pc_onset_reserve_threshold),
            "pc_onset_reserve_max_per_frame": int(pc_onset_reserve_max_per_frame),
            "pc_onset_reserved_candidates": int(pc_onset_reserved_candidates),
            "pc_onset_select_reserve_fraction": float(pc_onset_select_reserve_fraction),
            "pc_onset_selected_reservations": int(pc_onset_selected_reservations),
            "pc_onset_assign_threshold": float(pc_onset_assign_threshold),
            "pc_onset_assign_fraction": float(pc_onset_assign_fraction),
            "pc_onset_assign_window_frames": int(pc_onset_assign_window_frames),
            "pc_onset_assign_source_weight": float(pc_onset_assign_source_weight),
            "pc_onset_assign_event_weight": float(pc_onset_assign_event_weight),
            "pc_onset_assign_distance_penalty": float(pc_onset_assign_distance_penalty),
            "pc_onset_assignment_events": int(pc_onset_assignment_events),
            "pc_onset_assigned_notes": int(pc_onset_assigned_notes),
        },
        "decode_source_chroma_guidance": {
            "available": bool(source_chroma is not None),
            "weight": float(source_chroma_guidance_weight),
        },
        "decode_section_diversity_guidance": {
            "available": bool(section_diversity_plan is not None),
            "weight": float(section_diversity_guidance_weight),
            "reserve_fraction": float(section_diversity_reserve_fraction),
            "reserve_min_note_score": float(section_diversity_reserve_min_note_score),
            "reserved_notes": int(section_diversity_reserved_notes),
            "unique_weight": float(section_diversity_unique_weight),
            "pc_weight": float(section_diversity_pc_weight),
            "range_weight": float(section_diversity_range_weight),
            "onset_weight": float(section_diversity_onset_weight),
            "section_seconds": float(decode.section_diversity_section_seconds),
            "section_frames": int(section_diversity_section_frames),
        },
        "decode_dynamic_velocity": {
            "source_energy_available": bool(source_energy is not None),
            "source_energy_velocity_weight": float(source_energy_velocity_weight),
            "density_plan_velocity_weight": float(density_plan_velocity_weight),
        },
    }
    return PianoArrangement(notes=notes, tempo_bpm=120.0, duration=float(duration), sustain=sustain, metadata=metadata)


@torch.no_grad()
def _write_epoch_samples(
    model: PianoRollGenerator,
    ds: PianoRollCacheDataset,
    run_dir: Path,
    epoch: int,
    *,
    cfg: PianoTrainConfig,
    device: torch.device,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    model.eval()
    sample_dir = run_dir / "samples" / f"epoch_{int(epoch):03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    frame_hz = float(meta.get("frame_hz", 25.0))
    duration = float(meta.get("seconds", ds.source.shape[-1] / frame_hz))
    rows: List[Dict[str, Any]] = []
    eval_reports: List[Dict[str, Any]] = []
    count = min(int(cfg.sample_count), len(ds))
    for i in range(count):
        item = ds[i]
        source = item["source"].unsqueeze(0).to(device)
        pred = model(source)
        arrangement = prediction_to_arrangement(
            pred,
            frame_hz=frame_hz,
            duration=duration,
            onset_threshold=float(cfg.onset_threshold),
            frame_threshold=float(cfg.frame_threshold),
            decode_config=PianoDecodeConfig(
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
                min_selected_notes=int(cfg.min_selected_notes),
                min_unique_pitches=int(cfg.min_unique_pitches),
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
            ),
            metadata={"cache_idx": int(i), "epoch": int(epoch), "source": "model_prediction"},
            source_condition=item["source"],
        )
        out_stem = sample_dir / f"sample_{i:03d}"
        bundle = write_arrangement_bundle(arrangement, out_stem=out_stem, render_wav=True)
        row: Dict[str, Any] = {"cache_idx": int(i), "notes": int(len(arrangement.notes)), **bundle}
        if bool(cfg.sample_eval):
            source_audio: Path | None = None
            if bool(cfg.sample_source_eval):
                source_audio = _source_audio_for_sample(ds, i)
            eval_report = evaluate_arrangement_dict(
                arrangement_to_dict(arrangement),
                label=f"epoch_{int(epoch):03d}_sample_{i:03d}",
                source_audio=source_audio,
                source_seconds=duration,
                frame_hz=frame_hz,
                max_frames=int(ds.source.shape[-1]),
            )
            if bool(cfg.sample_section_eval):
                section_report = section_report_arrangement_dict(
                    arrangement_to_dict(arrangement),
                    label=f"epoch_{int(epoch):03d}_sample_{i:03d}",
                    section_seconds=float(cfg.sample_section_seconds),
                )
                section_warnings = list(section_report.get("warnings", []))
                eval_report["warnings"].extend(section_warnings)
                eval_report["passed"] = bool(eval_report.get("passed", False)) and not section_warnings
                eval_report["section_report"] = section_report
                section_summary = dict(section_report.get("summary", {}))
                eval_report["metrics"].update(
                    {
                        "section_warning_count": float(len(section_warnings)),
                        "min_section_notes": section_summary.get("min_section_notes"),
                        "min_section_unique_pitches": section_summary.get("min_section_unique_pitches"),
                        "min_section_bass_coverage_fraction": section_summary.get(
                            "min_section_bass_coverage_fraction"
                        ),
                        "min_section_chord_frame_fraction": section_summary.get(
                            "min_section_chord_frame_fraction"
                        ),
                        "min_section_fullness_score": section_summary.get("min_section_fullness_score"),
                    }
                )
            wav_info = bundle.get("wav", {}) if isinstance(bundle.get("wav", {}), dict) else {}
            wav_path = wav_info.get("out_wav")
            if wav_path:
                try:
                    import soundfile as sf

                    wav, _sr = sf.read(str(wav_path), always_2d=False)
                    eval_report["metrics"]["wav_rms"] = float(np.sqrt(np.mean(np.square(wav)))) if len(wav) else 0.0
                except Exception as exc:
                    eval_report["metrics"]["wav_rms_error"] = str(exc)
            target_midi: Path | None = None
            if bool(cfg.sample_target_eval):
                target_midi = _target_midi_for_sample(ds, i)
            if target_midi is not None:
                eval_report["metrics"].update(
                    target_midi_alignment_metrics(
                        arrangement_to_dict(arrangement),
                        target_midi=target_midi,
                        target_seconds=duration,
                        frame_hz=frame_hz,
                        max_frames=int(ds.source.shape[-1]),
                    )
                )
                target_warnings = target_midi_alignment_warnings(
                    eval_report["metrics"],
                    min_target_global_chroma_cosine=float(cfg.min_target_global_chroma_cosine),
                    min_target_active_chroma_cosine=float(cfg.min_target_active_chroma_cosine),
                    min_target_onset_correlation=float(cfg.min_target_onset_correlation),
                    min_target_onset_frame_f1=float(cfg.min_target_onset_frame_f1),
                    min_target_pitch_class_onset_f1=float(cfg.min_target_pitch_class_onset_f1),
                    min_target_note_count_ratio=float(cfg.min_target_note_count_ratio),
                    max_target_note_count_ratio=float(cfg.max_target_note_count_ratio),
                )
                eval_report["warnings"].extend(target_warnings)
                eval_report["passed"] = bool(eval_report.get("passed", False)) and not target_warnings
                eval_report["target_thresholds"] = {
                    "enabled": True,
                    "min_target_global_chroma_cosine": float(cfg.min_target_global_chroma_cosine),
                    "min_target_active_chroma_cosine": float(cfg.min_target_active_chroma_cosine),
                    "min_target_onset_correlation": float(cfg.min_target_onset_correlation),
                    "min_target_onset_frame_f1": float(cfg.min_target_onset_frame_f1),
                    "min_target_pitch_class_onset_f1": float(cfg.min_target_pitch_class_onset_f1),
                    "min_target_note_count_ratio": float(cfg.min_target_note_count_ratio),
                    "max_target_note_count_ratio": float(cfg.max_target_note_count_ratio),
                }
            eval_path = sample_dir / f"sample_{i:03d}.eval.json"
            _write_json(eval_path, eval_report)
            eval_reports.append(eval_report)
            row["eval"] = {
                "passed": bool(eval_report["passed"]),
                "warnings": list(eval_report["warnings"]),
                "report": str(eval_path),
                "source_global_chroma_cosine": eval_report["metrics"].get("source_global_chroma_cosine"),
                "source_active_chroma_cosine": eval_report["metrics"].get("source_active_chroma_cosine"),
                "source_onset_correlation": eval_report["metrics"].get("source_onset_correlation"),
                "target_global_chroma_cosine": eval_report["metrics"].get("target_global_chroma_cosine"),
                "target_active_chroma_cosine": eval_report["metrics"].get("target_active_chroma_cosine"),
                "target_onset_correlation": eval_report["metrics"].get("target_onset_correlation"),
                "target_onset_frame_f1": eval_report["metrics"].get("target_onset_frame_f1"),
                "target_pitch_class_onset_f1": eval_report["metrics"].get("target_pitch_class_onset_f1"),
                "target_note_count_ratio": eval_report["metrics"].get("target_note_count_ratio"),
                "single_pitch_fraction": eval_report["metrics"].get("single_pitch_fraction"),
                "single_pitch_class_fraction": eval_report["metrics"].get("single_pitch_class_fraction"),
                "mid_note_fraction": eval_report["metrics"].get("mid_note_fraction"),
                "high_note_fraction": eval_report["metrics"].get("high_note_fraction"),
                "chord_frame_fraction": eval_report["metrics"].get("chord_frame_fraction"),
                "bass_coverage_fraction": eval_report["metrics"].get("bass_coverage_fraction"),
                "melody_coverage_fraction": eval_report["metrics"].get("melody_coverage_fraction"),
                "mean_active_polyphony": eval_report["metrics"].get("mean_active_polyphony"),
                "fullness_score": eval_report["metrics"].get("fullness_score"),
                "wav_rms": eval_report["metrics"].get("wav_rms"),
                "section_warning_count": eval_report["metrics"].get("section_warning_count"),
                "min_section_notes": eval_report["metrics"].get("min_section_notes"),
                "min_section_unique_pitches": eval_report["metrics"].get("min_section_unique_pitches"),
                "min_section_bass_coverage_fraction": eval_report["metrics"].get(
                    "min_section_bass_coverage_fraction"
                ),
                "min_section_chord_frame_fraction": eval_report["metrics"].get(
                    "min_section_chord_frame_fraction"
                ),
                "min_section_fullness_score": eval_report["metrics"].get("min_section_fullness_score"),
            }
            if bool(cfg.sample_source_eval) and source_audio is None:
                row["eval"]["source_eval_skipped"] = "cache row path is not an audio file"
            if bool(cfg.sample_target_eval) and target_midi is None:
                row["eval"]["target_eval_skipped"] = "cache row target_midi is missing or not a MIDI file"
        rows.append(row)
    aggregate_eval: Dict[str, Any] = {}
    if eval_reports:
        metric_keys = [
            "notes_per_second",
            "max_simultaneous_notes",
            "source_global_chroma_cosine",
            "source_active_chroma_cosine",
            "source_onset_correlation",
            "target_global_chroma_cosine",
            "target_active_chroma_cosine",
            "target_onset_correlation",
            "target_onset_frame_f1",
            "target_pitch_class_onset_f1",
            "target_note_count_ratio",
            "single_pitch_fraction",
            "single_pitch_class_fraction",
            "mid_note_fraction",
            "high_note_fraction",
            "chord_frame_fraction",
            "bass_coverage_fraction",
            "melody_coverage_fraction",
            "mean_active_polyphony",
            "fullness_score",
            "wav_rms",
            "section_warning_count",
            "min_section_notes",
            "min_section_unique_pitches",
            "min_section_bass_coverage_fraction",
            "min_section_chord_frame_fraction",
            "min_section_fullness_score",
        ]
        aggregate_eval = {
            "sample_eval_count": int(len(eval_reports)),
            "sample_eval_pass_count": int(sum(1 for r in eval_reports if bool(r.get("passed")))),
            "warning_counts": {},
        }
        for report in eval_reports:
            for warning in report.get("warnings", []):
                code = str(warning).split(":", 1)[0]
                aggregate_eval["warning_counts"][code] = int(aggregate_eval["warning_counts"].get(code, 0)) + 1
        for key in metric_keys:
            vals = [float(r["metrics"][key]) for r in eval_reports if key in r.get("metrics", {})]
            if vals:
                aggregate_eval[f"mean_{key}"] = float(np.mean(vals))
    summary = {"epoch": int(epoch), "n_samples": int(len(rows)), "rows": rows, "aggregate_eval": aggregate_eval}
    _write_json(sample_dir / "summary.json", summary)
    model.train()
    return summary


def _sample_eval_score_components(aggregate_eval: Dict[str, Any], cfg: PianoTrainConfig) -> Dict[str, float]:
    warning_count = int(sum(int(v) for v in aggregate_eval.get("warning_counts", {}).values()))
    pass_score = float(cfg.sample_score_pass_weight) * float(aggregate_eval.get("sample_eval_pass_count", 0))
    warning_penalty = float(cfg.sample_score_warning_penalty) * float(warning_count)
    source_score = (
        float(cfg.sample_score_source_active_weight)
        * float(aggregate_eval.get("mean_source_active_chroma_cosine", 0.0))
        + float(cfg.sample_score_source_onset_weight)
        * float(aggregate_eval.get("mean_source_onset_correlation", 0.0))
    )
    target_score = (
        float(cfg.sample_score_target_active_weight)
        * float(aggregate_eval.get("mean_target_active_chroma_cosine", 0.0))
        + float(cfg.sample_score_target_onset_weight)
        * float(aggregate_eval.get("mean_target_onset_correlation", 0.0))
    )
    chord_score = min(
        1.0,
        float(aggregate_eval.get("mean_chord_frame_fraction", 0.0))
        / max(1e-6, float(cfg.sample_score_chord_frame_target)),
    )
    melody_score = min(
        1.0,
        float(aggregate_eval.get("mean_melody_coverage_fraction", 0.0))
        / max(1e-6, float(cfg.sample_score_melody_coverage_target)),
    )
    bass = float(aggregate_eval.get("mean_bass_coverage_fraction", 0.0))
    bass_score = 1.0
    if bass < float(cfg.sample_score_bass_coverage_min):
        bass_score = max(0.0, bass / max(1e-6, float(cfg.sample_score_bass_coverage_min)))
    elif bass > float(cfg.sample_score_bass_coverage_max):
        span = max(1e-6, 1.0 - float(cfg.sample_score_bass_coverage_max))
        bass_score = max(0.0, 1.0 - (bass - float(cfg.sample_score_bass_coverage_max)) / span)
    polyphony_score = min(
        1.0,
        float(aggregate_eval.get("mean_mean_active_polyphony", 0.0))
        / max(1e-6, float(cfg.sample_score_polyphony_target)),
    )
    fullness_score = float(aggregate_eval.get("mean_fullness_score", 0.0))
    rms_score = min(1.0, float(aggregate_eval.get("mean_wav_rms", 0.0)) / max(1e-6, float(cfg.sample_score_rms_target)))
    nps = float(aggregate_eval.get("mean_notes_per_second", 0.0))
    single_pc = float(aggregate_eval.get("mean_single_pitch_class_fraction", 0.0))
    mid_fraction = float(aggregate_eval.get("mean_mid_note_fraction", 0.0))
    high_fraction = float(aggregate_eval.get("mean_high_note_fraction", 0.0))
    min_section_notes = float(aggregate_eval.get("mean_min_section_notes", 0.0))
    min_section_unique = float(aggregate_eval.get("mean_min_section_unique_pitches", 0.0))
    min_section_chord = float(aggregate_eval.get("mean_min_section_chord_frame_fraction", 0.0))
    min_section_fullness = float(aggregate_eval.get("mean_min_section_fullness_score", 0.0))
    density_penalty = max(
        0.0,
        (float(cfg.sample_score_min_notes_per_second) - nps)
        / max(1e-6, float(cfg.sample_score_min_notes_per_second)),
    )
    rms_penalty = max(
        0.0,
        (float(cfg.sample_score_rms_target) - float(aggregate_eval.get("mean_wav_rms", 0.0)))
        / max(1e-6, float(cfg.sample_score_rms_target)),
    )
    pc_penalty = max(
        0.0,
        (single_pc - float(cfg.sample_score_max_single_pitch_class_fraction))
        / max(1e-6, 1.0 - float(cfg.sample_score_max_single_pitch_class_fraction)),
    )
    mid_penalty = max(
        0.0,
        (float(cfg.sample_score_min_mid_note_fraction) - mid_fraction)
        / max(1e-6, float(cfg.sample_score_min_mid_note_fraction)),
    )
    high_penalty = max(
        0.0,
        (high_fraction - float(cfg.sample_score_max_high_note_fraction))
        / max(1e-6, 1.0 - float(cfg.sample_score_max_high_note_fraction)),
    )
    section_notes_penalty = max(
        0.0,
        (float(cfg.sample_score_min_section_notes) - min_section_notes)
        / max(1e-6, float(cfg.sample_score_min_section_notes)),
    )
    section_unique_penalty = max(
        0.0,
        (float(cfg.sample_score_min_section_unique_pitches) - min_section_unique)
        / max(1e-6, float(cfg.sample_score_min_section_unique_pitches)),
    )
    section_chord_penalty = max(
        0.0,
        (float(cfg.sample_score_min_section_chord_frame) - min_section_chord)
        / max(1e-6, float(cfg.sample_score_min_section_chord_frame)),
    )
    section_fullness_penalty = max(
        0.0,
        (float(cfg.sample_score_min_section_fullness) - min_section_fullness)
        / max(1e-6, float(cfg.sample_score_min_section_fullness)),
    )
    quality_penalty_raw = (
        density_penalty
        + rms_penalty
        + pc_penalty
        + mid_penalty
        + high_penalty
        + section_notes_penalty
        + section_unique_penalty
        + section_chord_penalty
        + section_fullness_penalty
    )
    quality_penalty = float(cfg.sample_score_quality_penalty_weight) * quality_penalty_raw
    role_balance_raw = (
        chord_score
        + melody_score
        + bass_score
        + polyphony_score
        + fullness_score
        + rms_score
    ) / 6.0
    role_balance_score = float(cfg.sample_score_role_balance_weight) * role_balance_raw
    total = pass_score - warning_penalty - quality_penalty + source_score + target_score + role_balance_score
    return {
        "pass_score": float(pass_score),
        "warning_penalty": float(warning_penalty),
        "quality_penalty": float(quality_penalty),
        "quality_penalty_raw": float(quality_penalty_raw),
        "density_penalty": float(density_penalty),
        "rms_penalty": float(rms_penalty),
        "pitch_class_penalty": float(pc_penalty),
        "mid_note_penalty": float(mid_penalty),
        "high_note_penalty": float(high_penalty),
        "section_notes_penalty": float(section_notes_penalty),
        "section_unique_penalty": float(section_unique_penalty),
        "section_chord_penalty": float(section_chord_penalty),
        "section_fullness_penalty": float(section_fullness_penalty),
        "source_score": float(source_score),
        "target_score": float(target_score),
        "role_balance_raw": float(role_balance_raw),
        "role_balance_score": float(role_balance_score),
        "total": float(total),
    }


def train_piano_roll_model(cfg: PianoTrainConfig = PianoTrainConfig()) -> Dict[str, Any]:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    device = _device(str(cfg.device))
    cache_dir = Path(cfg.cache_dir)
    meta_path = cache_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    ds = PianoRollCacheDataset(cache_dir)
    if len(ds) == 0:
        raise ValueError(f"Empty piano cache: {cache_dir}")

    loader = DataLoader(ds, batch_size=max(1, int(cfg.batch_size)), shuffle=True, drop_last=False)
    model_cfg = PianoRollModelConfig(
        in_channels=int(ds.source.shape[1]),
        hidden_channels=int(cfg.hidden_channels),
        n_keys=int(ds.onset.shape[1]),
        n_blocks=int(cfg.n_blocks),
        dropout=float(cfg.dropout),
        architecture=str(cfg.model_architecture),
        key_embed_dim=int(cfg.key_embed_dim),
    )
    model = PianoRollGenerator(model_cfg).to(device)
    warm_start_info: Dict[str, Any] = {"enabled": False}
    if cfg.warm_start_checkpoint is not None and str(cfg.warm_start_checkpoint) != "":
        warm_start_info = {
            "enabled": True,
            **_load_warm_start_checkpoint(
                model,
                Path(cfg.warm_start_checkpoint),
                device=device,
            ),
        }
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    run_dir = _new_run_dir(cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    started = time.time()
    best_sample_eval_score = float("-inf")
    best_sample_eval_epoch = 0

    for epoch in range(1, int(cfg.epochs) + 1):
        epoch_started = time.time()
        totals: Dict[str, float] = {
            "loss": 0.0,
            "onset_loss": 0.0,
            "frame_loss": 0.0,
            "velocity_loss": 0.0,
            "pedal_loss": 0.0,
            "density_loss": 0.0,
            "chroma_loss": 0.0,
            "pitch_usage_loss": 0.0,
            "hierarchy_loss": 0.0,
            "musical_plan_loss": 0.0,
            "event_plan_loss": 0.0,
            "pc_onset_plan_loss": 0.0,
            "pc_onset_distribution_loss": 0.0,
            "pc_onset_f1_loss": 0.0,
            "pc_onset_alignment_loss": 0.0,
            "role_plan_loss": 0.0,
            "texture_balance_loss": 0.0,
            "melody_plan_loss": 0.0,
            "melody_balance_loss": 0.0,
            "texture_role_plan_loss": 0.0,
            "texture_role_balance_loss": 0.0,
            "section_role_plan_loss": 0.0,
            "section_role_balance_loss": 0.0,
            "arranger_state_plan_loss": 0.0,
            "bass_continuity_plan_loss": 0.0,
            "body_melody_state_plan_loss": 0.0,
            "body_melody_state_balance_loss": 0.0,
            "section_diversity_plan_loss": 0.0,
            "section_diversity_balance_loss": 0.0,
            "anti_collapse_loss": 0.0,
            "source_onset_loss": 0.0,
            "source_chroma_loss": 0.0,
            "harmonic_plan_loss": 0.0,
        }
        batches = 0
        for batch in loader:
            source = batch["source"].to(device)
            target_onset = batch["onset"].to(device)
            target_frame = batch["frame"].to(device)
            target_velocity = batch["velocity"].to(device)
            target_pedal = batch["pedal"].to(device)
            target_density = batch["density"].to(device)
            target_register = batch["register"].to(device)
            target_chord = batch["chord"].to(device)
            target_bass = batch["bass"].to(device)
            target_voicing = batch["voicing"].to(device)
            target_event = batch["event"].to(device)
            target_pc_onset = batch["pc_onset"].to(device)
            target_role = batch["role"].to(device)
            target_melody = batch["melody"].to(device)
            target_texture_role = batch["texture_role"].to(device)
            target_section_role = batch["section_role"].to(device)
            target_arranger_state = batch["arranger_state"].to(device)
            target_bass_continuity = batch["bass_continuity"].to(device)
            target_body_melody_state = batch["body_melody_state"].to(device)
            target_section_diversity = batch["section_diversity"].to(device)
            source_onset = source[:, 12, :]
            source_chroma = source[:, :12, :]
            pred = model(source)
            losses = piano_roll_loss(
                pred,
                target_onset,
                target_frame,
                target_velocity,
                target_pedal,
                target_density=target_density,
                target_register=target_register,
                target_chord=target_chord,
                target_bass=target_bass,
                target_voicing=target_voicing,
                target_event=target_event,
                target_pc_onset=target_pc_onset,
                target_role=target_role,
                target_melody=target_melody,
                target_texture_role=target_texture_role,
                target_section_role=target_section_role,
                target_arranger_state=target_arranger_state,
                target_bass_continuity=target_bass_continuity,
                target_body_melody_state=target_body_melody_state,
                target_section_diversity=target_section_diversity,
                source_onset=source_onset,
                source_chroma=source_chroma,
                density_weight=float(cfg.density_loss_weight),
                chroma_weight=float(cfg.chroma_loss_weight),
                pitch_usage_weight=float(cfg.pitch_usage_loss_weight),
                hierarchy_weight=float(cfg.hierarchy_loss_weight),
                musical_plan_weight=float(cfg.musical_plan_loss_weight),
                event_plan_weight=float(cfg.event_plan_loss_weight),
                pc_onset_plan_weight=float(cfg.pc_onset_plan_loss_weight),
                pc_onset_f1_weight=float(cfg.pc_onset_f1_loss_weight),
                pc_onset_alignment_weight=float(cfg.pc_onset_alignment_loss_weight),
                role_plan_weight=float(cfg.role_plan_loss_weight),
                texture_balance_weight=float(cfg.texture_balance_loss_weight),
                melody_plan_weight=float(cfg.melody_plan_loss_weight),
                melody_balance_weight=float(cfg.melody_balance_loss_weight),
                texture_role_plan_weight=float(cfg.texture_role_plan_loss_weight),
                texture_role_balance_weight=float(cfg.texture_role_balance_loss_weight),
                section_role_plan_weight=float(cfg.section_role_plan_loss_weight),
                section_role_balance_weight=float(cfg.section_role_balance_loss_weight),
                arranger_state_plan_weight=float(cfg.arranger_state_plan_loss_weight),
                bass_continuity_plan_weight=float(cfg.bass_continuity_plan_loss_weight),
                body_melody_state_plan_weight=float(cfg.body_melody_state_plan_loss_weight),
                body_melody_state_balance_weight=float(cfg.body_melody_state_balance_loss_weight),
                section_diversity_plan_weight=float(cfg.section_diversity_plan_loss_weight),
                section_diversity_balance_weight=float(cfg.section_diversity_balance_loss_weight),
                anti_collapse_weight=float(cfg.anti_collapse_loss_weight),
                source_onset_weight=float(cfg.source_onset_loss_weight),
                source_chroma_weight=float(cfg.source_chroma_loss_weight),
                harmonic_plan_weight=float(cfg.harmonic_plan_loss_weight),
                piano_min_midi=PIANO_MIN_MIDI,
            )
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item())
            batches += 1
            if int(cfg.max_batches_per_epoch) > 0 and batches >= int(cfg.max_batches_per_epoch):
                break
        row = {
            "epoch": int(epoch),
            "batches": int(batches),
            **{k: float(v / max(1, batches)) for k, v in totals.items()},
            "epoch_seconds": float(time.time() - epoch_started),
        }
        if int(cfg.sample_every) > 0 and epoch % int(cfg.sample_every) == 0:
            sample_summary = _write_epoch_samples(model, ds, run_dir, epoch, cfg=cfg, device=device, meta=meta)
            row["sample_count"] = int(sample_summary["n_samples"])
            aggregate_eval = sample_summary.get("aggregate_eval", {})
            if aggregate_eval:
                row["sample_eval_pass_count"] = int(aggregate_eval.get("sample_eval_pass_count", 0))
                row["sample_eval_warning_counts"] = aggregate_eval.get("warning_counts", {})
                for key, value in aggregate_eval.items():
                    if str(key).startswith("mean_"):
                        row[f"sample_eval_{key}"] = value
                score_components = _sample_eval_score_components(aggregate_eval, cfg)
                row["sample_eval_score_components"] = score_components
                row["sample_eval_score"] = float(score_components["total"])
        history.append(row)
        _write_json(run_dir / "history.json", {"history": history})
        checkpoint_payload = {
            "model": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(cfg),
            "cache_meta": meta,
            "warm_start": warm_start_info,
            "epoch": int(epoch),
            "history": history,
        }
        torch.save(checkpoint_payload, ckpt_dir / "latest.pt")
        if "sample_eval_score" in row and float(row["sample_eval_score"]) > float(best_sample_eval_score):
            best_sample_eval_score = float(row["sample_eval_score"])
            best_sample_eval_epoch = int(epoch)
            best_payload = dict(checkpoint_payload)
            best_payload["best_sample_eval_score"] = float(best_sample_eval_score)
            best_payload["best_sample_eval_epoch"] = int(best_sample_eval_epoch)
            torch.save(best_payload, ckpt_dir / "best_sample_eval.pt")

    summary = {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "n_samples": int(len(ds)),
        "history": history,
        "latest_checkpoint": str(ckpt_dir / "latest.pt"),
        "best_sample_eval_checkpoint": str(ckpt_dir / "best_sample_eval.pt") if best_sample_eval_epoch > 0 else "",
        "best_sample_eval_epoch": int(best_sample_eval_epoch),
        "best_sample_eval_score": float(best_sample_eval_score) if best_sample_eval_epoch > 0 else None,
        "warm_start": warm_start_info,
        "seconds": float(time.time() - started),
        "device": str(device),
        "source_features": meta.get("source_features", SOURCE_FEATURE_NAMES),
    }
    _write_json(run_dir / "summary.json", summary)
    return summary


__all__ = [
    "DEFAULT_PIANO_RUN_ROOT",
    "PianoDecodeConfig",
    "PianoRollCacheDataset",
    "PianoTrainConfig",
    "prediction_to_arrangement",
    "train_piano_roll_model",
]
