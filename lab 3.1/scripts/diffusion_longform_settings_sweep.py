from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import diffusion_longform_compare as dlc


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _safe_console_text(text: str) -> str:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii", errors="replace")


@dataclass
class DiffusionSettingsSweepConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    downloads_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    output_root: Path = field(
        default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_settings_sweep"
    )
    run_dir: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_vocal_crackle_retool" / "run_20260324_221729")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    checkpoint_path: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_vocal_crackle_retool" / "run_20260324_221729" / "checkpoints" / "epoch_001.pt")
    lab1_checkpoint: Path = field(
        default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    )
    n_songs: int = 2
    targets_per_song: int = 2
    source_seconds: float = 45.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    ddim_steps: int = 50
    assemble_domain: str = "mel"
    device: str = "auto"
    seed: int = 328

    def materialize(self) -> "DiffusionSettingsSweepConfig":
        self.downloads_dir = Path(self.downloads_dir)
        self.output_root = Path(self.output_root)
        self.run_dir = Path(self.run_dir)
        self.cache_dir = Path(self.cache_dir)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.lab1_checkpoint = Path(self.lab1_checkpoint)
        return self


def default_settings_panel() -> List[Dict[str, Any]]:
    return [
        {
            "label": "stable_clean_a",
            "note": "Most source-anchored; lowest crackle risk.",
            "t_start": 210,
            "t_start_end": 170,
            "reanchor_every": 3,
            "reanchor_t_start": 150,
            "guidance_scale": 1.60,
            "style_strength": 0.50,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.50,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.24,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "stable_clean_b",
            "note": "Slightly more movement, still strongly anchored.",
            "t_start": 230,
            "t_start_end": 180,
            "reanchor_every": 4,
            "reanchor_t_start": 160,
            "guidance_scale": 1.75,
            "style_strength": 0.56,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.48,
            "source_mel_blend": 0.11,
            "hf_source_blend": 0.22,
            "hf_start_bin": 55,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "balanced_a",
            "note": "Balanced content and style.",
            "t_start": 250,
            "t_start_end": 190,
            "reanchor_every": 4,
            "reanchor_t_start": 165,
            "guidance_scale": 1.90,
            "style_strength": 0.62,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.45,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.20,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "balanced_b",
            "note": "A little more freedom than balanced_a.",
            "t_start": 265,
            "t_start_end": 195,
            "reanchor_every": 4,
            "reanchor_t_start": 170,
            "guidance_scale": 2.00,
            "style_strength": 0.66,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.43,
            "source_mel_blend": 0.09,
            "hf_source_blend": 0.18,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_push_a",
            "note": "Higher style shift with moderate HF protection.",
            "t_start": 285,
            "t_start_end": 205,
            "reanchor_every": 4,
            "reanchor_t_start": 175,
            "guidance_scale": 2.10,
            "style_strength": 0.72,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.41,
            "source_mel_blend": 0.08,
            "hf_source_blend": 0.16,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_push_b",
            "note": "Aggressive style push; still keeps frequent reanchors.",
            "t_start": 305,
            "t_start_end": 215,
            "reanchor_every": 3,
            "reanchor_t_start": 180,
            "guidance_scale": 2.20,
            "style_strength": 0.78,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.39,
            "source_mel_blend": 0.07,
            "hf_source_blend": 0.15,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "vocal_safe",
            "note": "Extra source anchoring for vocals and body.",
            "t_start": 235,
            "t_start_end": 185,
            "reanchor_every": 3,
            "reanchor_t_start": 155,
            "guidance_scale": 1.80,
            "style_strength": 0.58,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.52,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.20,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hf_safe",
            "note": "Extra high-frequency protection against crackle.",
            "t_start": 245,
            "t_start_end": 190,
            "reanchor_every": 4,
            "reanchor_t_start": 160,
            "guidance_scale": 1.85,
            "style_strength": 0.60,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.47,
            "source_mel_blend": 0.11,
            "hf_source_blend": 0.26,
            "hf_start_bin": 52,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "low_noise_content",
            "note": "Lowest noise freedom; strongest content retention.",
            "t_start": 190,
            "t_start_end": 160,
            "reanchor_every": 3,
            "reanchor_t_start": 145,
            "guidance_scale": 1.70,
            "style_strength": 0.52,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.53,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.22,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "reanchor_dense",
            "note": "Frequent reset schedule to fight compounding drift.",
            "t_start": 260,
            "t_start_end": 190,
            "reanchor_every": 2,
            "reanchor_t_start": 155,
            "guidance_scale": 1.95,
            "style_strength": 0.64,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.46,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.18,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_reanchor",
            "note": "Style-forward, but with denser reanchors to keep it sane.",
            "t_start": 295,
            "t_start_end": 205,
            "reanchor_every": 2,
            "reanchor_t_start": 170,
            "guidance_scale": 2.05,
            "style_strength": 0.74,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.41,
            "source_mel_blend": 0.08,
            "hf_source_blend": 0.16,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "mellow_style",
            "note": "Softer guidance with moderate style push.",
            "t_start": 255,
            "t_start_end": 190,
            "reanchor_every": 4,
            "reanchor_t_start": 165,
            "guidance_scale": 1.80,
            "style_strength": 0.65,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.44,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.19,
            "hf_start_bin": 56,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
    ]


def vocal_low_noise_settings_panel() -> List[Dict[str, Any]]:
    return [
        {
            "label": "low_noise_vocal_a",
            "note": "Very conservative baseline around low-noise plus vocal-safe anchoring.",
            "t_start": 180,
            "t_start_end": 150,
            "reanchor_every": 3,
            "reanchor_t_start": 145,
            "guidance_scale": 1.58,
            "style_strength": 0.50,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.56,
            "source_mel_blend": 0.14,
            "hf_source_blend": 0.28,
            "hf_start_bin": 52,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "low_noise_vocal_b",
            "note": "Same family, slightly more style and slightly less HF correction.",
            "t_start": 190,
            "t_start_end": 155,
            "reanchor_every": 3,
            "reanchor_t_start": 145,
            "guidance_scale": 1.64,
            "style_strength": 0.53,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.55,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.26,
            "hf_start_bin": 53,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "low_noise_vocal_c",
            "note": "A touch more freedom while staying close to the same regime.",
            "t_start": 200,
            "t_start_end": 160,
            "reanchor_every": 3,
            "reanchor_t_start": 148,
            "guidance_scale": 1.68,
            "style_strength": 0.55,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.54,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.24,
            "hf_start_bin": 53,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "low_noise_vocal_d",
            "note": "Upper edge of the low-noise family before it becomes more style-forward.",
            "t_start": 208,
            "t_start_end": 164,
            "reanchor_every": 3,
            "reanchor_t_start": 150,
            "guidance_scale": 1.72,
            "style_strength": 0.57,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.53,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.23,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "vocal_iso_a",
            "note": "Stronger source-body preservation for vocal intelligibility.",
            "t_start": 195,
            "t_start_end": 158,
            "reanchor_every": 2,
            "reanchor_t_start": 148,
            "guidance_scale": 1.66,
            "style_strength": 0.52,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.57,
            "source_mel_blend": 0.15,
            "hf_source_blend": 0.24,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "vocal_iso_b",
            "note": "Vocal-focused with slightly looser prefix anchor.",
            "t_start": 205,
            "t_start_end": 164,
            "reanchor_every": 2,
            "reanchor_t_start": 150,
            "guidance_scale": 1.72,
            "style_strength": 0.55,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.55,
            "source_mel_blend": 0.14,
            "hf_source_blend": 0.23,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "vocal_iso_c",
            "note": "Voice-safe but lets the chunk move a little more on style.",
            "t_start": 215,
            "t_start_end": 170,
            "reanchor_every": 2,
            "reanchor_t_start": 152,
            "guidance_scale": 1.78,
            "style_strength": 0.58,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.53,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.22,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "vocal_iso_d",
            "note": "Highest style in the vocal-isolation cluster.",
            "t_start": 225,
            "t_start_end": 176,
            "reanchor_every": 2,
            "reanchor_t_start": 154,
            "guidance_scale": 1.82,
            "style_strength": 0.60,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.52,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.21,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hf_guard_a",
            "note": "Most HF-protected variant in the panel.",
            "t_start": 190,
            "t_start_end": 158,
            "reanchor_every": 3,
            "reanchor_t_start": 145,
            "guidance_scale": 1.62,
            "style_strength": 0.51,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.55,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.30,
            "hf_start_bin": 50,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hf_guard_b",
            "note": "Strong HF damping with a little more style freedom.",
            "t_start": 198,
            "t_start_end": 162,
            "reanchor_every": 3,
            "reanchor_t_start": 148,
            "guidance_scale": 1.68,
            "style_strength": 0.54,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.54,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.28,
            "hf_start_bin": 51,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hf_guard_c",
            "note": "Balances HF protection and moderate style push.",
            "t_start": 208,
            "t_start_end": 168,
            "reanchor_every": 3,
            "reanchor_t_start": 150,
            "guidance_scale": 1.74,
            "style_strength": 0.57,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.53,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.26,
            "hf_start_bin": 52,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "hf_guard_d",
            "note": "Still HF-safe, but near the most style-forward edge of that family.",
            "t_start": 218,
            "t_start_end": 174,
            "reanchor_every": 3,
            "reanchor_t_start": 152,
            "guidance_scale": 1.80,
            "style_strength": 0.60,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.52,
            "source_mel_blend": 0.11,
            "hf_source_blend": 0.24,
            "hf_start_bin": 53,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "dense_voice_a",
            "note": "Dense reanchors around a low-noise vocal-safe setting.",
            "t_start": 200,
            "t_start_end": 162,
            "reanchor_every": 2,
            "reanchor_t_start": 145,
            "guidance_scale": 1.68,
            "style_strength": 0.54,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.55,
            "source_mel_blend": 0.13,
            "hf_source_blend": 0.24,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "dense_voice_b",
            "note": "Dense reset schedule with slightly higher style strength.",
            "t_start": 210,
            "t_start_end": 168,
            "reanchor_every": 2,
            "reanchor_t_start": 148,
            "guidance_scale": 1.74,
            "style_strength": 0.57,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.54,
            "source_mel_blend": 0.12,
            "hf_source_blend": 0.23,
            "hf_start_bin": 54,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "dense_voice_c",
            "note": "More aggressive within the dense-reanchor family.",
            "t_start": 220,
            "t_start_end": 174,
            "reanchor_every": 2,
            "reanchor_t_start": 150,
            "guidance_scale": 1.80,
            "style_strength": 0.60,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.52,
            "source_mel_blend": 0.11,
            "hf_source_blend": 0.21,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "dense_voice_d",
            "note": "Highest-style dense reanchor variant before getting risky.",
            "t_start": 230,
            "t_start_end": 180,
            "reanchor_every": 2,
            "reanchor_t_start": 152,
            "guidance_scale": 1.86,
            "style_strength": 0.63,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.50,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.20,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_safe_a",
            "note": "Pushes style a bit harder while keeping low-noise safeguards.",
            "t_start": 215,
            "t_start_end": 170,
            "reanchor_every": 3,
            "reanchor_t_start": 150,
            "guidance_scale": 1.82,
            "style_strength": 0.59,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.51,
            "source_mel_blend": 0.11,
            "hf_source_blend": 0.21,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_safe_b",
            "note": "Slightly more style pressure with still-dense source correction.",
            "t_start": 225,
            "t_start_end": 176,
            "reanchor_every": 3,
            "reanchor_t_start": 152,
            "guidance_scale": 1.88,
            "style_strength": 0.62,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.50,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.20,
            "hf_start_bin": 55,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_safe_c",
            "note": "Upper style-safe edge with frequent enough correction to avoid static.",
            "t_start": 235,
            "t_start_end": 182,
            "reanchor_every": 3,
            "reanchor_t_start": 155,
            "guidance_scale": 1.92,
            "style_strength": 0.64,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.49,
            "source_mel_blend": 0.10,
            "hf_source_blend": 0.19,
            "hf_start_bin": 56,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
        {
            "label": "style_safe_d",
            "note": "Most style-forward member of this low-noise family.",
            "t_start": 240,
            "t_start_end": 185,
            "reanchor_every": 3,
            "reanchor_t_start": 158,
            "guidance_scale": 1.96,
            "style_strength": 0.66,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.48,
            "source_mel_blend": 0.09,
            "hf_source_blend": 0.18,
            "hf_start_bin": 56,
            "mel_time_smooth": 2,
            "mel_freq_smooth": 0,
        },
    ]


def resolve_epoch_panel(run_dir: Path, epoch_names: Sequence[str] = ("epoch_001.pt", "epoch_002.pt", "epoch_003.pt")) -> List[Path]:
    ckpt_dir = Path(run_dir) / "checkpoints"
    panel: List[Path] = []
    for name in epoch_names:
        ckpt = ckpt_dir / name
        if ckpt.exists():
            panel.append(ckpt)
    return panel


def _write_manifest(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "setting_label",
        "setting_note",
        "job_idx",
        "song_idx",
        "source_audio",
        "source_genre",
        "target_genre",
        "start_sec",
        "source_seconds",
        "duration_seconds",
        "size_bytes",
        "output_dir",
        "source_wav",
        "generated_wav",
        "metrics_json",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serial = dict(row)
            for key, value in list(serial.items()):
                if isinstance(value, Path):
                    serial[key] = str(value)
            writer.writerow(serial)
    return path


def _run_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(str(x) for x in cmd)}")


def run_settings_sweep(
    cfg: DiffusionSettingsSweepConfig,
    settings_panel: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = cfg.materialize()
    out_dir = cfg.output_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan_cfg = dlc.DiffusionLongformCompareConfig(
        downloads_dir=cfg.downloads_dir,
        output_root=cfg.output_root,
        run_dir=cfg.run_dir,
        cache_dir=cfg.cache_dir,
        lab1_checkpoint=cfg.lab1_checkpoint,
        n_songs=cfg.n_songs,
        targets_per_song=cfg.targets_per_song,
        source_seconds=cfg.source_seconds,
        chunk_seconds=cfg.chunk_seconds,
        overlap_seconds=cfg.overlap_seconds,
        n_frames=cfg.n_frames,
        ddim_steps=cfg.ddim_steps,
        assemble_domain=cfg.assemble_domain,
        device=cfg.device,
        seed=cfg.seed,
        snapshot_latest_checkpoint=False,
    )
    jobs = dlc.plan_longform_jobs(plan_cfg)
    (out_dir / "jobs.json").write_text(json.dumps(jobs, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "settings_panel.json").write_text(
        json.dumps(list(settings_panel), indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    for setting in settings_panel:
        label = str(setting["label"])
        note = str(setting.get("note", ""))
        print(f"\n=== setting: {label} ===")
        print(note)
        for job in jobs:
            source_audio = Path(job["source_audio"])
            job_tag = f"{int(job['job_idx']):02d}_{dlc.ddb._slug(source_audio.stem)[:40]}__to__{dlc.ddb._slug(str(job['target_genre']))}"
            job_out_dir = out_dir / "clips" / label / job_tag
            generated_wav = job_out_dir / "longform_coherent.wav"
            metrics_json = job_out_dir / "coherence_metrics.json"
            cmd = [
                "python",
                str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
                "--cache-dir", str(cfg.cache_dir),
                "--checkpoint", str(cfg.checkpoint_path),
                "--lab1-checkpoint", str(cfg.lab1_checkpoint),
                "--source-audio", str(source_audio),
                "--source-genre", str(job["source_genre"]),
                "--target-genre", str(job["target_genre"]),
                "--source-start-sec", str(job["start_sec"]),
                "--source-seconds", str(cfg.source_seconds),
                "--out-dir", str(job_out_dir),
                "--chunk-seconds", str(cfg.chunk_seconds),
                "--overlap-seconds", str(cfg.overlap_seconds),
                "--n-frames", str(cfg.n_frames),
                "--ddim-steps", str(cfg.ddim_steps),
                "--assemble-domain", str(cfg.assemble_domain),
                "--device", str(cfg.device),
                "--seed", str(cfg.seed + int(job["job_idx"])),
                "--t-start", str(setting["t_start"]),
                "--t-start-end", str(setting["t_start_end"]),
                "--reanchor-every", str(setting["reanchor_every"]),
                "--reanchor-t-start", str(setting["reanchor_t_start"]),
                "--guidance-scale", str(setting["guidance_scale"]),
                "--style-strength", str(setting["style_strength"]),
                "--prefix-blend", str(setting["prefix_blend"]),
                "--source-prefix-blend", str(setting["source_prefix_blend"]),
                "--source-mel-blend", str(setting["source_mel_blend"]),
                "--hf-source-blend", str(setting["hf_source_blend"]),
                "--hf-start-bin", str(setting["hf_start_bin"]),
                "--mel-time-smooth", str(setting["mel_time_smooth"]),
                "--mel-freq-smooth", str(setting["mel_freq_smooth"]),
            ]
            log_path = logs_dir / f"{label}__{job_tag}.log"
            print(
                _safe_console_text(
                    f"[{int(job['job_idx']) + 1:02d}/{len(jobs)}] "
                    f"{source_audio.name} {float(job['start_sec']):.1f}s -> {job['target_genre']}"
                )
            )
            if generated_wav.exists() and metrics_json.exists():
                print("  reusing existing output")
            else:
                _run_command(cmd, cwd=REPO_ROOT, log_path=log_path)
            manifest_rows.append(
                {
                    "setting_label": label,
                    "setting_note": note,
                    **job,
                    "output_dir": job_out_dir,
                    "source_wav": job_out_dir / "source.wav",
                    "generated_wav": generated_wav,
                    "metrics_json": metrics_json,
                    "checkpoint_path": cfg.checkpoint_path,
                }
            )

    manifest_path = _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "tag": cfg.tag,
        "output_dir": out_dir,
        "run_dir": cfg.run_dir,
        "cache_dir": cfg.cache_dir,
        "checkpoint_path": cfg.checkpoint_path,
        "jobs_path": out_dir / "jobs.json",
        "settings_panel_path": out_dir / "settings_panel.json",
        "manifest_path": manifest_path,
        "n_jobs": len(jobs),
        "n_settings": len(settings_panel),
        "total_runs": len(jobs) * len(settings_panel),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary
