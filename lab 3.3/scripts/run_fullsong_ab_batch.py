from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB31_SCRIPTS = REPO_ROOT / "lab 3.1" / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LAB31_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB31_SCRIPTS))

import diffusion_downloads_batch as ddb


TARGET_GENRES = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]


def _slug(value: str) -> str:
    chars: List[str] = []
    for ch in value.lower():
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported value: {type(value)!r}")


@dataclass
class FullsongBatchConfig:
    downloads_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    output_root: Path = field(default_factory=lambda: Path.home() / "Desktop" / "dggr_fullsong_ab_batch")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt")
    lab1_checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt")
    n_songs: int = 12
    min_duration_seconds: float = 120.0
    max_duration_seconds: float = 360.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    ddim_steps: int = 50
    assemble_domain: str = "mel"
    device: str = "auto"
    seed: int = 328

    def materialize(self) -> "FullsongBatchConfig":
        self.downloads_dir = Path(self.downloads_dir)
        self.output_root = Path(self.output_root)
        self.cache_dir = Path(self.cache_dir)
        self.checkpoint = Path(self.checkpoint)
        self.lab1_checkpoint = Path(self.lab1_checkpoint)
        return self


def settings_ab() -> List[Dict[str, Any]]:
    return [
        {
            "label": "a_best_mid_275_b_style",
            "note": "Style-forward winner.",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 2.05,
            "style_strength": 0.74,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.41,
            "source_mel_blend": 0.07,
            "hf_source_blend": 0.16,
            "hf_start_bin": 56,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
        {
            "label": "b_best_mid_275_b_dewarble",
            "note": "Cleaner fallback with more dewarble control.",
            "t_start": 275,
            "t_start_end": 202,
            "reanchor_every": 3,
            "reanchor_t_start": 170,
            "guidance_scale": 1.95,
            "style_strength": 0.70,
            "prefix_blend": 1.0,
            "source_prefix_blend": 0.43,
            "source_mel_blend": 0.08,
            "hf_source_blend": 0.20,
            "hf_start_bin": 54,
            "mel_time_smooth": 3,
            "mel_freq_smooth": 0,
        },
    ]


def choose_songs(cfg: FullsongBatchConfig) -> List[Dict[str, Any]]:
    rows = ddb.discover_download_audio(cfg.downloads_dir)
    usable: List[Dict[str, Any]] = []
    for row in rows:
        duration = row.get("duration_seconds")
        if duration is None:
            continue
        duration = float(duration)
        if duration < float(cfg.min_duration_seconds) or duration > float(cfg.max_duration_seconds):
            continue
        path = Path(str(row["path"]))
        usable.append(
            {
                "path": path,
                "duration_seconds": duration,
                "source_genre": str(ddb.infer_source_genre(path)),
            }
        )
    rng = random.Random(cfg.seed)
    rng.shuffle(usable)
    chosen = usable[: int(cfg.n_songs)]
    if len(chosen) < int(cfg.n_songs):
        raise RuntimeError(f"Only found {len(chosen)} usable songs in {cfg.downloads_dir}")
    return chosen


def run_job(cfg: FullsongBatchConfig, setting: Dict[str, Any], song: Dict[str, Any], target_genre: str, out_dir: Path, seed: int) -> Dict[str, Any]:
    generated = out_dir / "longform_coherent.wav"
    metrics = out_dir / "coherence_metrics.json"
    if generated.exists() and metrics.exists():
        return {
            "status": "reused",
            "output_dir": out_dir,
            "generated_wav": generated,
            "metrics_json": metrics,
        }

    cmd = [
        "python",
        str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
        "--cache-dir", str(cfg.cache_dir),
        "--checkpoint", str(cfg.checkpoint),
        "--lab1-checkpoint", str(cfg.lab1_checkpoint),
        "--source-audio", str(song["path"]),
        "--source-genre", str(song["source_genre"]),
        "--target-genre", str(target_genre),
        "--source-start-sec", "0.0",
        "--source-seconds", "-1",
        "--out-dir", str(out_dir),
        "--chunk-seconds", str(cfg.chunk_seconds),
        "--overlap-seconds", str(cfg.overlap_seconds),
        "--n-frames", str(cfg.n_frames),
        "--ddim-steps", str(cfg.ddim_steps),
        "--assemble-domain", str(cfg.assemble_domain),
        "--device", str(cfg.device),
        "--seed", str(seed),
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
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Longform job failed for {song['path']} -> {target_genre} [{setting['label']}]")
    return {
        "status": "generated",
        "output_dir": out_dir,
        "generated_wav": generated,
        "metrics_json": metrics,
    }


def main() -> None:
    cfg = FullsongBatchConfig().materialize()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = cfg.output_root / f"fullsong_ab_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    songs = choose_songs(cfg)
    settings = settings_ab()

    (out_root / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")
    (out_root / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (out_root / "songs.json").write_text(json.dumps([
        {"path": str(row["path"]), "duration_seconds": row["duration_seconds"], "source_genre": row["source_genre"]}
        for row in songs
    ], indent=2), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    job_idx = 0
    for setting in settings:
        for song in songs:
            for target_genre in TARGET_GENRES:
                song_stem = _slug(song["path"].stem)[:56]
                out_dir = out_root / "clips" / setting["label"] / f"{job_idx:03d}_{song_stem}__to__{_slug(target_genre)}"
                result = run_job(cfg, setting, song, target_genre, out_dir, seed=cfg.seed + job_idx)
                manifest_rows.append(
                    {
                        "job_idx": job_idx,
                        "setting_label": setting["label"],
                        "source_audio": str(song["path"]),
                        "source_genre": song["source_genre"],
                        "target_genre": target_genre,
                        "duration_seconds": song["duration_seconds"],
                        "output_dir": str(result["output_dir"]),
                        "generated_wav": str(result["generated_wav"]),
                        "metrics_json": str(result["metrics_json"]),
                        "status": result["status"],
                    }
                )
                with (out_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(manifest_rows)
                job_idx += 1

    summary = {
        "output_dir": str(out_root),
        "settings": [row["label"] for row in settings],
        "n_songs": len(songs),
        "target_genres": TARGET_GENRES,
        "total_jobs": len(manifest_rows),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
