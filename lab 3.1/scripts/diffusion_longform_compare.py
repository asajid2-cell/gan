from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import diffusion_downloads_batch as ddb


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _pick_longform_targets(source_genre: str, max_targets: int = 2) -> List[str]:
    preferred = ["baroque_classical", "lofi_hh_lfbb", "hiphop_xtc", "cc0_other"]
    targets = [g for g in preferred if g != source_genre]
    return targets[: max(1, int(max_targets))]


@dataclass
class DiffusionLongformCompareConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    downloads_dir: Path = field(default_factory=lambda: ddb.DEFAULT_DOWNLOADS)
    output_root: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_compare")
    run_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    lab1_checkpoint: Path = field(
        default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    )
    n_songs: int = 2
    targets_per_song: int = 2
    source_seconds: float = 45.0
    chunk_seconds: float = 3.0
    overlap_seconds: float = 0.5
    n_frames: int = 256
    t_start: int = 240
    t_start_end: int = 180
    reanchor_every: int = 4
    reanchor_t_start: int = 160
    ddim_steps: int = 50
    guidance_scale: float = 1.75
    style_strength: float = 0.60
    prefix_blend: float = 1.0
    source_prefix_blend: float = 0.45
    source_mel_blend: float = 0.10
    hf_source_blend: float = 0.18
    hf_start_bin: int = 56
    mel_time_smooth: int = 3
    mel_freq_smooth: int = 0
    assemble_domain: str = "mel"
    device: str = "auto"
    seed: int = 328
    snapshot_latest_checkpoint: bool = True
    stable_wait_s: float = 1.0

    def materialize(self) -> "DiffusionLongformCompareConfig":
        self.downloads_dir = Path(self.downloads_dir)
        self.output_root = Path(self.output_root)
        self.lab1_checkpoint = Path(self.lab1_checkpoint)
        if self.run_dir is not None:
            self.run_dir = Path(self.run_dir)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        return self


def make_style_shift_stable_config(**overrides: Any) -> DiffusionLongformCompareConfig:
    cfg = DiffusionLongformCompareConfig(
        output_root=REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_longform_best_panel",
        run_dir=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002",
        cache_dir=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache",
        n_songs=3,
        targets_per_song=2,
        source_seconds=45.0,
        chunk_seconds=3.0,
        overlap_seconds=0.5,
        t_start=280,
        t_start_end=200,
        reanchor_every=4,
        reanchor_t_start=170,
        ddim_steps=50,
        guidance_scale=2.10,
        style_strength=0.72,
        prefix_blend=1.0,
        source_prefix_blend=0.42,
        source_mel_blend=0.09,
        hf_source_blend=0.16,
        hf_start_bin=56,
        mel_time_smooth=3,
        mel_freq_smooth=0,
        assemble_domain="mel",
        device="auto",
        seed=328,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg.materialize()


def resolve_curated_best_diffusion_panel() -> List[Dict[str, Any]]:
    candidates = [
        {
            "label": "run_d002_best",
            "path": REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt",
            "note": "Best audited realism and style balance.",
        },
        {
            "label": "run_d002_epoch_006",
            "path": REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "epoch_006.pt",
            "note": "Documented subjective diffusion winner.",
        },
        {
            "label": "reset_best",
            "path": REPO_ROOT
            / "lab 3.1"
            / "outputs"
            / "overnight_runs"
            / "20260310_051425"
            / "diffusion_v2_reset_20260310_051425"
            / "checkpoints"
            / "best.pt",
            "note": "Fresh reset run best checkpoint.",
        },
        {
            "label": "reset_epoch_005",
            "path": REPO_ROOT
            / "lab 3.1"
            / "outputs"
            / "overnight_runs"
            / "20260310_051425"
            / "diffusion_v2_reset_20260310_051425"
            / "checkpoints"
            / "epoch_005.pt",
            "note": "Earlier checkpoint with stronger perceived style shift.",
        },
    ]
    return [row for row in candidates if Path(row["path"]).exists()]


def plan_longform_jobs(cfg: DiffusionLongformCompareConfig) -> List[Dict[str, Any]]:
    cfg = cfg.materialize()
    rng = random.Random(cfg.seed)
    audio_rows = ddb.discover_download_audio(cfg.downloads_dir)
    long_rows = [
        row
        for row in audio_rows
        if (row["duration_seconds"] or 0.0) >= float(cfg.source_seconds) + 5.0
        and row["size_bytes"] >= 10_000_000
    ]
    if len(long_rows) < max(1, cfg.n_songs):
        long_rows = sorted(audio_rows, key=lambda r: (r["duration_seconds"] or 0.0, r["size_bytes"]), reverse=True)
    if not long_rows:
        raise RuntimeError(f"No suitable long-form songs found in {cfg.downloads_dir}")

    selected = rng.sample(long_rows, k=min(int(cfg.n_songs), len(long_rows)))
    jobs: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected):
        path = Path(row["path"])
        duration = float(row["duration_seconds"] or 0.0)
        max_start = max(0.0, duration - float(cfg.source_seconds) - 0.1)
        start_sec = rng.uniform(0.0, max_start) if max_start > 0.0 else 0.0
        source_genre = ddb.infer_source_genre(path)
        targets = _pick_longform_targets(source_genre, max_targets=cfg.targets_per_song)
        for target_genre in targets:
            jobs.append(
                {
                    "job_idx": len(jobs),
                    "song_idx": idx,
                    "source_audio": path,
                    "source_genre": source_genre,
                    "target_genre": target_genre,
                    "start_sec": round(float(start_sec), 3),
                    "source_seconds": float(cfg.source_seconds),
                    "duration_seconds": duration if duration > 0 else None,
                    "size_bytes": int(row["size_bytes"]),
                }
            )
    return jobs


def resolve_checkpoint_panel(
    cfg: DiffusionLongformCompareConfig,
    *,
    include_best: bool = True,
    include_epoch5: bool = True,
) -> List[Dict[str, Any]]:
    ctx = ddb.resolve_inference_context(
        ddb.DiffusionDownloadsBatchConfig(
            run_dir=cfg.run_dir,
            cache_dir=cfg.cache_dir,
            snapshot_latest_checkpoint=cfg.snapshot_latest_checkpoint,
            stable_wait_s=cfg.stable_wait_s,
        )
    )
    ckpt_dir = Path(ctx["run_dir"]) / "checkpoints"
    panel: List[Dict[str, Any]] = []
    if include_best and (ckpt_dir / "best.pt").exists():
        panel.append({"label": "best", "path": ckpt_dir / "best.pt"})
    if include_epoch5 and (ckpt_dir / "epoch_005.pt").exists():
        panel.append({"label": "epoch_005", "path": ckpt_dir / "epoch_005.pt"})
    if not panel:
        panel.append({"label": ctx["checkpoint_path"].stem, "path": ctx["checkpoint_path"]})
    return panel


def _write_manifest(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    fieldnames = [
        "checkpoint_label",
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


def run_longform_compare(
    cfg: DiffusionLongformCompareConfig,
    checkpoint_panel: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = cfg.materialize()
    ctx = ddb.resolve_inference_context(
        ddb.DiffusionDownloadsBatchConfig(
            run_dir=cfg.run_dir,
            cache_dir=cfg.cache_dir,
            snapshot_latest_checkpoint=cfg.snapshot_latest_checkpoint,
            stable_wait_s=cfg.stable_wait_s,
        )
    )
    out_dir = cfg.output_root / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    jobs = plan_longform_jobs(cfg)
    jobs_path = out_dir / "jobs.json"
    jobs_path.write_text(json.dumps(jobs, indent=2, default=_json_default), encoding="utf-8")

    normalized_panel: List[Dict[str, Any]] = []
    for row in checkpoint_panel:
        label = str(row["label"])
        path = Path(row["path"])
        if cfg.snapshot_latest_checkpoint and path.name == "latest.pt":
            path = ddb.snapshot_checkpoint(path, out_dir / "checkpoint_snapshot" / label, stable_wait_s=cfg.stable_wait_s)
        normalized_panel.append({"label": label, "path": path})
    (out_dir / "checkpoint_panel.json").write_text(
        json.dumps(normalized_panel, indent=2, default=_json_default),
        encoding="utf-8",
    )

    cfg_dump = dict(asdict(cfg))
    cfg_dump["resolved_run_dir"] = str(ctx["run_dir"])
    cfg_dump["resolved_cache_dir"] = str(ctx["cache_dir"])
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2, default=_json_default), encoding="utf-8")

    manifest_rows: List[Dict[str, Any]] = []
    for ckpt in normalized_panel:
        label = str(ckpt["label"])
        checkpoint_path = Path(ckpt["path"])
        print(f"\n=== checkpoint: {label} ===")
        print(f"path: {checkpoint_path}")
        for job in jobs:
            source_audio = Path(job["source_audio"])
            job_tag = f"{int(job['job_idx']):02d}_{ddb._slug(source_audio.stem)[:40]}__to__{ddb._slug(str(job['target_genre']))}"
            job_out_dir = out_dir / "clips" / label / job_tag
            cmd = [
                "python",
                str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
                "--cache-dir",
                str(ctx["cache_dir"]),
                "--checkpoint",
                str(checkpoint_path),
                "--lab1-checkpoint",
                str(cfg.lab1_checkpoint),
                "--source-audio",
                str(source_audio),
                "--source-genre",
                str(job["source_genre"]),
                "--target-genre",
                str(job["target_genre"]),
                "--source-start-sec",
                str(job["start_sec"]),
                "--source-seconds",
                str(cfg.source_seconds),
                "--out-dir",
                str(job_out_dir),
                "--chunk-seconds",
                str(cfg.chunk_seconds),
                "--overlap-seconds",
                str(cfg.overlap_seconds),
                "--n-frames",
                str(cfg.n_frames),
                "--t-start",
                str(cfg.t_start),
                "--t-start-end",
                str(cfg.t_start_end),
                "--reanchor-every",
                str(cfg.reanchor_every),
                "--reanchor-t-start",
                str(cfg.reanchor_t_start),
                "--ddim-steps",
                str(cfg.ddim_steps),
                "--guidance-scale",
                str(cfg.guidance_scale),
                "--style-strength",
                str(cfg.style_strength),
                "--prefix-blend",
                str(cfg.prefix_blend),
                "--source-prefix-blend",
                str(cfg.source_prefix_blend),
                "--source-mel-blend",
                str(cfg.source_mel_blend),
                "--hf-source-blend",
                str(cfg.hf_source_blend),
                "--hf-start-bin",
                str(cfg.hf_start_bin),
                "--mel-time-smooth",
                str(cfg.mel_time_smooth),
                "--mel-freq-smooth",
                str(cfg.mel_freq_smooth),
                "--assemble-domain",
                str(cfg.assemble_domain),
                "--device",
                str(cfg.device),
                "--seed",
                str(cfg.seed + int(job["job_idx"])),
            ]
            log_path = logs_dir / f"{label}__{job_tag}.log"
            print(
                f"[{int(job['job_idx']) + 1:02d}/{len(jobs)}] "
                f"{source_audio.name} {float(job['start_sec']):.1f}s -> {job['target_genre']}"
            )
            _run_command(cmd, cwd=REPO_ROOT, log_path=log_path)
            manifest_rows.append(
                {
                    "checkpoint_label": label,
                    **job,
                    "output_dir": job_out_dir,
                    "source_wav": job_out_dir / "source.wav",
                    "generated_wav": job_out_dir / "longform_coherent.wav",
                    "metrics_json": job_out_dir / "coherence_metrics.json",
                    "checkpoint_path": checkpoint_path,
                }
            )

    manifest_path = _write_manifest(manifest_rows, out_dir / "manifest.csv")
    summary = {
        "tag": cfg.tag,
        "output_dir": out_dir,
        "run_dir": ctx["run_dir"],
        "cache_dir": ctx["cache_dir"],
        "checkpoint_panel_path": out_dir / "checkpoint_panel.json",
        "jobs_path": jobs_path,
        "manifest_path": manifest_path,
        "n_jobs": len(jobs),
        "n_checkpoints": len(normalized_panel),
        "total_runs": len(jobs) * len(normalized_panel),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary
