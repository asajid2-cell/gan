from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOADS = Path.home() / "Downloads"
ALLOWED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
KNOWN_GENRES = ["baroque_classical", "hiphop_xtc", "lofi_hh_lfbb", "cc0_other"]


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _safe_duration_seconds(path: Path) -> float | None:
    try:
        import librosa  # local optional dependency already used in repo

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def _discover_download_audio(downloads_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not downloads_dir.exists():
        return rows
    for path in sorted(downloads_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_AUDIO_EXTS:
            continue
        stat = path.stat()
        if stat.st_size < 2_000_000:
            continue
        rows.append(
            {
                "path": path,
                "size_bytes": int(stat.st_size),
                "duration_seconds": _safe_duration_seconds(path),
                "extension": path.suffix.lower(),
            }
        )
    return rows


def _infer_source_genre(path: Path) -> str:
    name = path.name.lower()
    hiphop_tokens = [
        "drake",
        "juice wrld",
        "j. cole",
        "cole",
        "future",
        "lil ",
        "uzi",
        "boogie",
        "6ix9ine",
        "playboi",
        "a$ap",
        "butcher",
        "cochise",
        "doechii",
        "durk",
        "wizkid",
    ]
    classical_tokens = [
        "bach",
        "vivaldi",
        "mozart",
        "chopin",
        "beethoven",
        "baroque",
        "classical",
        "cello",
        "orchestra",
    ]
    lofi_tokens = [
        "lofi",
        "chill",
        "study",
        "beats",
        "ambient",
        "dream",
        "seaspray",
    ]
    if any(tok in name for tok in classical_tokens):
        return "baroque_classical"
    if any(tok in name for tok in lofi_tokens):
        return "lofi_hh_lfbb"
    if any(tok in name for tok in hiphop_tokens):
        return "hiphop_xtc"
    return "cc0_other"


def _pick_longform_targets(source_genre: str) -> List[str]:
    preferred = ["baroque_classical", "lofi_hh_lfbb", "hiphop_xtc"]
    targets = [g for g in preferred if g != source_genre]
    return targets[:2]


@dataclass
class OvernightConfig:
    tag: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    downloads_dir: Path = field(default_factory=lambda: DEFAULT_DOWNLOADS)
    work_root: Path = field(default_factory=lambda: REPO_ROOT / "lab 3.1" / "outputs" / "overnight_runs")
    codec_reuse_cache_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "saves2" / "lab3_codec_transfer" / "run1051" / "cache"
    )
    diffusion_cache_dir: Path = field(
        default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache"
    )
    lab1_checkpoint: Path = field(
        default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"
    )
    seed: int = 328
    n_monitor_clips: int = 6
    n_longform_clips: int = 2
    codec_stage1_epochs: int = 8
    codec_stage2_epochs: int = 10
    codec_stage3_epochs: int = 6
    codec_max_batches_per_epoch: int = 96
    diffusion_epochs: int = 18
    diffusion_epoch_samples: int = 6
    diffusion_ddim_steps: int = 50
    diffusion_guidance_scale: float = 2.0
    longform_seconds: float = 75.0
    longform_ddim_steps: int = 60
    longform_guidance_scale: float = 2.5
    longform_style_strength: float = 0.75
    longform_t_start: int = 320
    longform_t_start_end: int = 260
    longform_reanchor_every: int = 8
    codec_run_name: str | None = None
    diffusion_v2_run_name: str | None = None
    diffusion_v3_run_name: str | None = None
    run_audit: bool = True
    run_codec: bool = True
    run_codec_sweep: bool = True
    run_diffusion_v2: bool = True
    run_diffusion_v2_sweep: bool = True
    run_diffusion_v3: bool = True
    run_diffusion_v3_sweep: bool = True
    run_longform: bool = True
    diffusion_v3_epochs: int = 6
    diffusion_v3_lr: float = 5e-5
    diffusion_v3_disc_lr: float = 1e-4
    diffusion_v3_adv_weight: float = 0.05
    diffusion_v3_fm_weight: float = 1.0
    diffusion_v3_disc_warmup_steps: int = 1200

    def materialize(self) -> "OvernightConfig":
        if not self.codec_run_name:
            self.codec_run_name = f"codec_reset_{self.tag}"
        if not self.diffusion_v2_run_name:
            self.diffusion_v2_run_name = f"diffusion_v2_reset_{self.tag}"
        if not self.diffusion_v3_run_name:
            self.diffusion_v3_run_name = f"diffusion_v3_reset_{self.tag}"
        return self


def _work_dir(cfg: OvernightConfig) -> Path:
    return cfg.work_root / cfg.tag


def prepare_clip_plan(cfg: OvernightConfig) -> Dict[str, Any]:
    cfg = cfg.materialize()
    downloads_dir = Path(cfg.downloads_dir)
    audio_rows = _discover_download_audio(downloads_dir)
    if len(audio_rows) < 4:
        raise RuntimeError(f"Not enough audio files found in {downloads_dir}")

    rng = random.Random(cfg.seed)
    preferred_monitor = [
        row for row in audio_rows if row["extension"] in {".flac", ".wav"} and row["size_bytes"] >= 8_000_000
    ]
    if len(preferred_monitor) < cfg.n_monitor_clips:
        preferred_monitor = [row for row in audio_rows if row["size_bytes"] >= 4_000_000]
    monitor_rows = rng.sample(preferred_monitor, k=min(cfg.n_monitor_clips, len(preferred_monitor)))

    remaining = [row for row in audio_rows if row["path"] not in {r["path"] for r in monitor_rows}]
    long_candidates = [
        row for row in remaining if row["extension"] in {".flac", ".wav"} and row["size_bytes"] >= 15_000_000
    ]
    if len(long_candidates) < cfg.n_longform_clips:
        long_candidates = sorted(remaining, key=lambda r: r["size_bytes"], reverse=True)
    long_rows = long_candidates[: cfg.n_longform_clips]
    if len(long_rows) < 1:
        long_rows = monitor_rows[:1]

    codec_monitor = monitor_rows[0]["path"]
    clip_plan = {
        "seed": cfg.seed,
        "downloads_dir": str(downloads_dir),
        "codec_epoch_monitor_clip": str(codec_monitor),
        "monitor_clips": [
            {
                "path": str(row["path"]),
                "duration_seconds": row["duration_seconds"],
                "size_bytes": row["size_bytes"],
                "source_genre_guess": _infer_source_genre(row["path"]),
            }
            for row in monitor_rows
        ],
        "longform_jobs": [],
    }
    for row in long_rows:
        source_genre = _infer_source_genre(row["path"])
        targets = _pick_longform_targets(source_genre)
        for target_genre in targets:
            clip_plan["longform_jobs"].append(
                {
                    "source_audio": str(row["path"]),
                    "source_genre": source_genre,
                    "target_genre": target_genre,
                    "duration_seconds": row["duration_seconds"],
                    "size_bytes": row["size_bytes"],
                }
            )
    return clip_plan


def save_clip_plan(cfg: OvernightConfig, clip_plan: Dict[str, Any]) -> Path:
    out_dir = _work_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "clip_plan.json"
    path.write_text(json.dumps(clip_plan, indent=2, default=_json_default), encoding="utf-8")
    return path


def _run_command(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    with log_path.open("a", encoding="utf-8") as log:
        header = "\n$ " + " ".join(str(x) for x in cmd) + "\n\n"
        log.write(header)
        log.flush()
        start_pos = log.tell()

    with log_path.open("ab", buffering=0) as raw_log:
        process = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=raw_log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    last_pos = start_pos
    with log_path.open("r", encoding="utf-8", errors="replace") as reader:
        while True:
            reader.seek(last_pos)
            chunk = reader.read()
            if chunk:
                print(chunk, end="")
                last_pos = reader.tell()
            rc = process.poll()
            if rc is not None:
                reader.seek(last_pos)
                chunk = reader.read()
                if chunk:
                    print(chunk, end="")
                    last_pos = reader.tell()
                if rc != 0:
                    raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(str(x) for x in cmd)}")
                return
            time.sleep(1.0)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_codec_command(cfg: OvernightConfig, clip_plan: Dict[str, Any]) -> List[str]:
    run_dir = _work_dir(cfg) / "codec_run"
    return [
        sys.executable,
        str(REPO_ROOT / "lab 3" / "run_lab3_codec.py"),
        "--run-name",
        str(cfg.codec_run_name),
        "--force-custom-run-name",
        "--out-root",
        str(run_dir.parent),
        "--reuse-cache-dir",
        str(cfg.codec_reuse_cache_dir),
        "--style-cond-source",
        "mert_probe_embed",
        "--style-loss-mode",
        "mert_probe_ce",
        "--translator-direct-output",
        "--translator-direct-mix",
        "0.40",
        "--batch-size",
        "6",
        "--max-batches-per-epoch",
        str(cfg.codec_max_batches_per_epoch),
        "--stage1-epochs",
        str(cfg.codec_stage1_epochs),
        "--stage2-epochs",
        str(cfg.codec_stage2_epochs),
        "--stage3-epochs",
        str(cfg.codec_stage3_epochs),
        "--stage2-cond-mode",
        "exemplar",
        "--stage3-cond-mode",
        "exemplar",
        "--stage2-cond-alpha-start",
        "0.55",
        "--stage2-cond-alpha-end",
        "0.20",
        "--stage3-cond-alpha-start",
        "0.20",
        "--stage3-cond-alpha-end",
        "0.05",
        "--stage2-adv-weight",
        "0.50",
        "--stage3-adv-weight",
        "0.64",
        "--stage2-style-weight",
        "9.2",
        "--stage3-style-weight",
        "11.0",
        "--stage2-content-weight",
        "2.5",
        "--stage3-content-weight",
        "1.9",
        "--stage2-mrstft-weight",
        "0.12",
        "--stage3-mrstft-weight",
        "0.04",
        "--stage2-latent-l1-weight",
        "0.14",
        "--stage3-latent-l1-weight",
        "0.04",
        "--stage2-delta-budget",
        "0.18",
        "--stage3-delta-budget",
        "0.22",
        "--stage2-delta-budget-weight",
        "0.0",
        "--stage3-delta-budget-weight",
        "0.15",
        "--stage2-style-dropout-p",
        "0.08",
        "--stage3-style-dropout-p",
        "0.18",
        "--stage2-style-jitter-std",
        "0.04",
        "--stage3-style-jitter-std",
        "0.07",
        "--stage2-exemplar-noise-std",
        "0.03",
        "--stage3-exemplar-noise-std",
        "0.05",
        "--stage2-style-embed-align-weight",
        "0.30",
        "--stage3-style-embed-align-weight",
        "0.55",
        "--stage2-generated-mert-weight",
        "0.45",
        "--stage3-generated-mert-weight",
        "0.65",
        "--stage2-generated-mert-align-weight",
        "0.15",
        "--stage3-generated-mert-align-weight",
        "0.25",
        "--stage2-generated-mert-every",
        "4",
        "--stage3-generated-mert-every",
        "2",
        "--stage3-mode-seeking-weight",
        "0.05",
        "--stage3-mode-seeking-target",
        "0.03",
        "--sample-count",
        "20",
        "--sample-export-tag",
        "overnight_samples",
        "--epoch-sample-source-file",
        str(clip_plan["codec_epoch_monitor_clip"]),
        "--epoch-sample-every",
        "1",
        "--epoch-sample-tag",
        "epoch_samples",
    ]


def build_codec_sweep_command(cfg: OvernightConfig) -> List[str]:
    codec_run_dir = _work_dir(cfg) / cfg.codec_run_name
    return [
        sys.executable,
        str(REPO_ROOT / "lab 3" / "run_lab3_realism_sweep.py"),
        "codec",
        "--run-dir",
        str(codec_run_dir),
        "--n-samples",
        "16",
        "--write-audio-count",
        "4",
        "--max-fad-mert",
        "32",
        "--min-mps",
        "0.94",
        "--min-style-target-acc",
        "0.18",
        "--min-style-target-cos",
        "0.02",
    ]


def build_diffusion_command(cfg: OvernightConfig) -> List[str]:
    diffusion_run_dir = _work_dir(cfg) / cfg.diffusion_v2_run_name
    return [
        sys.executable,
        str(REPO_ROOT / "lab 3" / "run_lab3_diffusion_v2.py"),
        "--cache-dir",
        str(cfg.diffusion_cache_dir),
        "--out-dir",
        str(diffusion_run_dir),
        "--epochs",
        str(cfg.diffusion_epochs),
        "--epoch-samples",
        str(cfg.diffusion_epoch_samples),
        "--ddim-steps",
        str(cfg.diffusion_ddim_steps),
        "--guidance-scale",
        str(cfg.diffusion_guidance_scale),
    ]


def build_diffusion_sweep_command(cfg: OvernightConfig, *, phase: str = "v2") -> List[str]:
    if phase == "v2":
        diffusion_run_dir = _work_dir(cfg) / cfg.diffusion_v2_run_name
    elif phase == "v3":
        diffusion_run_dir = _work_dir(cfg) / cfg.diffusion_v3_run_name
    else:
        raise ValueError(f"Unsupported diffusion phase: {phase}")
    return [
        sys.executable,
        str(REPO_ROOT / "lab 3" / "run_lab3_realism_sweep.py"),
        "diffusion",
        "--run-dir",
        str(diffusion_run_dir),
        "--include-all-epochs",
        "--n-samples",
        "12",
        "--write-audio-count",
        "4",
        "--max-fad-mert",
        "35",
        "--min-mps",
        "0.90",
        "--min-style-target-acc",
        "0.30",
        "--min-style-target-cos",
        "0.05",
        "--ddim-steps",
        str(cfg.diffusion_ddim_steps),
        "--guidance-scale",
        str(cfg.diffusion_guidance_scale),
    ]


def _select_best_v2_checkpoint(cfg: OvernightConfig) -> Path:
    best_json = _work_dir(cfg) / cfg.diffusion_v2_run_name / "realism_supervisor" / "diffusion_realism_best.json"
    if not best_json.exists():
        raise RuntimeError(f"Missing diffusion realism summary: {best_json}")
    best = (_read_json(best_json).get("best") or {})
    ckpt_name = best.get("checkpoint") or "best.pt"
    return _work_dir(cfg) / cfg.diffusion_v2_run_name / "checkpoints" / ckpt_name


def build_diffusion_v3_command(cfg: OvernightConfig, *, v2_checkpoint: Path) -> List[str]:
    diffusion_v3_run_dir = _work_dir(cfg) / cfg.diffusion_v3_run_name
    return [
        sys.executable,
        str(REPO_ROOT / "lab 3" / "run_lab3_diffusion_v3.py"),
        "--cache-dir",
        str(cfg.diffusion_cache_dir),
        "--out-dir",
        str(diffusion_v3_run_dir),
        "--v2-checkpoint",
        str(v2_checkpoint),
        "--restart",
        "--epochs",
        str(cfg.diffusion_v3_epochs),
        "--lr",
        str(cfg.diffusion_v3_lr),
        "--disc-lr",
        str(cfg.diffusion_v3_disc_lr),
        "--batch-size",
        "4",
        "--grad-accum",
        "4",
        "--ema-decay",
        "0.999",
        "--cfg-dropout-p",
        "0.12",
        "--disc-warmup-steps",
        str(cfg.diffusion_v3_disc_warmup_steps),
        "--adv-weight",
        str(cfg.diffusion_v3_adv_weight),
        "--fm-weight",
        str(cfg.diffusion_v3_fm_weight),
        "--epoch-samples",
        "6",
        "--ddim-steps",
        str(cfg.diffusion_ddim_steps),
        "--guidance-scale",
        str(cfg.diffusion_guidance_scale),
    ]


def _best_from_realism_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Missing realism summary: {path}")
    return _read_json(path).get("best") or {}


def _select_best_diffusion_candidate(cfg: OvernightConfig) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    v2_json = _work_dir(cfg) / cfg.diffusion_v2_run_name / "realism_supervisor" / "diffusion_realism_best.json"
    if v2_json.exists():
        best = _best_from_realism_json(v2_json)
        best["phase"] = "v2"
        best["run_dir"] = str(_work_dir(cfg) / cfg.diffusion_v2_run_name)
        candidates.append(best)
    v3_json = _work_dir(cfg) / cfg.diffusion_v3_run_name / "realism_supervisor" / "diffusion_realism_best.json"
    if v3_json.exists():
        best = _best_from_realism_json(v3_json)
        best["phase"] = "v3"
        best["run_dir"] = str(_work_dir(cfg) / cfg.diffusion_v3_run_name)
        candidates.append(best)
    if not candidates:
        raise RuntimeError("No diffusion realism summaries found to select from.")
    candidates = sorted(
        candidates,
        key=lambda row: (
            float(row.get("rank", 10**9)),
            float(row.get("fad_mert", 10**9)),
            -float(row.get("style_target_acc", -10**9)),
            -float(row.get("style_target_cos", -10**9)),
        ),
    )
    return candidates[0]


def build_longform_commands(cfg: OvernightConfig, clip_plan: Dict[str, Any], checkpoint: Path) -> List[List[str]]:
    jobs: List[List[str]] = []
    for job in clip_plan["longform_jobs"][:2]:
        out_dir = _work_dir(cfg) / "longform" / (
            _slug(Path(job["source_audio"]).stem) + "__to__" + _slug(job["target_genre"])
        )
        jobs.append(
            [
                sys.executable,
                str(REPO_ROOT / "lab 4" / "run_lab4_longform_coherence.py"),
                "--cache-dir",
                str(cfg.diffusion_cache_dir),
                "--checkpoint",
                str(checkpoint),
                "--lab1-checkpoint",
                str(cfg.lab1_checkpoint),
                "--source-audio",
                str(job["source_audio"]),
                "--source-genre",
                str(job["source_genre"]),
                "--target-genre",
                str(job["target_genre"]),
                "--source-seconds",
                str(cfg.longform_seconds),
                "--out-dir",
                str(out_dir),
                "--t-start",
                str(cfg.longform_t_start),
                "--t-start-end",
                str(cfg.longform_t_start_end),
                "--reanchor-every",
                str(cfg.longform_reanchor_every),
                "--reanchor-t-start",
                "220",
                "--ddim-steps",
                str(cfg.longform_ddim_steps),
                "--guidance-scale",
                str(cfg.longform_guidance_scale),
                "--style-strength",
                str(cfg.longform_style_strength),
                "--prefix-blend",
                "1.0",
                "--source-prefix-blend",
                "0.25",
                "--source-mel-blend",
                "0.20",
                "--hf-source-blend",
                "0.30",
                "--mel-time-smooth",
                "3",
                "--mel-freq-smooth",
                "3",
                "--assemble-domain",
                "mel",
            ]
        )
    return jobs


def _write_summary(cfg: OvernightConfig, clip_plan: Dict[str, Any], summary: Dict[str, Any]) -> Path:
    out_dir = _work_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "morning_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")

    md = out_dir / "morning_summary.md"
    lines = [
        f"# Overnight Summary ({cfg.tag})",
        "",
        "## Clip plan",
        f"- Codec epoch monitor: `{clip_plan['codec_epoch_monitor_clip']}`",
        "",
        "## Key artifacts",
    ]
    for key, value in summary.get("artifacts", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Morning listening order",
            "1. Codec epoch samples from the monitor clip.",
            "2. Codec realism sweep written examples.",
            "3. Diffusion V2 realism sweep written examples.",
            "4. Diffusion V3 realism sweep written examples.",
            "5. Long-form diffusion outputs from the selected best diffusion candidate.",
        ]
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_overnight_suite(cfg: OvernightConfig) -> Dict[str, Any]:
    cfg = cfg.materialize()
    work_dir = _work_dir(cfg)
    logs_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    clip_plan = prepare_clip_plan(cfg)
    clip_plan_path = save_clip_plan(cfg, clip_plan)

    summary: Dict[str, Any] = {
        "config": asdict(cfg),
        "clip_plan_path": clip_plan_path,
        "artifacts": {},
        "started_at": datetime.now(),
    }

    if cfg.run_audit:
        audit_log = logs_dir / "pipeline_audit.log"
        _run_command(
            [sys.executable, str(REPO_ROOT / "lab 3.1" / "scripts" / "pipeline_audit.py")],
            cwd=REPO_ROOT,
            log_path=audit_log,
        )
        summary["artifacts"]["audit_dir"] = REPO_ROOT / "lab 3.1" / "outputs" / "audit"

    codec_run_dir = work_dir / cfg.codec_run_name
    if cfg.run_codec:
        _run_command(build_codec_command(cfg, clip_plan), cwd=REPO_ROOT, log_path=logs_dir / "codec_train.log")
        summary["artifacts"]["codec_run_dir"] = codec_run_dir

    if cfg.run_codec_sweep:
        _run_command(build_codec_sweep_command(cfg), cwd=REPO_ROOT, log_path=logs_dir / "codec_sweep.log")
        summary["artifacts"]["codec_realism_best"] = codec_run_dir / "realism_supervisor" / "codec_realism_best.json"

    diffusion_v2_run_dir = work_dir / cfg.diffusion_v2_run_name
    if cfg.run_diffusion_v2:
        _run_command(build_diffusion_command(cfg), cwd=REPO_ROOT, log_path=logs_dir / "diffusion_train.log")
        summary["artifacts"]["diffusion_v2_run_dir"] = diffusion_v2_run_dir

    if cfg.run_diffusion_v2_sweep:
        _run_command(build_diffusion_sweep_command(cfg, phase="v2"), cwd=REPO_ROOT, log_path=logs_dir / "diffusion_v2_sweep.log")
        summary["artifacts"]["diffusion_v2_realism_best"] = (
            diffusion_v2_run_dir / "realism_supervisor" / "diffusion_realism_best.json"
        )

    if cfg.run_diffusion_v3:
        best_v2_ckpt = _select_best_v2_checkpoint(cfg)
        summary["artifacts"]["selected_v2_checkpoint_for_v3"] = best_v2_ckpt
        _run_command(
            build_diffusion_v3_command(cfg, v2_checkpoint=best_v2_ckpt),
            cwd=REPO_ROOT,
            log_path=logs_dir / "diffusion_v3_train.log",
        )
        summary["artifacts"]["diffusion_v3_run_dir"] = work_dir / cfg.diffusion_v3_run_name

    if cfg.run_diffusion_v3_sweep:
        _run_command(build_diffusion_sweep_command(cfg, phase="v3"), cwd=REPO_ROOT, log_path=logs_dir / "diffusion_v3_sweep.log")
        summary["artifacts"]["diffusion_v3_realism_best"] = (
            work_dir / cfg.diffusion_v3_run_name / "realism_supervisor" / "diffusion_realism_best.json"
        )

    if cfg.run_longform:
        best_candidate = _select_best_diffusion_candidate(cfg)
        best_ckpt = Path(best_candidate["run_dir"]) / "checkpoints" / str(best_candidate["checkpoint"])
        summary["artifacts"]["selected_diffusion_candidate"] = best_candidate
        summary["artifacts"]["selected_diffusion_checkpoint"] = best_ckpt
        for idx, cmd in enumerate(build_longform_commands(cfg, clip_plan, best_ckpt), start=1):
            _run_command(cmd, cwd=REPO_ROOT, log_path=logs_dir / f"longform_{idx:02d}.log")
        summary["artifacts"]["longform_dir"] = work_dir / "longform"

    summary["finished_at"] = datetime.now()
    summary_path = _write_summary(cfg, clip_plan, summary)
    summary["summary_path"] = summary_path
    return summary


def print_clip_plan(clip_plan: Dict[str, Any]) -> None:
    print("Codec epoch monitor clip:")
    print(" ", clip_plan["codec_epoch_monitor_clip"])
    print("\nMonitor clips:")
    for clip in clip_plan["monitor_clips"]:
        print(
            f" - {clip['path']} | genre~{clip['source_genre_guess']} | "
            f"dur={clip['duration_seconds']} | size={clip['size_bytes']}"
        )
    print("\nLong-form jobs:")
    for job in clip_plan["longform_jobs"][:2]:
        print(
            f" - {job['source_audio']} | src={job['source_genre']} -> tgt={job['target_genre']} "
            f"| dur={job['duration_seconds']}"
        )
