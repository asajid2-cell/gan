from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB31_SCRIPTS = REPO_ROOT / "lab 3.1" / "scripts"
if str(LAB31_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB31_SCRIPTS))

import diffusion_downloads_batch as ddb


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
    raise TypeError(f"Unsupported json value: {type(value)!r}")


def _write_status(path: Path, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["ts"] = datetime.now().isoformat()
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)


def _latest_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    for name in ["best.pt", "latest.pt"]:
        path = ckpt_dir / name
        if path.exists():
            return path
    pts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def _to_stereo_44k(audio: torch.Tensor, sr: int) -> torch.Tensor:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.size(0) == 1:
        audio = audio.repeat(2, 1)
    elif audio.size(0) > 2:
        audio = audio[:2]
    if sr != 44100:
        audio = torchaudio.functional.resample(audio, sr, 44100)
    return audio


def _to_mono_sr(audio: torch.Tensor, sr_in: int, sr_out: int) -> np.ndarray:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr_in != sr_out:
        audio = torchaudio.functional.resample(audio, sr_in, sr_out)
    return audio.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class RoundConfig:
    output_root: Path = field(default_factory=lambda: Path.home() / "Desktop" / "dggr_new_model_rounds")
    downloads_dir: Path = field(default_factory=lambda: Path.home() / "Downloads")
    cache_dir: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    bootstrap_checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt")
    lab1_checkpoint: Path = field(default_factory=lambda: REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt")
    max_source_songs: int = 14
    source_seconds: float = 75.0
    epochs: int = 2
    max_batches_per_epoch: int = 180
    max_frames: int = 320
    seed: int = 328


def _discover_training_sources(cfg: RoundConfig) -> List[Dict[str, Any]]:
    rows = ddb.discover_download_audio(cfg.downloads_dir)
    min_seconds = max(float(cfg.source_seconds) + 4.0, 24.0)
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        duration = float(row.get("duration_seconds") or 0.0)
        if duration < min_seconds:
            continue
        path = Path(str(row["path"]))
        if any(ord(ch) >= 128 for ch in str(path)):
            continue
        filtered.append(row)
    filtered.sort(key=lambda r: (-(float(r.get("duration_seconds") or 0.0)), -int(r.get("size_bytes") or 0), str(r["path"])))
    return filtered[: max(1, int(cfg.max_source_songs))]


def _build_accompaniment_cache(cfg: RoundConfig, round_dir: Path, status_path: Path, run_log: Path) -> Path:
    cache_root = round_dir / "accompaniment_cache"
    audio_root = cache_root / "audio"
    manifest_path = cache_root / "manifest.csv"
    if manifest_path.exists() and audio_root.exists():
        _append_log(run_log, f"Reusing accompaniment cache at {cache_root}")
        _write_status(status_path, {"phase": "cache_ready", "cache_root": cache_root, "manifest": manifest_path})
        return audio_root

    audio_root.mkdir(parents=True, exist_ok=True)
    rows = _discover_training_sources(cfg)
    if not rows:
        raise RuntimeError(f"No usable downloads sources found in {cfg.downloads_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model().to(device).eval()
    sources = list(model.sources)
    vocal_idx = sources.index("vocals")

    csv_rows: List[Dict[str, Any]] = []
    _append_log(run_log, f"Building accompaniment cache from {len(rows)} Downloads songs on {device}")
    for idx, row in enumerate(rows):
        src_path = Path(str(row["path"]))
        stem_slug = f"{idx:03d}_{_slug(src_path.stem)}"
        out_audio = audio_root / f"{stem_slug}__accompaniment.wav"
        if out_audio.exists():
            csv_rows.append(
                {
                    "idx": idx,
                    "source_path": str(src_path),
                    "audio_path": str(out_audio),
                    "duration_seconds": float(row.get("duration_seconds") or 0.0),
                    "source_genre": str(ddb.infer_source_genre(src_path)),
                }
            )
            continue

        _write_status(
            status_path,
            {
                "phase": "building_cache",
                "current_index": idx,
                "total_sources": len(rows),
                "source_path": src_path,
                "cache_root": cache_root,
            },
        )
        audio, sr = torchaudio.load(str(src_path))
        audio = _to_stereo_44k(audio, sr)
        max_len = int(round(float(cfg.source_seconds) * 44100))
        audio = audio[:, :max_len]
        with torch.no_grad():
            est = model(audio.unsqueeze(0).to(device)).cpu()[0]
        vocals = est[vocal_idx]
        accomp = est.sum(dim=0) - vocals
        accomp_mono = _to_mono_sr(accomp, 44100, 22050)
        sf.write(str(out_audio), accomp_mono, 22050)
        csv_rows.append(
            {
                "idx": idx,
                "source_path": str(src_path),
                "audio_path": str(out_audio),
                "duration_seconds": float(row.get("duration_seconds") or 0.0),
                "source_genre": str(ddb.infer_source_genre(src_path)),
            }
        )
        _append_log(run_log, f"built accompaniment stem {idx + 1}/{len(rows)} -> {out_audio.name}")

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "source_path", "audio_path", "duration_seconds", "source_genre"])
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_status(status_path, {"phase": "cache_ready", "cache_root": cache_root, "manifest": manifest_path, "n_sources": len(csv_rows)})
    return audio_root


def _launch_training(cfg: RoundConfig, round_dir: Path, source_audio_root: Path, status_path: Path, run_log: Path) -> Path:
    run_dir = round_dir / "train_run"
    logs_dir = round_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / "train.out.log"
    stderr_log = logs_dir / "train.err.log"

    cmd = [
        "python",
        str(REPO_ROOT / "lab 3.1" / "scripts" / "diffusion_longform_retool_train.py"),
        "--cache-dir",
        str(cfg.cache_dir),
        "--out-dir",
        str(run_dir),
        "--bootstrap-checkpoint",
        str(cfg.bootstrap_checkpoint),
        "--epochs",
        str(int(cfg.epochs)),
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--max-frames",
        str(int(cfg.max_frames)),
        "--lr",
        "3e-5",
        "--warmup-steps",
        "40",
        "--max-batches-per-epoch",
        str(int(cfg.max_batches_per_epoch)),
        "--identity-weight",
        "1.0",
        "--style-weight",
        "2.55",
        "--anchor-weight",
        "0.24",
        "--envelope-weight",
        "0.15",
        "--continuity-weight",
        "0.38",
        "--hf-penalty-weight",
        "0.08",
        "--vocal-weight",
        "0.00",
        "--crackle-weight",
        "0.10",
        "--anchor-bins",
        "30",
        "--hf-start-bin",
        "58",
        "--style-probe-frames",
        "96",
        "--style-every-steps",
        "6",
        "--style-batch-splits",
        "4",
        "--monitor-steps",
        "15",
        "--save-every-steps",
        "30",
        "--source-mode",
        "mixed",
        "--downloads-dir",
        str(source_audio_root),
        "--downloads-source-samples-per-epoch",
        "700",
        "--mixed-source-samples-per-epoch",
        "700",
        "--downloads-mix-ratio",
        "0.35",
        "--source-aug-prob",
        "0.35",
        "--source-noise-std",
        "0.008",
        "--source-cond-noise-std",
        "0.006",
        "--source-global-offset-std",
        "0.02",
        "--source-hf-tilt-std",
        "0.03",
        "--source-time-mask-prob",
        "0.15",
        "--source-time-mask-frames",
        "18",
        "--epoch-train-samples",
        "2",
        "--epoch-download-samples",
        "4",
        "--epoch-sample-ddim-steps",
        "50",
        "--epoch-sample-t-start",
        "310",
        "--epoch-sample-guidance-scale",
        "2.25",
        "--epoch-sample-style-strength",
        "0.96",
        "--lab1-checkpoint",
        str(cfg.lab1_checkpoint),
        "--seed",
        str(int(cfg.seed)),
    ]

    _append_log(run_log, "Launching new accompaniment-first stable-style training run")
    _write_status(status_path, {"phase": "training", "run_dir": run_dir, "stdout_log": stdout_log, "stderr_log": stderr_log, "command": cmd})
    with stdout_log.open("w", encoding="utf-8", errors="replace") as out_f, stderr_log.open("w", encoding="utf-8", errors="replace") as err_f:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=out_f, stderr=err_f, text=True)
    monitor_log = logs_dir / "monitor.jsonl"
    mon_cmd = [
        "python",
        str(REPO_ROOT / "lab 3.3" / "scripts" / "monitor_training_run.py"),
        "--run-dir",
        str(run_dir),
        "--stdout-log",
        str(stdout_log),
        "--stderr-log",
        str(stderr_log),
        "--pid",
        str(int(proc.pid)),
        "--interval-sec",
        "300",
        "--monitor-log",
        str(monitor_log),
    ]
    subprocess.Popen(mon_cmd, cwd=str(REPO_ROOT))

    while True:
        code = proc.poll()
        history_path = run_dir / "v2_history.json"
        epochs_done = 0
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                epochs_done = len(history) if isinstance(history, list) else 0
            except Exception:
                epochs_done = 0
        sample_count = sum(1 for _ in (run_dir / "epoch_samples").rglob("*.wav")) if (run_dir / "epoch_samples").exists() else 0
        _write_status(
            status_path,
            {
                "phase": "training",
                "run_dir": run_dir,
                "trainer_pid": int(proc.pid),
                "epochs_done": int(epochs_done),
                "sample_wavs": int(sample_count),
                "stdout_log": stdout_log,
                "stderr_log": stderr_log,
                "monitor_log": monitor_log,
            },
        )
        if code is not None:
            if code != 0:
                raise RuntimeError(f"Training failed with exit code {code}. See {stderr_log}")
            break
        time.sleep(60)

    _append_log(run_log, f"Training completed successfully in {run_dir}")
    _write_status(status_path, {"phase": "training_complete", "run_dir": run_dir, "best_checkpoint": _latest_checkpoint(run_dir)})
    return run_dir


def _run_compare_pack(cfg: RoundConfig, round_dir: Path, checkpoint: Path, status_path: Path, run_log: Path) -> Path:
    compare_dir = round_dir / "compare_pack"
    compare_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(REPO_ROOT / "lab 3.3" / "scripts" / "run_hybrid_vocal_auto_best.py"),
        "--out-dir",
        str(compare_dir),
        "--checkpoint",
        str(checkpoint),
        "--use-picked-songs",
    ]
    compare_log = round_dir / "logs" / "compare.log"
    _append_log(run_log, f"Launching fixed 12-output compare pack with checkpoint {checkpoint.name}")
    _write_status(status_path, {"phase": "compare_pack", "compare_dir": compare_dir, "checkpoint": checkpoint, "command": cmd})
    with compare_log.open("w", encoding="utf-8", errors="replace") as log_f:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Compare pack generation failed with exit code {code}. See {compare_log}")
    _append_log(run_log, f"Compare pack completed in {compare_dir}")
    _write_status(status_path, {"phase": "complete", "compare_dir": compare_dir, "checkpoint": checkpoint})
    return compare_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an accompaniment cache, train a new stable-style diffusion checkpoint, and emit a fixed compare pack.")
    parser.add_argument("--out-root", type=Path, default=Path.home() / "Desktop" / "dggr_new_model_rounds")
    parser.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--max-source-songs", type=int, default=14)
    parser.add_argument("--source-seconds", type=float, default=75.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-batches-per-epoch", type=int, default=180)
    parser.add_argument("--max-frames", type=int, default=320)
    parser.add_argument("--seed", type=int, default=328)
    args = parser.parse_args()

    cfg = RoundConfig(
        output_root=Path(args.out_root),
        downloads_dir=Path(args.downloads_dir),
        max_source_songs=int(args.max_source_songs),
        source_seconds=float(args.source_seconds),
        epochs=int(args.epochs),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        max_frames=int(args.max_frames),
        seed=int(args.seed),
    )
    round_dir = cfg.output_root / f"round_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    round_dir.mkdir(parents=True, exist_ok=True)
    status_path = round_dir / "status.json"
    run_log = round_dir / "round.log"
    _write_status(status_path, {"phase": "init", "round_dir": round_dir, "config": asdict(cfg)})
    (round_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=_json_default), encoding="utf-8")

    source_audio_root = _build_accompaniment_cache(cfg, round_dir, status_path, run_log)
    train_run_dir = _launch_training(cfg, round_dir, source_audio_root, status_path, run_log)
    best_checkpoint = _latest_checkpoint(train_run_dir)
    compare_dir = _run_compare_pack(cfg, round_dir, best_checkpoint, status_path, run_log)
    summary = {
        "round_dir": str(round_dir),
        "source_audio_root": str(source_audio_root),
        "train_run_dir": str(train_run_dir),
        "best_checkpoint": str(best_checkpoint),
        "compare_dir": str(compare_dir),
        "status_path": str(status_path),
        "run_log": str(run_log),
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
