#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dggr.lab3_bridge import FrozenLab1Encoder
from dggr.lab3_diffusion_data import build_diffusion_cache
from dggr.real_music_discovery import discover_genre_manifest
from dggr.real_music_manifest import (
    DEFAULT_REAL_MUSIC_ROOT,
    build_real_music_manifest,
    parse_source_specs,
)
from dggr.real_music_transfer import (
    RealTransferTrainConfig,
    infer_real_transfer,
    train_real_transfer,
)


def _default_lab1_checkpoint() -> Path:
    candidates = [
        REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt",
        REPO_ROOT / "saves" / "lab1_run_combo_af_gate" / "latest.pt",
        REPO_ROOT / "saves" / "lab1_run_a" / "latest.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def _build_manifest(args: argparse.Namespace) -> Path:
    sources = parse_source_specs(args.source, default_root=Path(args.default_data_root))
    manifest_path = Path(args.manifest_path)
    df = build_real_music_manifest(
        sources,
        manifest_path,
        min_bytes=int(args.min_bytes),
        max_files_per_source=int(args.max_files_per_source),
        seed=int(args.seed),
        prefer_ogg_export=bool(args.prefer_ogg_export),
    )
    summary = {
        "manifest_path": str(manifest_path),
        "rows": int(len(df)),
        "genres": {str(k): int(v) for k, v in df["genre"].value_counts().sort_index().to_dict().items()},
        "sources": {str(k): int(v) for k, v in df["source"].value_counts().sort_index().to_dict().items()},
    }
    _write_json(manifest_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return manifest_path


def _discover_manifest(args: argparse.Namespace) -> Path:
    out_csv = Path(args.discover_manifest_path)
    report_path = Path(args.discovery_report_path) if str(args.discovery_report_path) else out_csv.with_suffix(".discovery_report.json")
    discover_genre_manifest(
        root=Path(args.default_data_root),
        out_csv=out_csv,
        report_path=report_path,
        n_clusters=int(args.discover_clusters),
        min_bytes=int(args.min_bytes),
        max_files=int(args.max_files_per_source),
        audio_feature_limit=int(args.discover_audio_feature_limit),
        audio_feature_seconds=float(args.discover_audio_feature_seconds),
        audio_workers=int(args.discover_audio_workers),
        seed=int(args.seed),
    )
    return out_csv


def _build_cache(args: argparse.Namespace, manifest_path: Path) -> Path:
    cache_dir = Path(args.cache_dir)
    if (cache_dir / "diff_meta.json").exists() and not bool(args.rebuild_cache):
        print(f"Using existing cache: {cache_dir}")
        return cache_dir
    encoder = FrozenLab1Encoder(Path(args.lab1_checkpoint), device=str(args.device))
    build_diffusion_cache(
        manifests_root=manifest_path.parent,
        manifest_files=[manifest_path.name],
        lab1_encoder=encoder,
        cache_dir=cache_dir,
        chunk_sec=float(args.chunk_seconds),
        max_chunks_per_track=int(args.max_chunks_per_track),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
        shard_size=int(args.shard_size),
    )
    return cache_dir


def _train(args: argparse.Namespace, cache_dir: Path) -> Dict[str, Any]:
    cfg = RealTransferTrainConfig(
        cache_dir=cache_dir,
        out_root=Path(args.train_out_root),
        resume_checkpoint=Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        resume_out_dir=Path(args.resume_out_dir) if args.resume_out_dir else None,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_batches_per_epoch=int(args.max_batches_per_epoch),
        val_batches=int(args.val_batches),
        max_frames=int(args.max_frames),
        base_ch=int(args.base_ch),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        judge_steps=int(args.judge_steps),
        judge_loss_weight=float(args.judge_loss_weight),
        donor_timbre_weight=float(args.donor_timbre_weight),
        checkpoint_every_batches=int(args.checkpoint_every_batches),
        seed=int(args.seed),
        device=str(args.device),
        epoch_sample_plan=Path(args.epoch_sample_plan),
        epoch_sample_count=int(args.epoch_sample_count),
        epoch_sample_every=int(args.epoch_sample_every),
        epoch_sample_seconds=float(args.epoch_sample_seconds),
        epoch_longform_seconds=float(args.epoch_longform_seconds),
        epoch_sample_chunk_seconds=float(args.epoch_sample_chunk_seconds),
        epoch_sample_overlap_seconds=float(args.epoch_sample_overlap_seconds),
        epoch_sample_style_strength=float(args.epoch_sample_style_strength),
        epoch_sample_envelope_strength=float(args.epoch_sample_envelope_strength),
    )
    summary = train_real_transfer(cfg)
    print(json.dumps(summary, indent=2))
    return summary


def _infer(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for infer")
    if not args.source_audio:
        raise ValueError("--source-audio is required for infer")
    if not args.target_genre:
        raise ValueError("--target-genre is required for infer")
    out_wav = Path(args.out_wav)
    if str(out_wav) == "":
        out_wav = Path(args.train_out_root) / "inference" / f"{Path(args.source_audio).stem}__to__{args.target_genre}.wav"
    summary = infer_real_transfer(
        checkpoint=Path(args.checkpoint),
        cache_dir=Path(args.cache_dir),
        source_audio=Path(args.source_audio),
        target_genre=str(args.target_genre),
        out_wav=out_wav,
        seconds=float(args.infer_seconds),
        chunk_seconds=float(args.infer_chunk_seconds),
        overlap_seconds=float(args.infer_overlap_seconds),
        style_strength=float(args.infer_style_strength),
        envelope_strength=float(args.infer_envelope_strength),
        device_arg=str(args.device),
    )
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a real-music manifest/cache, train the retrieval-fusion generator, or run inference."
    )
    ap.add_argument("--action", choices=["manifest", "discover", "cache", "train", "infer", "all"], default="all")
    ap.add_argument(
        "--source",
        action="append",
        default=None,
        help="Raw source as genre=PATH. Repeat for multiple genre folders. If omitted, uses the downloaded Spotify pop_0 folder.",
    )
    ap.add_argument("--default-data-root", type=Path, default=DEFAULT_REAL_MUSIC_ROOT)
    ap.add_argument("--manifest-path", type=Path, default=REPO_ROOT / "data" / "real_music_manifests" / "spotify_pop_0_manifest.csv")
    ap.add_argument("--min-bytes", type=int, default=64_000)
    ap.add_argument("--max-files-per-source", type=int, default=0, help="0 means all files. Use a small value for smoke tests.")
    ap.add_argument("--prefer-ogg-export", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--discover-manifest-path",
        type=Path,
        default=REPO_ROOT / "data" / "real_music_manifests" / "spotify_discovered_genres.csv",
    )
    ap.add_argument("--discovery-report-path", type=Path, default=Path(""))
    ap.add_argument("--discover-clusters", type=int, default=12)
    ap.add_argument("--discover-audio-feature-limit", type=int, default=2000)
    ap.add_argument("--discover-audio-feature-seconds", type=float, default=8.0)
    ap.add_argument("--discover-audio-workers", type=int, default=1)

    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "spotify_pop_0_cache")
    ap.add_argument("--lab1-checkpoint", type=Path, default=_default_lab1_checkpoint())
    ap.add_argument("--chunk-seconds", type=float, default=5.0)
    ap.add_argument("--max-chunks-per-track", type=int, default=4)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--rebuild-cache", action="store_true")

    ap.add_argument("--train-out-root", type=Path, default=REPO_ROOT / "saves2" / "real_music_transfer" / "runs")
    ap.add_argument("--resume-checkpoint", type=Path, default=None)
    ap.add_argument("--resume-out-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches-per-epoch", type=int, default=0, help="0 means full epoch.")
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=320)
    ap.add_argument("--base-ch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--judge-steps", type=int, default=400)
    ap.add_argument("--judge-loss-weight", type=float, default=0.25)
    ap.add_argument("--donor-timbre-weight", type=float, default=0.15)
    ap.add_argument("--checkpoint-every-batches", type=int, default=1000)
    ap.add_argument(
        "--epoch-sample-plan",
        type=Path,
        default=REPO_ROOT / "saves2" / "real_music_transfer" / "validation_plan.json",
        help="Validation-plan JSON used to render shortform and longform samples after each training epoch.",
    )
    ap.add_argument("--epoch-sample-count", type=int, default=2)
    ap.add_argument("--epoch-sample-every", type=int, default=1)
    ap.add_argument("--epoch-sample-seconds", type=float, default=12.0)
    ap.add_argument("--epoch-longform-seconds", type=float, default=60.0)
    ap.add_argument("--epoch-sample-chunk-seconds", type=float, default=3.0)
    ap.add_argument("--epoch-sample-overlap-seconds", type=float, default=0.5)
    ap.add_argument("--epoch-sample-style-strength", type=float, default=1.0)
    ap.add_argument("--epoch-sample-envelope-strength", type=float, default=0.75)

    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--source-audio", type=Path, default=None)
    ap.add_argument("--target-genre", type=str, default="")
    ap.add_argument("--out-wav", type=Path, default=Path(""))
    ap.add_argument("--infer-seconds", type=float, default=30.0)
    ap.add_argument("--infer-chunk-seconds", type=float, default=3.0)
    ap.add_argument("--infer-overlap-seconds", type=float, default=0.5)
    ap.add_argument("--infer-style-strength", type=float, default=1.0)
    ap.add_argument("--infer-envelope-strength", type=float, default=0.35)

    ap.add_argument("--seed", type=int, default=328)
    ap.add_argument("--device", type=str, default="auto")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    if args.action == "discover":
        _discover_manifest(args)
        return
    if args.action == "all":
        manifest_path = _discover_manifest(args)
    elif args.action == "manifest":
        manifest_path = _build_manifest(args)
    elif args.action == "cache":
        manifest_path = Path(args.manifest_path)
        if not manifest_path.exists():
            manifest_path = _build_manifest(args)
    if args.action == "manifest":
        return
    if args.action in {"cache", "all"}:
        cache_dir = _build_cache(args, manifest_path)
    else:
        cache_dir = Path(args.cache_dir)
    if args.action == "cache":
        return
    if args.action in {"train", "all"}:
        _train(args, cache_dir)
    elif args.action == "infer":
        _infer(args)


if __name__ == "__main__":
    main()
