from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import EncodecModel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LAB33_SCRIPTS = REPO_ROOT / "lab 3.3" / "scripts"
if str(LAB33_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LAB33_SCRIPTS))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dggr.lab3_bridge import load_audio_chunk
from dggr.lab3_codec_bridge import FrozenEncodec
from dggr.lab3_diffusion_data import DIFFUSION_SR, load_diffusion_cache
from dggr.lab3_diffusion_train import load_bigvgan_robust
from optimize_retrieval_pretrained_blend import _candidate_backings, _local_score
from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _json_default, _resolve_stems, picked_songs
from train_pretrained_encodec_fusion import EncodecLatentFusionNet, _build_inference_donor_rows, generate_longform_encodec_from_rows
from train_scratch_retrieval_fusion import (
    RetrievalFusionUNet,
    _build_track_bank,
    _choose_donor_track,
    _device_from_arg,
    _judge_probs_for_audio,
    _load_or_train_judge,
    _mix_preserved_vocals,
    _slug,
    generate_longform,
)
from train_scratch_structure_diffusion import _audio_metrics


def _songs_from_downloads(root: Path, limit: int, song_filter: str = "") -> List[Dict[str, Any]]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if song_filter.strip():
        pat = song_filter.strip().lower()
        files = [p for p in files if pat in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No audio files found in {root}")
    return [{"path": p, "source_genre": "cc0_other"} for p in files[:limit]]


def _load_retrieval_models(suite_dir: Path, device: torch.device) -> Tuple[Dict[str, RetrievalFusionUNet], Dict[str, Dict[str, Any]]]:
    winner_map = json.loads((suite_dir / "winner_map.json").read_text(encoding="utf-8"))
    models: Dict[str, RetrievalFusionUNet] = {}
    payloads: Dict[str, Dict[str, Any]] = {}
    for target, ckpt in winner_map.items():
        payload = torch.load(str(ckpt), map_location=device, weights_only=False)
        model = RetrievalFusionUNet(in_ch=16, num_genres=len(payload["genre_to_idx"]), base_ch=int(payload["cfg"]["base_ch"])).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        models[str(target)] = model
        payloads[str(target)] = payload
    return models, payloads


def _load_pretrained_models(suite_dir: Path, device: torch.device) -> Dict[str, Dict[str, Any]]:
    winner_map = json.loads((suite_dir / "winner_map.json").read_text(encoding="utf-8"))
    loaded: Dict[str, Dict[str, Any]] = {}
    for target, ckpt in winner_map.items():
        payload = torch.load(str(ckpt), map_location=device, weights_only=False)
        cfg = payload["cfg"]
        model = EncodecLatentFusionNet(
            latent_ch=int(payload["codec_cfg"]["latent_channels"]),
            cond_ch=14,
            num_genres=len(payload["genre_to_idx"]),
            base_ch=int(cfg["base_ch"]),
            proposal_scale=float(cfg.get("proposal_scale", 0.65)),
            source_skip_mix=float(cfg.get("source_skip_mix", 1.0)),
        ).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        codec = FrozenEncodec(
            model_id=str(cfg["model_id"]),
            bandwidth=float(cfg["bandwidth"]),
            chunk_seconds=float(cfg["chunk_seconds"]),
            device=str(device),
        )
        codec_model = EncodecModel.from_pretrained(str(cfg["model_id"]), local_files_only=True).to(device)
        codec_model.decoder.load_state_dict(payload["codec_decoder"])
        codec_model.eval()
        loaded[str(target)] = {"model": model, "payload": payload, "codec": codec, "codec_model": codec_model}
    return loaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the retrieval + pretrained blend production path.")
    ap.add_argument("--retrieval-suite-dir", type=Path, required=True)
    ap.add_argument("--pretrained-suite-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "Desktop" / "dggr_retrieval_pretrained_blend_production")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--judge-ckpt", type=Path, default=Path.home() / "Desktop" / "dggr_per_genre_structure_suite" / "suite_20260331_205731" / "judge_compare" / "genre_judge.pt")
    ap.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--song-filter", type=str, default="")
    ap.add_argument("--use-picked-songs", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = _device_from_arg(str(args.device))
    retrieval_suite_dir = Path(args.retrieval_suite_dir)
    pretrained_suite_dir = Path(args.pretrained_suite_dir)
    out_dir = Path(args.out_dir) / f"retrieval_pretrained_blend_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(args.cache_dir), mmap=True)
    all_indices = np.arange(len(index_df), dtype=np.int64)
    track_bank = _build_track_bank(index_df, arrays, all_indices)
    retrieval_models, retrieval_payloads = _load_retrieval_models(retrieval_suite_dir, device)
    pretrained_models = _load_pretrained_models(pretrained_suite_dir, device)
    vocoder = load_bigvgan_robust(device=device)
    judge, _ = _load_or_train_judge(Path(args.judge_ckpt), Path(args.cache_dir), out_dir, device, 256)
    hybrid_cfg = HybridPushConfig()

    songs = picked_songs() if bool(args.use_picked_songs) else _songs_from_downloads(Path(args.downloads_dir), limit=max(1, int(args.limit)), song_filter=args.song_filter)
    _write = lambda p, obj: p.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    _write(out_dir / "config.json", {"retrieval_suite_dir": str(retrieval_suite_dir), "pretrained_suite_dir": str(pretrained_suite_dir)})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_idx", "target", "source_song", "winner_label", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    job_idx = 0
    rows: List[Dict[str, Any]] = []
    target_label_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for song in songs:
        stems = _resolve_stems(hybrid_cfg, song)
        source_key = _slug(Path(song["path"]).stem)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=60.0, start_sec=0.0)
        for target in TARGET_GENRES:
            donor_track = _choose_donor_track(source_acc, track_bank, int(genre_to_idx[target]))

            retrieval_model = retrieval_models[str(target)]
            retrieval_payload = retrieval_payloads[str(target)]
            retrieval = generate_longform(
                retrieval_model,
                source_audio=source_acc,
                target_genre_idx=int(genre_to_idx[target]),
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(meta.mel_min),
                mel_max=float(meta.mel_max),
                max_frames=int(retrieval_payload["cfg"].get("max_frames", 256)),
                chunk_seconds=3.0,
                overlap_seconds=0.5,
                vocoder=vocoder,
                device=device,
            ).astype(np.float32)

            pretrained_bundle = pretrained_models[str(target)]
            donor_rows = _build_inference_donor_rows(index_df, donor_track)
            pretrained = generate_longform_encodec_from_rows(
                pretrained_bundle["model"],
                codec_model=pretrained_bundle["codec_model"],
                source_audio_22k=source_acc,
                target_genre_idx=int(genre_to_idx[target]),
                donor_rows=donor_rows,
                codec=pretrained_bundle["codec"],
                cond_frames=int(round(float(pretrained_bundle["payload"]["cfg"]["chunk_seconds"]) * 75.0)),
                device=device,
                source_hint_low_keep=float(pretrained_bundle["payload"]["cfg"].get("source_hint_low_keep", 0.75)),
                source_hint_mid_keep=float(pretrained_bundle["payload"]["cfg"].get("source_hint_mid_keep", 0.45)),
                source_hint_high_keep=float(pretrained_bundle["payload"]["cfg"].get("source_hint_high_keep", 0.20)),
            ).astype(np.float32)

            n = min(len(source_acc), len(retrieval), len(pretrained))
            retrieval = retrieval[:n]
            pretrained = pretrained[:n]
            source_for_metrics = source_acc[:n].astype(np.float32)
            candidates = _candidate_backings(retrieval, pretrained)
            render_dir = out_dir / source_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            best: Dict[str, Any] | None = None
            for label, backing in candidates.items():
                cand_dir = render_dir / label
                cand_dir.mkdir(parents=True, exist_ok=True)
                accomp_path = cand_dir / "accompaniment_generated.wav"
                sf.write(str(accomp_path), backing, DIFFUSION_SR)
                final_mix = _mix_preserved_vocals(stems["vocals"], backing, cand_dir, vocal_gain=0.95, accomp_gain=1.0)
                probs = _judge_probs_for_audio(backing, judge, device, 256)
                target_idx = int(genre_to_idx[target])
                conf = float(probs[target_idx])
                margin = float(conf - float(np.max(np.delete(probs, target_idx))))
                metrics = _audio_metrics(source_for_metrics, backing, DIFFUSION_SR)
                local_score = _local_score(conf, margin, float(metrics["fullness"]), float(metrics["structure"]), float(metrics["warble"]))
                row = {
                    "song": source_key,
                    "target": target,
                    "winner_label": label,
                    "target_conf": conf,
                    "target_margin": margin,
                    "warble": float(metrics["warble"]),
                    "fullness": float(metrics["fullness"]),
                    "structure": float(metrics["structure"]),
                    "judge_probs": probs.tolist(),
                    "local_score": local_score,
                    "hybrid_wav": str(final_mix),
                    "accompaniment_wav": str(accomp_path),
                }
                _write(cand_dir / "candidate_summary.json", row)
                if best is None or row["local_score"] > best["local_score"]:
                    best = row
            assert best is not None
            chosen_dir = render_dir / "selected"
            chosen_dir.mkdir(parents=True, exist_ok=True)
            chosen_backing, sr = sf.read(best["accompaniment_wav"], dtype="float32")
            sf.write(str(chosen_dir / "accompaniment_generated.wav"), chosen_backing, sr)
            chosen_mix, sr = sf.read(best["hybrid_wav"], dtype="float32")
            sf.write(str(chosen_dir / "hybrid_longform_coherent.wav"), chosen_mix, sr)
            row = dict(best)
            row["hybrid_wav"] = str(chosen_dir / "hybrid_longform_coherent.wav")
            row["accompaniment_wav"] = str(chosen_dir / "accompaniment_generated.wav")
            rows.append(row)
            target_label_counts[str(target)][str(best["winner_label"])] += 1
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["job_idx", "target", "source_song", "winner_label", "hybrid_wav", "accompaniment_wav"])
                writer.writerow(
                    {
                        "job_idx": int(job_idx),
                        "target": str(target),
                        "source_song": source_key,
                        "winner_label": str(best["winner_label"]),
                        "hybrid_wav": row["hybrid_wav"],
                        "accompaniment_wav": row["accompaniment_wav"],
                    }
                )
            job_idx += 1

    sep_vals: List[float] = []
    by_song: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_song[str(row["song"])].append(np.asarray(row["judge_probs"], dtype=np.float32))
    for probs_list in by_song.values():
        for i in range(len(probs_list)):
            for j in range(i + 1, len(probs_list)):
                sep_vals.append(float(np.mean(np.abs(probs_list[i] - probs_list[j]))))
    mean_sep = float(np.mean(sep_vals)) if sep_vals else 0.0
    for row in rows:
        row["separation"] = mean_sep
        row["overall"] = float(row["local_score"] + 0.20 * mean_sep)

    winner_map = {target: counts.most_common(1)[0][0] for target, counts in target_label_counts.items() if counts}
    summary = {
        "out_dir": str(out_dir),
        "n_jobs": int(job_idx),
        "retrieval_suite_dir": str(retrieval_suite_dir),
        "pretrained_suite_dir": str(pretrained_suite_dir),
        "winner_map": winner_map,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in rows])) if rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in rows])) if rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in rows])) if rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in rows])) if rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in rows])) if rows else 0.0,
        "mean_separation": mean_sep,
    }
    (out_dir / "winner_map.json").write_text(json.dumps(winner_map, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "rows.json").write_text(json.dumps({"rows": rows}, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "diagnosis_report.md").write_text(
        "\n".join(
            [
                "# Retrieval + Pretrained Blend Production",
                "",
                "- This runner renders both the retrieval-fusion accompaniment and the aggressive pretrained Encodec accompaniment.",
                "- It scores direct blends locally and keeps the best candidate per song/target.",
                f"- Mean target confidence: {summary['mean_target_conf']:.4f}",
                f"- Mean target margin: {summary['mean_target_margin']:.4f}",
                f"- Mean fullness: {summary['mean_fullness']:.4f}",
                f"- Mean structure: {summary['mean_structure']:.4f}",
                f"- Mean warble: {summary['mean_warble']:.4f}",
                f"- Mean separation: {summary['mean_separation']:.4f}",
                f"- Winner map: {json.dumps(winner_map)}",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
