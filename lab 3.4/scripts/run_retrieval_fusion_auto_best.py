from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch


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
from dggr.lab3_diffusion_data import DIFFUSION_SR, load_diffusion_cache
from dggr.lab3_diffusion_train import load_bigvgan_robust
from run_hybrid_vocal_push_compare import HybridPushConfig, TARGET_GENRES, _json_default, _resolve_stems, picked_songs
from train_scratch_retrieval_fusion import (
    RetrievalFusionUNet,
    _build_track_bank,
    _choose_donor_track,
    _device_from_arg,
    _mix_preserved_vocals,
    _slug,
    generate_longform,
)


def _songs_from_downloads(root: Path, limit: int, song_filter: str = "") -> List[Dict[str, Any]]:
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = [p for p in sorted(root.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if song_filter.strip():
        pat = song_filter.strip().lower()
        files = [p for p in files if pat in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No audio files found in {root}")
    picked = files[:limit]
    return [{"path": p, "source_genre": "cc0_other"} for p in picked]


def _load_target_models(suite_dir: Path, device: torch.device) -> Dict[str, RetrievalFusionUNet]:
    winner_map = json.loads((suite_dir / "winner_map.json").read_text(encoding="utf-8"))
    models: Dict[str, RetrievalFusionUNet] = {}
    for target, ckpt in winner_map.items():
        payload = torch.load(str(ckpt), map_location=device, weights_only=False)
        model = RetrievalFusionUNet(in_ch=16, num_genres=len(payload["genre_to_idx"]), base_ch=int(payload["cfg"]["base_ch"])).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        models[str(target)] = model
    return models


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the selected scratch retrieval-fusion production path.")
    ap.add_argument("--retrieval-suite-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "Desktop" / "dggr_retrieval_fusion_production")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d001" / "cache")
    ap.add_argument("--downloads-dir", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--song-filter", type=str, default="")
    ap.add_argument("--use-picked-songs", action="store_true")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    device = _device_from_arg(str(args.device))
    suite_dir = Path(args.retrieval_suite_dir)
    selected_dir = suite_dir / "selected_pack_final"
    if not selected_dir.exists():
        raise FileNotFoundError(f"Missing selected_pack_final in {suite_dir}")
    postmix_map = json.loads((selected_dir / "winner_map.json").read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir) / f"retrieval_prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_df, arrays, genre_to_idx, meta = load_diffusion_cache(Path(args.cache_dir), mmap=True)
    all_indices = np.arange(len(index_df), dtype=np.int64)
    track_bank = _build_track_bank(index_df, arrays, all_indices)
    models = _load_target_models(suite_dir, device)
    payload_map = {
        str(target): torch.load(str(ckpt), map_location=device, weights_only=False)
        for target, ckpt in json.loads((suite_dir / "winner_map.json").read_text(encoding="utf-8")).items()
    }
    vocoder = load_bigvgan_robust(device=device)
    hybrid_cfg = HybridPushConfig()

    songs = picked_songs() if bool(args.use_picked_songs) else _songs_from_downloads(Path(args.downloads_dir), limit=max(1, int(args.limit)), song_filter=args.song_filter)
    (out_dir / "config.json").write_text(json.dumps({"retrieval_suite_dir": str(suite_dir), "postmix_map": postmix_map}, indent=2), encoding="utf-8")

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["job_idx", "target", "source_song", "postmix_mode", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    job_idx = 0
    for song in songs:
        stems = _resolve_stems(hybrid_cfg, song)
        source_key = _slug(Path(song["path"]).stem)
        source_acc = load_audio_chunk(stems["accompaniment"], sample_rate=DIFFUSION_SR, seconds=60.0, start_sec=0.0)
        for target in TARGET_GENRES:
            donor_track = _choose_donor_track(source_acc, track_bank, int(genre_to_idx[target]))
            model = models[str(target)]
            render_dir = out_dir / source_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            max_frames = int(payload_map[str(target)]["cfg"].get("max_frames", 256))
            gen = generate_longform(
                model,
                source_audio=source_acc,
                target_genre_idx=int(genre_to_idx[target]),
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(meta.mel_min),
                mel_max=float(meta.mel_max),
                max_frames=max_frames,
                chunk_seconds=3.0,
                overlap_seconds=0.5,
                vocoder=vocoder,
                device=device,
            )
            n = min(len(source_acc), len(gen))
            src = source_acc[:n].astype(np.float32)
            gen = gen[:n].astype(np.float32)
            mode = str(postmix_map[str(target)])
            if mode == "full_020":
                gen = (0.8 * gen + 0.2 * src).astype(np.float32)
            peak = max(1e-6, float(np.max(np.abs(gen))))
            gen = (gen / max(1.0, peak)).astype(np.float32)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), gen, DIFFUSION_SR)
            final_mix = _mix_preserved_vocals(stems["vocals"], gen, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["job_idx", "target", "source_song", "postmix_mode", "hybrid_wav", "accompaniment_wav"])
                writer.writerow(
                    {
                        "job_idx": int(job_idx),
                        "target": str(target),
                        "source_song": source_key,
                        "postmix_mode": mode,
                        "hybrid_wav": str(final_mix),
                        "accompaniment_wav": str(accomp_path),
                    }
                )
            job_idx += 1

    summary = {"out_dir": str(out_dir), "n_jobs": int(job_idx), "retrieval_suite_dir": str(suite_dir), "postmix_map": postmix_map}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
