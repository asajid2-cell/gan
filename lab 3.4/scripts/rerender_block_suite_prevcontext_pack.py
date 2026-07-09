from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq_prevpack", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _bridge_skip_region(generated: np.ndarray, seam_positions: List[int], sr: int, skip_ms: float) -> np.ndarray:
    out = np.asarray(generated, dtype=np.float32).copy()
    skip = int(round(float(skip_ms) * 0.001 * float(sr)))
    if skip < 32:
        return out
    for seam in seam_positions:
        seam = int(seam)
        a = seam - skip
        b = seam + skip
        c = seam + 2 * skip
        if a < 0 or c > len(out):
            continue
        left = out[a:seam].astype(np.float32)
        right = out[b:c].astype(np.float32)
        if len(left) != skip or len(right) != skip:
            continue
        fade = np.linspace(0.0, 1.0, 2 * skip, dtype=np.float32)
        left_pad = np.concatenate([left, left[::-1]], axis=0)[: 2 * skip]
        right_pad = np.concatenate([right[:1].repeat(skip), right], axis=0)[: 2 * skip]
        bridge = ((1.0 - fade) * left_pad + fade * right_pad).astype(np.float32)
        out[a:b] = bridge
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _load_skip_map(path: Path | None) -> Dict[str, float]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.items()}


@torch.no_grad()
def generate_longform_prevmode(
    module: Any,
    model: Any,
    *,
    source_audio: np.ndarray,
    target_genre_idx: int,
    donor_track: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    mel_min: float,
    mel_max: float,
    max_frames: int,
    chunk_seconds: float,
    overlap_seconds: float,
    vocoder: Any,
    device: torch.device,
    prev_mode: str,
) -> np.ndarray:
    model.eval()
    chunks = module.split_audio_overlapping(source_audio, chunk_seconds=float(chunk_seconds), overlap_seconds=float(overlap_seconds), sr=module.DIFFUSION_SR)
    total = max(1, len(chunks) - 1)
    donor_rows = donor_track["rows"]
    out_mels: List[np.ndarray] = []
    trim_frames: List[int] = []
    prev_pred_mel: np.ndarray | None = None
    prev2_pred_mel: np.ndarray | None = None
    prev_src_mel: np.ndarray | None = None
    prev2_src_mel: np.ndarray | None = None

    for i, chunk in enumerate(chunks):
        mel_raw = module.extract_bigvgan_mel_np(chunk["audio"], sr=module.DIFFUSION_SR)
        mel_len = int(min(max_frames, mel_raw.shape[1]))
        mel = module.pad_or_trim(mel_raw, max_frames, axis=1, pad_val=float(mel_min))
        mel_norm = module._normalize_mel_np(mel, mel_min, mel_max)
        struct = module._structure_proxy_from_mel(mel_norm)
        donor_idx = int(donor_rows[min(len(donor_rows) - 1, round((i / total) * max(0, len(donor_rows) - 1)))])
        donor_mel = module.pad_or_trim(np.asarray(arrays["mel"][donor_idx], dtype=np.float32), max_frames, axis=1, pad_val=float(mel_min))
        donor_norm = module._normalize_mel_np(donor_mel, mel_min, mel_max)
        cond_feat = module._cond_feat_from_audio(chunk["audio"], max_frames, mel_norm.shape[0])
        src_ctx = mel_norm.astype(np.float32)
        zeros = np.zeros_like(src_ctx, dtype=np.float32)

        if prev_mode == "generated":
            prev_norm = zeros if prev_pred_mel is None else prev_pred_mel.astype(np.float32)
            prev2_norm = zeros if prev2_pred_mel is None else prev2_pred_mel.astype(np.float32)
        elif prev_mode == "source":
            prev_norm = zeros if prev_src_mel is None else prev_src_mel.astype(np.float32)
            prev2_norm = zeros if prev2_src_mel is None else prev2_src_mel.astype(np.float32)
        elif prev_mode == "blend35":
            src_prev = zeros if prev_src_mel is None else prev_src_mel.astype(np.float32)
            src_prev2 = zeros if prev2_src_mel is None else prev2_src_mel.astype(np.float32)
            gen_prev = zeros if prev_pred_mel is None else prev_pred_mel.astype(np.float32)
            gen_prev2 = zeros if prev2_pred_mel is None else prev2_pred_mel.astype(np.float32)
            prev_norm = 0.35 * gen_prev + 0.65 * src_prev
            prev2_norm = 0.35 * gen_prev2 + 0.65 * src_prev2
        else:
            raise ValueError(f"Unsupported prev_mode={prev_mode}")
        context_count = 0.0 if i == 0 else (0.5 if i == 1 else 1.0)

        pred = model(
            torch.from_numpy(struct[None, None, :, :]).to(device),
            torch.from_numpy(donor_norm[None, None, :, :]).to(device),
            torch.from_numpy(prev_norm[None, None, :, :]).to(device),
            torch.from_numpy(prev2_norm[None, None, :, :]).to(device),
            torch.from_numpy(cond_feat[None, :, :, :]).to(device),
            torch.tensor([int(target_genre_idx)], dtype=torch.long, device=device),
            torch.tensor([float(context_count)], dtype=torch.float32, device=device),
        )
        pred_mel = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        pred_mel[:12, :] = 0.55 * mel_norm[:12, :] + 0.45 * pred_mel[:12, :]
        if i > 0 and prev_mode in {"generated", "blend35"}:
            pred_mel = module.smooth_mel_tensor(torch.from_numpy(pred_mel[None, None, :, :]), time_kernel=7, freq_kernel=3)[0, 0].cpu().numpy().astype(np.float32)
            warm_cols = min(96, pred_mel.shape[1], prev_norm.shape[1])
            pred_mel[:, :warm_cols] = 0.82 * prev_norm[:, -warm_cols:] + 0.18 * pred_mel[:, :warm_cols]

        prev2_pred_mel = None if prev_pred_mel is None else prev_pred_mel.copy()
        prev_pred_mel = pred_mel.copy()
        prev2_src_mel = None if prev_src_mel is None else prev_src_mel.copy()
        prev_src_mel = src_ctx.copy()

        out_mels.append(pred_mel[:, :mel_len].astype(np.float32))
        frames_per_sample = float(mel_len) / float(max(1, len(chunk["audio"])))
        trim_frames.append(int(round(float(overlap_seconds) * float(module.DIFFUSION_SR) * frames_per_sample)))

    full_mel = module._assemble_mel_context_trim(out_mels, trim_frames)
    full_t = torch.from_numpy(full_mel[None, None, :, :]).to(device)
    audio = np.asarray(module.vocode_bigvgan(full_t, float(mel_min), float(mel_max), vocoder, device), dtype=np.float32).reshape(-1)
    return audio


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a full block-suite pack with a selected previous-context inference mode.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--prev-mode", choices=["generated", "source", "blend35"], required=True)
    ap.add_argument("--seamskip-ms", type=float, default=0.0)
    ap.add_argument("--skip-map-json", type=Path, default=None)
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    suite_summary = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
    module = _load_module()
    skip_map = _load_skip_map(args.skip_map_json)

    first_run = Path(suite_summary["target_runs"][0]["run_dir"])
    first_cfg = module.TrainConfig(**json.loads((first_run / "config.json").read_text(encoding="utf-8")))
    device = module._device_from_arg(str(first_cfg.device))
    index_df, arrays, genre_to_idx, meta = module.load_diffusion_cache(Path(first_cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    vocoder = module.load_bigvgan_robust(device=device)
    hybrid_cfg = module.HybridPushConfig()
    song_map = {module._slug(Path(song["path"]).stem): song for song in module.picked_songs()}

    if skip_map:
        skip_tag = "_".join(
            f"{target.replace('_', '-')}{int(round(ms))}"
            for target, ms in sorted(skip_map.items())
        )
        suffix = f"prev_{args.prev_mode}_skipmap_{skip_tag}"
    else:
        suffix = f"prev_{args.prev_mode}" + (f"_seamskip_{int(round(args.seamskip_ms))}ms" if args.seamskip_ms > 0 else "")
    combined_dir = suite_dir / f"combined_pack_{suffix}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    target_runs_out: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []
    separation_vals: List[float] = []

    for tr in suite_summary["target_runs"]:
        run_dir = Path(tr["run_dir"])
        run_cfg = module.TrainConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        keep = int(genre_to_idx[str(run_cfg.single_genre_target)])
        train_idx, _ = module.stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=int(run_cfg.seed))
        train_idx = train_idx[genre_idx[train_idx] == keep]
        track_bank = module._build_track_bank(index_df, arrays, train_idx)
        judge, judge_genre_to_idx = module._load_or_train_judge(Path(run_cfg.judge_ckpt), Path(run_cfg.cache_dir), run_dir, device, int(run_cfg.max_frames))
        if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
            raise RuntimeError("Judge genre mismatch during prev-context pack render.")
        model = module.RetrievalFusionUNet(in_ch=18, num_genres=len(genre_to_idx), base_ch=int(run_cfg.base_ch)).to(device)
        payload = torch.load(str(Path(tr["summary"]["best_checkpoint"])), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        target = str(tr["target"])
        out_dir = run_dir / f"final_pack_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        skip_ms = float(skip_map.get(target, args.seamskip_ms))
        for song_key, song in song_map.items():
            stems = module._resolve_stems(hybrid_cfg, song)
            source_acc = module.load_audio_chunk(stems["accompaniment"], sample_rate=module.DIFFUSION_SR, seconds=float(run_cfg.final_seconds), start_sec=0.0)
            donor_track = module._choose_donor_track(source_acc, track_bank, keep)
            chunk_seconds = float(max(6.0, (float(run_cfg.max_frames) / 320.0) * 3.0))
            overlap_seconds = float(max(3.0, ((float(run_cfg.max_frames) / 320.0) - 1.0) * 3.0))
            accomp = generate_longform_prevmode(
                module,
                model,
                source_audio=source_acc,
                target_genre_idx=keep,
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(meta.mel_min),
                mel_max=float(meta.mel_max),
                max_frames=int(run_cfg.max_frames),
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                vocoder=vocoder,
                device=device,
                prev_mode=args.prev_mode,
            )
            accomp = module._pad_audio(accomp, len(source_acc))
            if skip_ms > 0:
                chunks = module.split_audio_overlapping(source_acc, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds, sr=module.DIFFUSION_SR)
                overlap_samples = int(round(overlap_seconds * module.DIFFUSION_SR))
                seams = [int(chunk["start_sample"] + overlap_samples) for chunk in chunks[1:]]
                accomp = _bridge_skip_region(accomp, seams, module.DIFFUSION_SR, skip_ms)
            render_dir = out_dir / "renders" / song_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), accomp, module.DIFFUSION_SR)
            mix_path = module._mix_preserved_vocals(stems["vocals"], accomp, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = module._judge_probs_for_audio(accomp, judge, device, int(run_cfg.max_frames))
            target_idx = int(genre_to_idx[target])
            tgt_conf = float(probs[target_idx])
            tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
            metrics = module._audio_metrics(source_acc, accomp, module.DIFFUSION_SR)
            row = {
                "song": song_key,
                "target": target,
                "seamskip_ms": skip_ms,
                "target_conf": tgt_conf,
                "target_margin": tgt_margin,
                "warble": float(metrics["warble"]),
                "fullness": float(metrics["fullness"]),
                "structure": float(metrics["structure"]),
                "hybrid_wav": str(mix_path),
                "accompaniment_wav": str(accomp_path),
                "donor_track_id": donor_track["track_id"],
                "judge_probs": probs.tolist(),
            }
            rows.append(row)
            combined_rows.append(row)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
                writer.writerow(
                    {
                        "target": target,
                        "source_song": song_key,
                        "source_target_dir": str(render_dir),
                        "hybrid_wav": str(render_dir / "hybrid_longform_coherent.wav"),
                        "accompaniment_wav": str(accomp_path),
                    }
                )
            dst_dir = combined_dir / song_key / target
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ["hybrid_longform_coherent.wav", "accompaniment_generated.wav", "longform_coherent.wav", "backing_fixed.wav"]:
                src = render_dir / name
                if src.exists():
                    shutil.copyfile(src, dst_dir / name)
        target_runs_out.append({"target": target, "run_dir": str(run_dir), "out_dir": str(out_dir), "summary": {"rows": rows}})

    for i in range(len(combined_rows)):
        for j in range(i + 1, len(combined_rows)):
            pa = np.asarray(combined_rows[i]["judge_probs"], dtype=np.float32)
            pb = np.asarray(combined_rows[j]["judge_probs"], dtype=np.float32)
            separation_vals.append(float(np.mean(np.abs(pa - pb))))
    mean_sep = float(np.mean(separation_vals)) if separation_vals else 0.0
    for row in combined_rows:
        row["separation"] = mean_sep
        row["overall"] = float(
            0.32 * row["target_margin"] +
            0.18 * row["target_conf"] +
            0.28 * row["fullness"] +
            0.22 * row["structure"] +
            0.20 * row["separation"] -
            0.20 * row["warble"]
        )
    summary = {
        "suite_dir": str(suite_dir),
        "combined_pack": str(combined_dir),
        "prev_mode": args.prev_mode,
        "seamskip_ms": float(args.seamskip_ms),
        "skip_map": skip_map,
        "target_runs": target_runs_out,
        "mean_overall": float(np.mean([r["overall"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in combined_rows])) if combined_rows else 0.0,
    }
    out_path = suite_dir / f"{suffix}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
