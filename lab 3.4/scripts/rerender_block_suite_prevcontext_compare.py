from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq_prevctx", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _source_context_norm(module: Any, audio: np.ndarray, max_frames: int, mel_min: float, mel_max: float) -> np.ndarray:
    mel_raw = module.extract_bigvgan_mel_np(audio, sr=module.DIFFUSION_SR)
    mel = module.pad_or_trim(mel_raw, max_frames, axis=1, pad_val=float(mel_min))
    return module._normalize_mel_np(mel, mel_min, mel_max).astype(np.float32)


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
        elif prev_mode == "zero":
            prev_norm = zeros
            prev2_norm = zeros
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
            raise ValueError(f"Unknown prev_mode={prev_mode}")
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


def _seam_local_metrics(audio: np.ndarray, seam_positions: List[int], sr: int) -> Dict[str, float]:
    import librosa
    boundary: List[float] = []
    rough: List[float] = []
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    for seam in seam_positions:
        if seam - 4096 < 0 or seam + 4096 > len(y):
            continue
        left = y[seam - 2048 : seam]
        right = y[seam : seam + 2048]
        mleft = librosa.feature.melspectrogram(y=left, sr=sr, n_fft=1024, hop_length=256, n_mels=40, power=1.0)
        mright = librosa.feature.melspectrogram(y=right, sr=sr, n_fft=1024, hop_length=256, n_mels=40, power=1.0)
        boundary.append(float(np.mean(np.abs(np.log1p(mleft) - np.log1p(mright)))))
        seam_win = y[seam - 4096 : seam + 4096]
        spec = np.abs(librosa.stft(seam_win, n_fft=1024, hop_length=256))
        hf = spec[spec.shape[0] // 2 :, :]
        rough.append(float(np.mean(np.abs(np.diff(hf, axis=1)))))
    return {
        "mean_boundary_l1": float(np.mean(boundary)) if boundary else 0.0,
        "mean_seam_roughness": float(np.mean(rough)) if rough else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare block-suite inference with different previous-context modes.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--modes", nargs="*", default=["generated", "zero", "source", "blend35"])
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    suite_summary = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
    module = _load_module()

    first_run = Path(suite_summary["target_runs"][0]["run_dir"])
    first_cfg = module.TrainConfig(**json.loads((first_run / "config.json").read_text(encoding="utf-8")))
    device = module._device_from_arg(str(first_cfg.device))
    index_df, arrays, genre_to_idx, meta = module.load_diffusion_cache(Path(first_cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    vocoder = module.load_bigvgan_robust(device=device)
    hybrid_cfg = module.HybridPushConfig()
    song_map = {module._slug(Path(song["path"]).stem): song for song in module.picked_songs()}

    mode_reports: Dict[str, Any] = {}
    best_mode = None
    best_score = -1e18

    for prev_mode in args.modes:
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
                raise RuntimeError("Judge genre mismatch during prev-mode rerender.")
            model = module.RetrievalFusionUNet(in_ch=18, num_genres=len(genre_to_idx), base_ch=int(run_cfg.base_ch)).to(device)
            payload = torch.load(str(Path(tr["summary"]["best_checkpoint"])), map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            model.eval()
            target = str(tr["target"])
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
                    prev_mode=prev_mode,
                )
                accomp = module._pad_audio(accomp, len(source_acc))
                probs = module._judge_probs_for_audio(accomp, judge, device, int(run_cfg.max_frames))
                target_idx = int(genre_to_idx[target])
                tgt_conf = float(probs[target_idx])
                tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
                metrics = module._audio_metrics(source_acc, accomp, module.DIFFUSION_SR)
                chunks = module.split_audio_overlapping(source_acc, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds, sr=module.DIFFUSION_SR)
                overlap_samples = int(round(overlap_seconds * module.DIFFUSION_SR))
                seams = [int(chunk["start_sample"] + overlap_samples) for chunk in chunks[1:]]
                seam_metrics = _seam_local_metrics(accomp, seams, module.DIFFUSION_SR)
                combined_rows.append(
                    {
                        "song": song_key,
                        "target": target,
                        "target_conf": tgt_conf,
                        "target_margin": tgt_margin,
                        "warble": float(metrics["warble"]),
                        "fullness": float(metrics["fullness"]),
                        "structure": float(metrics["structure"]),
                        "mean_boundary_l1": seam_metrics["mean_boundary_l1"],
                        "mean_seam_roughness": seam_metrics["mean_seam_roughness"],
                        "judge_probs": probs.tolist(),
                    }
                )
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
        report = {
            "mean_overall": float(np.mean([r["overall"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_target_conf": float(np.mean([r["target_conf"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_target_margin": float(np.mean([r["target_margin"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_warble": float(np.mean([r["warble"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_fullness": float(np.mean([r["fullness"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_structure": float(np.mean([r["structure"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_boundary_l1": float(np.mean([r["mean_boundary_l1"] for r in combined_rows])) if combined_rows else 0.0,
            "mean_seam_roughness": float(np.mean([r["mean_seam_roughness"] for r in combined_rows])) if combined_rows else 0.0,
            "rows": combined_rows,
        }
        mode_reports[prev_mode] = report
        score = float(report["mean_overall"] - 0.25 * report["mean_boundary_l1"] - 0.10 * report["mean_seam_roughness"])
        if score > best_score:
            best_score = score
            best_mode = prev_mode

    output = {"suite_dir": str(suite_dir), "mode_reports": mode_reports, "selected_mode": best_mode}
    out_path = suite_dir / "prevcontext_compare.json"
    out_path.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
