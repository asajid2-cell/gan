from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq_centercut", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _window_with_edge_padding(audio: np.ndarray, start_sample: int, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    end_sample = int(start_sample) + int(length)
    left_pad = max(0, -int(start_sample))
    right_pad = max(0, end_sample - len(audio))
    a = max(0, int(start_sample))
    b = min(len(audio), end_sample)
    core = audio[a:b].astype(np.float32)
    if left_pad:
        core = np.concatenate([np.full(left_pad, float(audio[0]), dtype=np.float32), core], axis=0)
    if right_pad:
        core = np.concatenate([core, np.full(right_pad, float(audio[-1]), dtype=np.float32)], axis=0)
    if len(core) < length:
        core = np.pad(core, (0, length - len(core)), mode="edge")
    return core[:length].astype(np.float32)


def _boundary_match(prev_tail: np.ndarray, next_head: np.ndarray) -> np.ndarray:
    prev_tail = np.asarray(prev_tail, dtype=np.float32)
    next_head = np.asarray(next_head, dtype=np.float32)
    if len(prev_tail) == 0 or len(next_head) == 0:
        return next_head.astype(np.float32)
    prev_mean = float(np.mean(prev_tail))
    next_mean = float(np.mean(next_head))
    prev_rms = float(np.sqrt(np.mean(prev_tail ** 2)) + 1e-8)
    next_rms = float(np.sqrt(np.mean(next_head ** 2)) + 1e-8)
    matched = (next_head - next_mean) * (prev_rms / next_rms) + prev_mean
    return matched.astype(np.float32)


def _append_with_boundary_blend(assembled: np.ndarray, segment: np.ndarray, blend_samples: int) -> np.ndarray:
    assembled = np.asarray(assembled, dtype=np.float32).reshape(-1)
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    if len(assembled) == 0:
        return segment.astype(np.float32)
    blend = int(min(blend_samples, len(assembled), len(segment)))
    if blend < 16:
        return np.concatenate([assembled, segment], axis=0).astype(np.float32)
    seg_head = _boundary_match(assembled[-blend:], segment[:blend])
    fade = np.linspace(0.0, 1.0, blend, dtype=np.float32)
    blended = ((1.0 - fade) * assembled[-blend:] + fade * seg_head).astype(np.float32)
    return np.concatenate([assembled[:-blend], blended, segment[blend:]], axis=0).astype(np.float32)


def _append_mel_with_boundary_blend(assembled: np.ndarray, segment: np.ndarray, blend_frames: int) -> np.ndarray:
    assembled = np.asarray(assembled, dtype=np.float32)
    segment = np.asarray(segment, dtype=np.float32)
    if assembled.size == 0:
        return segment.astype(np.float32)
    blend = int(min(blend_frames, assembled.shape[1], segment.shape[1]))
    if blend < 2:
        return np.concatenate([assembled, segment], axis=1).astype(np.float32)
    tail = assembled[:, -blend:]
    head = segment[:, :blend]
    tail_mean = np.mean(tail, axis=1, keepdims=True)
    head_mean = np.mean(head, axis=1, keepdims=True)
    head = head + 0.25 * (tail_mean - head_mean)
    fade = np.linspace(0.0, 1.0, blend, dtype=np.float32)[None, :]
    blended = ((1.0 - fade) * tail + fade * head).astype(np.float32)
    return np.concatenate([assembled[:, :-blend], blended, segment[:, blend:]], axis=1).astype(np.float32)


@torch.no_grad()
def _predict_window_mel(
    module: Any,
    model: Any,
    arrays: Dict[str, np.ndarray],
    mel_min: float,
    mel_max: float,
    max_frames: int,
    device: torch.device,
    source_window_audio: np.ndarray,
    donor_idx: int,
    target_genre_idx: int,
    prev_pred_mel: np.ndarray | None,
    prev2_pred_mel: np.ndarray | None,
) -> np.ndarray:
    mel_raw = module.extract_bigvgan_mel_np(source_window_audio, sr=module.DIFFUSION_SR)
    mel_len = int(min(max_frames, mel_raw.shape[1]))
    mel = module.pad_or_trim(mel_raw, max_frames, axis=1, pad_val=float(mel_min))
    mel_norm = module._normalize_mel_np(mel, mel_min, mel_max)
    struct = module._structure_proxy_from_mel(mel_norm)
    donor_mel = module.pad_or_trim(
        np.asarray(arrays["mel"][int(donor_idx)], dtype=np.float32),
        max_frames,
        axis=1,
        pad_val=float(mel_min),
    )
    donor_norm = module._normalize_mel_np(donor_mel, mel_min, mel_max)
    cond_feat = module._cond_feat_from_audio(source_window_audio, max_frames, mel_norm.shape[0])
    if prev_pred_mel is None:
        prev_norm = np.zeros_like(mel_norm, dtype=np.float32)
        context_count = 0.0
    else:
        prev_norm = module.pad_or_trim(
            np.asarray(prev_pred_mel, dtype=np.float32),
            max_frames,
            axis=1,
            pad_val=0.0,
        )
        context_count = 0.5 if prev2_pred_mel is None else 1.0
    if prev2_pred_mel is None:
        prev2_norm = np.zeros_like(mel_norm, dtype=np.float32)
    else:
        prev2_norm = module.pad_or_trim(
            np.asarray(prev2_pred_mel, dtype=np.float32),
            max_frames,
            axis=1,
            pad_val=0.0,
        )
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
    if prev_pred_mel is not None:
        pred_mel = module.smooth_mel_tensor(torch.from_numpy(pred_mel[None, None, :, :]), time_kernel=7, freq_kernel=3)[0, 0].cpu().numpy().astype(np.float32)
        warm_cols = min(96, pred_mel.shape[1], prev_norm.shape[1])
        pred_mel[:, :warm_cols] = 0.82 * prev_norm[:, -warm_cols:] + 0.18 * pred_mel[:, :warm_cols]
    return pred_mel[:, :mel_len].astype(np.float32)


@torch.no_grad()
def generate_longform_centercut(
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
    vocoder: Any,
    device: torch.device,
) -> np.ndarray:
    chunk_seconds = float(max(6.0, (float(max_frames) / 320.0) * 3.0))
    hop_seconds = float(max(3.0, ((float(max_frames) / 320.0) - 1.0) * 3.0))
    center_keep_seconds = max(1.0, chunk_seconds - 2.0 * hop_seconds)
    sr = int(module.DIFFUSION_SR)
    chunk_samples = int(round(chunk_seconds * sr))
    hop_samples = int(round(hop_seconds * sr))
    center_keep_samples = int(round(center_keep_seconds * sr))
    left_context_samples = int(round((chunk_seconds - center_keep_seconds) * 0.5 * sr))
    blend_frames = 8
    total_samples = len(source_audio)
    donor_rows = donor_track["rows"]

    def donor_idx_for(progress_idx: int, total_steps: int) -> int:
        if total_steps <= 1:
            return int(donor_rows[0])
        frac = float(progress_idx) / float(max(1, total_steps - 1))
        row_idx = int(donor_rows[min(len(donor_rows) - 1, round(frac * max(0, len(donor_rows) - 1)))])
        return row_idx

    full_mel = np.zeros((80, 0), dtype=np.float32)
    prev_pred_mel = None
    prev2_pred_mel = None

    first_window = _window_with_edge_padding(source_audio, 0, chunk_samples)
    first_donor_idx = donor_idx_for(0, max(1, int(np.ceil(max(0, total_samples - chunk_samples) / float(hop_samples))) + 1))
    first_pred_mel = _predict_window_mel(
        module,
        model,
        arrays,
        mel_min,
        mel_max,
        max_frames,
        device,
        first_window,
        first_donor_idx,
        target_genre_idx,
        prev_pred_mel,
        prev2_pred_mel,
    )
    prev2_pred_mel = None if prev_pred_mel is None else prev_pred_mel.copy()
    prev_pred_mel = first_pred_mel.copy()
    first_keep = min(total_samples, chunk_samples)
    first_keep_frames = int(round(float(first_keep) / float(max(1, chunk_samples)) * float(first_pred_mel.shape[1])))
    full_mel = first_pred_mel[:, : max(1, first_keep_frames)].astype(np.float32)

    later_starts = list(range(int(round(chunk_seconds)), total_samples, hop_samples))
    total_steps = len(later_starts) + 1
    for step_idx, segment_start in enumerate(later_starts, start=1):
        window_start = int(segment_start) - left_context_samples
        source_window = _window_with_edge_padding(source_audio, window_start, chunk_samples)
        donor_idx = donor_idx_for(step_idx, total_steps)
        pred_mel = _predict_window_mel(
            module,
            model,
            arrays,
            mel_min,
            mel_max,
            max_frames,
            device,
            source_window,
            donor_idx,
            target_genre_idx,
            prev_pred_mel,
            prev2_pred_mel,
        )
        prev2_pred_mel = None if prev_pred_mel is None else prev_pred_mel.copy()
        prev_pred_mel = pred_mel.copy()
        seg_start_frames = int(round(float(left_context_samples) / float(max(1, chunk_samples)) * float(pred_mel.shape[1])))
        seg_len_frames = int(round(float(center_keep_samples) / float(max(1, chunk_samples)) * float(pred_mel.shape[1])))
        seg = pred_mel[:, seg_start_frames : seg_start_frames + max(1, seg_len_frames)].astype(np.float32)
        if seg.shape[1] == 0:
            break
        full_mel = _append_mel_with_boundary_blend(full_mel, seg, blend_frames=blend_frames)
    audio = np.asarray(
        module.vocode_bigvgan(
            torch.from_numpy(full_mel[None, None, :, :]).to(device),
            float(mel_min),
            float(mel_max),
            vocoder,
            device,
        ),
        dtype=np.float32,
    ).reshape(-1)
    return module._pad_audio(audio, total_samples)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerender an existing block suite using center-cut seam stabilization.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    summary_path = suite_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    suite_summary = json.loads(summary_path.read_text(encoding="utf-8"))
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

    combined_dir = suite_dir / "combined_pack_centercut"
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
            raise RuntimeError("Judge genre mismatch during centercut rerender.")
        model = module.RetrievalFusionUNet(in_ch=18, num_genres=len(genre_to_idx), base_ch=int(run_cfg.base_ch)).to(device)
        payload = torch.load(str(Path(tr["summary"]["best_checkpoint"])), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        target = str(tr["target"])
        out_dir = run_dir / "final_pack_centercut"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        for song_key, song in song_map.items():
            stems = module._resolve_stems(hybrid_cfg, song)
            source_acc = module.load_audio_chunk(stems["accompaniment"], sample_rate=module.DIFFUSION_SR, seconds=float(run_cfg.final_seconds), start_sec=0.0)
            donor_track = module._choose_donor_track(source_acc, track_bank, keep)
            accomp = generate_longform_centercut(
                module,
                model,
                source_audio=source_acc,
                target_genre_idx=keep,
                donor_track=donor_track,
                arrays=arrays,
                mel_min=float(meta.mel_min),
                mel_max=float(meta.mel_max),
                max_frames=int(run_cfg.max_frames),
                vocoder=vocoder,
                device=device,
            )
            accomp = module._pad_audio(accomp, len(source_acc))
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
        "combined_pack_centercut": str(combined_dir),
        "target_runs": target_runs_out,
        "mean_overall": float(np.mean([r["overall"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in combined_rows])) if combined_rows else 0.0,
    }
    (suite_dir / "centercut_rerender_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    baseline = {
        "mean_overall": float(suite_summary["mean_overall"]),
        "mean_target_conf": float(suite_summary["mean_target_conf"]),
        "mean_target_margin": float(suite_summary["mean_target_margin"]),
        "mean_warble": float(suite_summary["mean_warble"]),
        "mean_fullness": float(suite_summary["mean_fullness"]),
        "mean_structure": float(suite_summary["mean_structure"]),
    }
    compare = {}
    for key, val in baseline.items():
        compare[key] = {"original": val, "centercut": float(summary[key]), "delta": float(summary[key]) - float(val)}
    (suite_dir / "centercut_vs_original.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
