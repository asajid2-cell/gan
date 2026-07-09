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


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq_repair", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _smooth_segment(x: np.ndarray, kernel_size: int = 129) -> np.ndarray:
    if kernel_size <= 3 or len(x) < kernel_size:
        return x.astype(np.float32)
    win = np.hanning(kernel_size).astype(np.float32)
    win /= max(1e-8, float(win.sum()))
    return np.convolve(x.astype(np.float32), win, mode="same").astype(np.float32)


def _repair_audio_seams(
    generated: np.ndarray,
    source: np.ndarray,
    seam_positions: List[int],
    sr: int,
    *,
    seam_half_seconds: float = 0.18,
    peak_source_mix: float = 0.38,
) -> np.ndarray:
    out = np.asarray(generated, dtype=np.float32).copy()
    source = np.asarray(source, dtype=np.float32)
    half = int(round(float(seam_half_seconds) * float(sr)))
    if half < 8:
        return out
    for seam in seam_positions:
        a = max(0, int(seam) - half)
        b = min(len(out), int(seam) + half)
        if b - a < 16:
            continue
        seg_gen = out[a:b].astype(np.float32)
        seg_src = source[a:b].astype(np.float32)
        rg = float(np.sqrt(np.mean(seg_gen ** 2)) + 1e-8)
        rs = float(np.sqrt(np.mean(seg_src ** 2)) + 1e-8)
        seg_src = seg_src * (rg / rs)
        seg_sm = _smooth_segment(seg_gen, kernel_size=min(129, (b - a) // 2 * 2 + 1))
        x = np.linspace(-1.0, 1.0, b - a, dtype=np.float32)
        source_w = (0.5 * (np.cos(np.pi * x) + 1.0) * float(peak_source_mix)).astype(np.float32)
        repaired = (1.0 - source_w) * seg_sm + source_w * seg_src
        out[a:b] = repaired.astype(np.float32)
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair seam windows for an existing block suite using source-anchored local smoothing.")
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
    judge, judge_genre_to_idx = module._load_or_train_judge(Path(first_cfg.judge_ckpt), Path(first_cfg.cache_dir), suite_dir, device, int(first_cfg.max_frames))
    if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
        raise RuntimeError("Judge genre mismatch during seam repair.")
    hybrid_cfg = module.HybridPushConfig()
    song_map = {module._slug(Path(song["path"]).stem): song for song in module.picked_songs()}

    combined_dir = suite_dir / "combined_pack_seamrepair"
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
        target = str(tr["target"])
        final_pack = Path(str(tr["summary"]["final_pack_dir"]))
        out_dir = run_dir / "final_pack_seamrepair"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for row in tr["summary"]["final_summary"]["rows"]:
            song_key = str(row["song"])
            song = song_map[song_key]
            stems = module._resolve_stems(hybrid_cfg, song)
            source_acc = module.load_audio_chunk(stems["accompaniment"], sample_rate=module.DIFFUSION_SR, seconds=float(run_cfg.final_seconds), start_sec=0.0)
            gen_path = final_pack / "renders" / song_key / target / "accompaniment_generated.wav"
            gen, sr = sf.read(str(gen_path), dtype="float32")
            if sr != module.DIFFUSION_SR:
                raise RuntimeError(f"Unexpected sr={sr} for {gen_path}")
            gen = np.asarray(gen, dtype=np.float32).reshape(-1)
            gen = module._pad_audio(gen, len(source_acc))
            chunk_seconds = float(max(6.0, (float(run_cfg.max_frames) / 320.0) * 3.0))
            overlap_seconds = float(max(3.0, ((float(run_cfg.max_frames) / 320.0) - 1.0) * 3.0))
            chunks = module.split_audio_overlapping(source_acc, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds, sr=module.DIFFUSION_SR)
            overlap_samples = int(round(overlap_seconds * module.DIFFUSION_SR))
            seam_positions = []
            for chunk in chunks[1:]:
                seam = int(chunk["start_sample"]) + overlap_samples
                if 0 < seam < len(gen):
                    seam_positions.append(seam)
            repaired = _repair_audio_seams(gen, source_acc, seam_positions, module.DIFFUSION_SR)
            render_dir = out_dir / "renders" / song_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), repaired, module.DIFFUSION_SR)
            mix_path = module._mix_preserved_vocals(stems["vocals"], repaired, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = module._judge_probs_for_audio(repaired, judge, device, int(run_cfg.max_frames))
            target_idx = int(genre_to_idx[target])
            tgt_conf = float(probs[target_idx])
            tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
            metrics = module._audio_metrics(source_acc, repaired, module.DIFFUSION_SR)
            out_row = {
                "song": song_key,
                "target": target,
                "target_conf": tgt_conf,
                "target_margin": tgt_margin,
                "warble": float(metrics["warble"]),
                "fullness": float(metrics["fullness"]),
                "structure": float(metrics["structure"]),
                "hybrid_wav": str(mix_path),
                "accompaniment_wav": str(accomp_path),
                "donor_track_id": row["donor_track_id"],
                "judge_probs": probs.tolist(),
            }
            rows.append(out_row)
            combined_rows.append(out_row)
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
        mean_sep = 0.0
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
        "combined_pack_seamrepair": str(combined_dir),
        "target_runs": target_runs_out,
        "mean_overall": float(np.mean([r["overall"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in combined_rows])) if combined_rows else 0.0,
    }
    (suite_dir / "seamrepair_rerender_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
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
        compare[key] = {"original": val, "seamrepair": float(summary[key]), "delta": float(summary[key]) - float(val)}
    (suite_dir / "seamrepair_vs_original.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
