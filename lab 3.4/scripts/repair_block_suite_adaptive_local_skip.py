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


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq_adaptive_local_skip", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _repair_with_skip(audio: np.ndarray, seam: int, sr: int, skip_ms: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    skip = int(round(float(skip_ms) * 0.001 * float(sr)))
    if skip < 32:
        return out
    a = seam - skip
    b = seam + skip
    c = seam + 2 * skip
    if a < 0 or c > len(out):
        return out
    left = out[a:seam].astype(np.float32)
    right = out[b:c].astype(np.float32)
    if len(left) != skip or len(right) != skip:
        return out
    fade = np.linspace(0.0, 1.0, 2 * skip, dtype=np.float32)
    left_pad = np.concatenate([left, left[::-1]], axis=0)[: 2 * skip]
    right_pad = np.concatenate([right[:1].repeat(skip), right], axis=0)[: 2 * skip]
    bridge = ((1.0 - fade) * left_pad + fade * right_pad).astype(np.float32)
    out[a:b] = bridge
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _local_score(audio: np.ndarray, seam: int, sr: int, skip_ms: float) -> Tuple[float, Dict[str, float]]:
    cand = _repair_with_skip(audio, seam, sr, skip_ms)
    probe = int(round(0.046 * float(sr)))
    left = cand[max(0, seam - probe):seam]
    right = cand[seam:min(len(cand), seam + probe)]
    n = min(len(left), len(right))
    if n < 256:
        return 1e9, {"boundary_l1": 1e9, "roughness": 1e9, "env_gap": 1e9}
    left = left[-n:]
    right = right[:n]
    boundary_l1 = float(np.mean(np.abs(left - right)))
    roughness = float(np.mean(np.abs(np.diff(np.concatenate([left[-256:], right[:256]])))))
    env_gap = float(abs(np.sqrt(np.mean(left ** 2) + 1e-8) - np.sqrt(np.mean(right ** 2) + 1e-8)))
    skip_penalty = 0.000004 * float(skip_ms)
    score = boundary_l1 + 0.45 * roughness + 0.35 * env_gap + skip_penalty
    return float(score), {
        "boundary_l1": boundary_l1,
        "roughness": roughness,
        "env_gap": env_gap,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair blend35 block-suite seams using per-seam adaptive skip selection.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--input-pack", type=Path, required=True)
    ap.add_argument("--candidates-ms", type=float, nargs="*", default=[80.0, 120.0, 160.0, 240.0])
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    input_pack = Path(args.input_pack)
    suite_summary = json.loads((suite_dir / "summary.json").read_text(encoding="utf-8"))
    module = _load_module()

    first_run = Path(suite_summary["target_runs"][0]["run_dir"])
    first_cfg = module.TrainConfig(**json.loads((first_run / "config.json").read_text(encoding="utf-8")))
    device = module._device_from_arg(str(first_cfg.device))
    judge, judge_genre_to_idx = module._load_or_train_judge(Path(first_cfg.judge_ckpt), Path(first_cfg.cache_dir), suite_dir, device, int(first_cfg.max_frames))
    hybrid_cfg = module.HybridPushConfig()
    song_map = {module._slug(Path(song["path"]).stem): song for song in module.picked_songs()}

    suffix = "adaptive_local_skip"
    combined_dir = suite_dir / f"combined_pack_{suffix}"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    target_runs_out: List[Dict[str, Any]] = []
    combined_rows: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    separation_vals: List[float] = []

    for tr in suite_summary["target_runs"]:
        run_dir = Path(tr["run_dir"])
        run_cfg = module.TrainConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        target = str(tr["target"])
        out_dir = run_dir / f"final_pack_{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []

        for song_key, song in song_map.items():
            stems = module._resolve_stems(hybrid_cfg, song)
            source_acc = module.load_audio_chunk(stems["accompaniment"], sample_rate=module.DIFFUSION_SR, seconds=float(run_cfg.final_seconds), start_sec=0.0)
            chunks = module.split_audio_overlapping(source_acc, chunk_seconds=9.0, overlap_seconds=6.0, sr=module.DIFFUSION_SR)
            seams = [int(chunk["start_sample"] + round(6.0 * module.DIFFUSION_SR)) for chunk in chunks[1:]]
            src_wav = input_pack / song_key / target / "accompaniment_generated.wav"
            if not src_wav.exists():
                continue
            audio, sr = sf.read(str(src_wav), dtype="float32")
            if sr != module.DIFFUSION_SR:
                raise RuntimeError(f"Unexpected sr={sr} for {src_wav}")
            if audio.ndim > 1:
                audio = audio[:, 0]
            repaired = module._pad_audio(np.asarray(audio, dtype=np.float32), len(source_acc))

            seam_debug: List[Dict[str, float]] = []
            for seam in seams:
                best_audio = repaired
                best_skip = float(args.candidates_ms[0])
                best_score = 1e9
                best_metrics: Dict[str, float] = {}
                for skip_ms in args.candidates_ms:
                    cand = _repair_with_skip(repaired, seam, module.DIFFUSION_SR, float(skip_ms))
                    score, metrics = _local_score(cand, seam, module.DIFFUSION_SR, float(skip_ms))
                    if score < best_score:
                        best_score = score
                        best_skip = float(skip_ms)
                        best_audio = cand
                        best_metrics = metrics
                repaired = best_audio
                seam_debug.append(
                    {
                        "seam_sample": float(seam),
                        "selected_skip_ms": best_skip,
                        "score": float(best_score),
                        **best_metrics,
                    }
                )

            render_dir = out_dir / "renders" / song_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), repaired, module.DIFFUSION_SR)
            mix_path = module._mix_preserved_vocals(stems["vocals"], repaired, render_dir, vocal_gain=0.95, accomp_gain=1.0)
            probs = module._judge_probs_for_audio(repaired, judge, device, int(run_cfg.max_frames))
            target_idx = int(judge_genre_to_idx[target])
            tgt_conf = float(probs[target_idx])
            tgt_margin = float(tgt_conf - float(np.max(np.delete(probs, target_idx))))
            metrics = module._audio_metrics(source_acc, repaired, module.DIFFUSION_SR)
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
                "judge_probs": probs.tolist(),
                "selected_skips_ms": [float(x["selected_skip_ms"]) for x in seam_debug],
            }
            rows.append(row)
            combined_rows.append(row)
            debug_rows.append({"song": song_key, "target": target, "seams": seam_debug})
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
        "input_pack": str(input_pack),
        "combined_pack": str(combined_dir),
        "candidates_ms": [float(x) for x in args.candidates_ms],
        "target_runs": target_runs_out,
        "mean_overall": float(np.mean([r["overall"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in combined_rows])) if combined_rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in combined_rows])) if combined_rows else 0.0,
    }
    (suite_dir / "adaptive_local_skip_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (suite_dir / "adaptive_local_skip_debug.json").write_text(json.dumps(debug_rows, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
