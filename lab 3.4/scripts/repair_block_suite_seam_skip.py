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
    spec = importlib.util.spec_from_file_location("blockseq_skiprepair", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def _bridge_skip_region(
    generated: np.ndarray,
    seam_positions: List[int],
    sr: int,
    *,
    skip_ms: float,
) -> np.ndarray:
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


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "mean_overall": float(np.mean([r["overall"] for r in rows])) if rows else 0.0,
        "mean_target_conf": float(np.mean([r["target_conf"] for r in rows])) if rows else 0.0,
        "mean_target_margin": float(np.mean([r["target_margin"] for r in rows])) if rows else 0.0,
        "mean_warble": float(np.mean([r["warble"] for r in rows])) if rows else 0.0,
        "mean_fullness": float(np.mean([r["fullness"] for r in rows])) if rows else 0.0,
        "mean_structure": float(np.mean([r["structure"] for r in rows])) if rows else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Repair block-suite seams by skipping unstable startup samples of each new segment and bridging with generated audio only.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    ap.add_argument("--skip-ms", type=float, nargs="*", default=[80.0, 120.0, 160.0, 240.0])
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
    judge, judge_genre_to_idx = module._load_or_train_judge(Path(first_cfg.judge_ckpt), Path(first_cfg.cache_dir), suite_dir, device, int(first_cfg.max_frames))
    hybrid_cfg = module.HybridPushConfig()
    song_map = {module._slug(Path(song["path"]).stem): song for song in module.picked_songs()}
    genre_to_idx = judge_genre_to_idx

    candidate_summaries: Dict[str, Any] = {}
    best_rows: List[Dict[str, Any]] = []
    best_run_rows: List[Dict[str, Any]] = []
    best_score = -1e18
    best_skip = None

    for skip_ms in args.skip_ms:
        rows: List[Dict[str, Any]] = []
        per_target: List[Dict[str, Any]] = []
        separation_vals: List[float] = []
        temp_rows_by_target: Dict[str, List[Dict[str, Any]]] = {}
        for tr in suite_summary["target_runs"]:
            run_dir = Path(tr["run_dir"])
            run_cfg = module.TrainConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
            target = str(tr["target"])
            temp_rows_by_target[target] = []
            final_pack = Path(str(tr["summary"]["final_pack_dir"]))
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
                seam_positions = [int(chunk["start_sample"] + len(chunk["audio"])) for chunk in chunks[:-1]]
                repaired = _bridge_skip_region(gen, seam_positions, module.DIFFUSION_SR, skip_ms=float(skip_ms))
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
                    "accompaniment": repaired,
                    "judge_probs": probs.tolist(),
                }
                rows.append(out_row)
                temp_rows_by_target[target].append(out_row)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                pa = np.asarray(rows[i]["judge_probs"], dtype=np.float32)
                pb = np.asarray(rows[j]["judge_probs"], dtype=np.float32)
                separation_vals.append(float(np.mean(np.abs(pa - pb))))
        mean_sep = float(np.mean(separation_vals)) if separation_vals else 0.0
        final_rows: List[Dict[str, Any]] = []
        for target, target_rows in temp_rows_by_target.items():
            summarized = []
            for row in target_rows:
                out_row = dict(row)
                out_row["separation"] = mean_sep
                out_row["overall"] = float(
                    0.32 * out_row["target_margin"] +
                    0.18 * out_row["target_conf"] +
                    0.28 * out_row["fullness"] +
                    0.22 * out_row["structure"] +
                    0.20 * out_row["separation"] -
                    0.20 * out_row["warble"]
                )
                summarized.append(out_row)
                final_rows.append(out_row)
            per_target.append({"target": target, "rows": summarized})
        summary = _summarize_rows(final_rows)
        candidate_summaries[str(skip_ms)] = summary
        score = float(summary["mean_overall"] - 0.5 * summary["mean_warble"] + 0.15 * summary["mean_structure"])
        if score > best_score:
            best_score = score
            best_skip = float(skip_ms)
            best_rows = final_rows
            best_run_rows = per_target

    if best_skip is None:
        raise RuntimeError("No seam-skip candidate produced rows.")

    combined_dir = suite_dir / "combined_pack_seamskip"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    target_runs_out: List[Dict[str, Any]] = []
    for tr in suite_summary["target_runs"]:
        run_dir = Path(tr["run_dir"])
        target = str(tr["target"])
        out_dir = run_dir / "final_pack_seamskip"
        out_dir.mkdir(parents=True, exist_ok=True)
        target_rows = next(item["rows"] for item in best_run_rows if item["target"] == target)
        saved_rows: List[Dict[str, Any]] = []
        for row in target_rows:
            song_key = str(row["song"])
            song = song_map[song_key]
            stems = module._resolve_stems(hybrid_cfg, song)
            render_dir = out_dir / "renders" / song_key / target
            render_dir.mkdir(parents=True, exist_ok=True)
            accomp_path = render_dir / "accompaniment_generated.wav"
            sf.write(str(accomp_path), np.asarray(row["accompaniment"], dtype=np.float32), module.DIFFUSION_SR)
            mix_path = module._mix_preserved_vocals(stems["vocals"], np.asarray(row["accompaniment"], dtype=np.float32), render_dir, vocal_gain=0.95, accomp_gain=1.0)
            out_row = dict(row)
            out_row["hybrid_wav"] = str(mix_path)
            out_row["accompaniment_wav"] = str(accomp_path)
            out_row.pop("accompaniment", None)
            saved_rows.append(out_row)
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
        target_runs_out.append({"target": target, "run_dir": str(run_dir), "out_dir": str(out_dir), "summary": {"rows": saved_rows}})

    final_summary = _summarize_rows([dict(r, accompaniment=None) for r in best_rows])
    final_summary.update(
        {
            "suite_dir": str(suite_dir),
            "combined_pack_seamskip": str(combined_dir),
            "target_runs": target_runs_out,
            "selected_skip_ms": best_skip,
            "candidate_summaries": candidate_summaries,
        }
    )
    (suite_dir / "seamskip_rerender_summary.json").write_text(json.dumps(final_summary, indent=2, default=_json_default), encoding="utf-8")
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
        compare[key] = {"original": val, "seamskip": float(final_summary[key]), "delta": float(final_summary[key]) - float(val)}
    (suite_dir / "seamskip_vs_original.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
