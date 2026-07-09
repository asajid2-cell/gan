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
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "train_scratch_retrieval_body_style_block_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("blockseq", str(BLOCK_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported value: {type(value)!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Rerender an existing block suite using the current seam-blended long-form assembler.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    summary_path = suite_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    suite_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    module = _load_module()

    first_run = Path(suite_summary["target_runs"][0]["run_dir"])
    first_cfg = json.loads((first_run / "config.json").read_text(encoding="utf-8"))
    cfg = module.TrainConfig(**first_cfg)
    device = module._device_from_arg(str(cfg.device))
    index_df, arrays, genre_to_idx, meta = module.load_diffusion_cache(Path(cfg.cache_dir), mmap=True)
    genre_idx = np.asarray(arrays["genre_idx"], dtype=np.int64)
    group_ids = index_df["track_id"].astype(str).to_numpy()
    vocoder = module.load_bigvgan_robust(device=device)

    target_runs_out: List[Dict[str, Any]] = []
    combined_dir = suite_dir / "combined_pack_seamblend"
    combined_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = combined_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
        writer.writeheader()

    combined_rows: List[Dict[str, Any]] = []

    for tr in suite_summary["target_runs"]:
        run_dir = Path(tr["run_dir"])
        run_cfg = module.TrainConfig(**json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        keep = int(genre_to_idx[str(run_cfg.single_genre_target)])
        train_idx, _ = module.stratified_group_split_indices(genre_idx, group_ids, val_ratio=0.15, seed=int(run_cfg.seed))
        train_idx = train_idx[genre_idx[train_idx] == keep]
        track_bank = module._build_track_bank(index_df, arrays, train_idx)
        judge, judge_genre_to_idx = module._load_or_train_judge(Path(run_cfg.judge_ckpt), Path(run_cfg.cache_dir), run_dir, device, int(run_cfg.max_frames))
        if set(judge_genre_to_idx.keys()) != set(genre_to_idx.keys()):
            raise RuntimeError("Judge genre mismatch during rerender.")
        model = module.RetrievalFusionUNet(in_ch=18, num_genres=len(genre_to_idx), base_ch=int(run_cfg.base_ch)).to(device)
        payload = torch.load(str(Path(tr["summary"]["best_checkpoint"])), map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        out_dir = run_dir / "final_pack_seamblend"
        rerender_summary = module.benchmark_checkpoint(
            model=model,
            judge=judge,
            genre_to_idx=genre_to_idx,
            track_bank=track_bank,
            arrays=arrays,
            mel_min=float(meta.mel_min),
            mel_max=float(meta.mel_max),
            max_frames=int(run_cfg.max_frames),
            vocoder=vocoder,
            device=device,
            seconds=float(run_cfg.final_seconds),
            out_dir=out_dir,
            single_genre_target=str(run_cfg.single_genre_target),
        )
        target_runs_out.append(
            {
                "target": str(tr["target"]),
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "summary": rerender_summary,
            }
        )
        target = str(tr["target"])
        for row in rerender_summary["rows"]:
            song = str(row["song"])
            src_dir = out_dir / "renders" / song / target
            dst_dir = combined_dir / song / target
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in ["hybrid_longform_coherent.wav", "accompaniment_generated.wav", "longform_coherent.wav", "backing_fixed.wav"]:
                src = src_dir / name
                if src.exists():
                    shutil.copyfile(src, dst_dir / name)
            with manifest_path.open("a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["target", "source_song", "source_target_dir", "hybrid_wav", "accompaniment_wav"])
                writer.writerow(
                    {
                        "target": target,
                        "source_song": song,
                        "source_target_dir": str(src_dir),
                        "hybrid_wav": str(dst_dir / "hybrid_longform_coherent.wav"),
                        "accompaniment_wav": str(dst_dir / "accompaniment_generated.wav"),
                    }
                )
            combined_rows.append(dict(row))

    summary = {
        "suite_dir": str(suite_dir),
        "combined_pack_seamblend": str(combined_dir),
        "target_runs": target_runs_out,
        "mean_overall": float(sum(float(r["overall"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_target_conf": float(sum(float(r["target_conf"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_target_margin": float(sum(float(r["target_margin"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_warble": float(sum(float(r["warble"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_fullness": float(sum(float(r["fullness"]) for r in combined_rows) / max(1, len(combined_rows))),
        "mean_structure": float(sum(float(r["structure"]) for r in combined_rows) / max(1, len(combined_rows))),
    }
    (suite_dir / "seamblend_rerender_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
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
        compare[key] = {"original": val, "seamblend": float(summary[key]), "delta": float(summary[key]) - float(val)}
    (suite_dir / "seamblend_vs_original.json").write_text(json.dumps(compare, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
