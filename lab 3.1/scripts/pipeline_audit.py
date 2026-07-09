from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def summarize_codec_runs(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or (REPO_ROOT / "saves2" / "lab3_codec_transfer"))
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.glob("run*")):
        state = _maybe_load_json(run_dir / "run_state.json")
        if not state:
            continue
        cfg = state.get("config") or {}
        gate = ((state.get("codec_gate_eval") or {}).get("metrics") or {})
        best = (_maybe_load_json(run_dir / "realism_supervisor" / "codec_realism_best.json").get("best") or {})
        rows.append(
            {
                "run": run_dir.name,
                "current_stage": state.get("current_stage"),
                "style_cond_source": cfg.get("style_cond_source"),
                "style_loss_mode": cfg.get("style_loss_mode"),
                "translator_direct_output": cfg.get("translator_direct_output"),
                "translator_direct_mix": cfg.get("translator_direct_mix"),
                "stage2_style_weight": cfg.get("stage2_style_weight"),
                "stage3_style_weight": cfg.get("stage3_style_weight"),
                "stage2_adv_weight": cfg.get("stage2_adv_weight"),
                "stage3_adv_weight": cfg.get("stage3_adv_weight"),
                "stage2_latent_l1_weight": cfg.get("stage2_latent_l1_weight"),
                "stage3_latent_l1_weight": cfg.get("stage3_latent_l1_weight"),
                "stage2_mrstft_weight": cfg.get("stage2_mrstft_weight"),
                "stage3_mrstft_weight": cfg.get("stage3_mrstft_weight"),
                "stage2_delta_budget_weight": cfg.get("stage2_delta_budget_weight"),
                "stage3_delta_budget_weight": cfg.get("stage3_delta_budget_weight"),
                "gate_mps": gate.get("mps"),
                "gate_style_conf": gate.get("style_conf"),
                "gate_style_acc": gate.get("style_acc"),
                "gate_pairwise_cos": gate.get("pairwise_cos"),
                "realism_stage": best.get("stage"),
                "realism_fad_mert": best.get("fad_mert"),
                "realism_style_target_acc": best.get("style_target_acc"),
                "realism_style_target_cos": best.get("style_target_cos"),
                "realism_target_hf_mae": best.get("target_hf_mae"),
                "realism_target_dynamic_range_mae_db": best.get("target_dynamic_range_mae_db"),
                "realism_score": best.get("realism_score"),
            }
        )
    return pd.DataFrame(rows)


def summarize_diffusion_runs(root: Path | None = None) -> pd.DataFrame:
    root = Path(root or (REPO_ROOT / "saves2" / "lab3_diffusion"))
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(root.glob("run_*")):
        state = _maybe_load_json(run_dir / "run_state.json")
        best = (_maybe_load_json(run_dir / "realism_supervisor" / "diffusion_realism_best.json").get("best") or {})
        rows.append(
            {
                "run": run_dir.name,
                "current_stage": state.get("current_stage"),
                "best_ckpt": state.get("best_ckpt"),
                "best_val_loss": state.get("best_val_loss"),
                "realism_checkpoint": best.get("checkpoint"),
                "realism_fad_mert": best.get("fad_mert"),
                "realism_style_target_acc": best.get("style_target_acc"),
                "realism_style_target_cos": best.get("style_target_cos"),
                "realism_target_hf_mae": best.get("target_hf_mae"),
                "realism_target_dynamic_range_mae_db": best.get("target_dynamic_range_mae_db"),
                "realism_score": best.get("realism_score"),
            }
        )
    return pd.DataFrame(rows)


def summarize_lab_checkpoints() -> Dict[str, Any]:
    results_path = REPO_ROOT / "docs" / "explanation" / "results.md"
    return {
        "lab1_checkpoint": str(REPO_ROOT / "saves" / "lab1_run_combo_af_gate_exit_v2" / "latest.pt"),
        "lab2_summary": str(REPO_ROOT / "saves" / "lab2_calibration" / "lab2_20260211_015118_lda_cleanup_v2" / "validation_summary.json"),
        "lab3_results_doc": str(results_path),
        "lab4_metrics": str(REPO_ROOT / "saves2" / "lab4_longform_coherence" / "fullsong_test" / "coherence_metrics.json"),
    }


def build_diagnosis(codec_df: pd.DataFrame, diffusion_df: pd.DataFrame) -> Dict[str, Any]:
    diagnosis: Dict[str, Any] = {
        "lab1_lab2_status": "healthy",
        "codec_branch_status": "misaligned",
        "diffusion_branch_status": "promising_but_unstable",
        "primary_bottleneck": "lab3_objective_and_architecture_mismatch",
        "notes": [],
    }

    if not codec_df.empty:
        best_gate = codec_df.sort_values("gate_style_acc", ascending=False, na_position="last").iloc[0].to_dict()
        best_real = codec_df.sort_values("realism_fad_mert", ascending=True, na_position="last").iloc[0].to_dict()
        diagnosis["best_codec_gate_run"] = best_gate["run"]
        diagnosis["best_codec_realism_run"] = best_real["run"]
        diagnosis["notes"].append(
            "Codec internal gate and realism supervisor disagree: the runs with the highest internal style metrics do not produce strong target-style realism on generated audio."
        )
        diagnosis["notes"].append(
            "The best codec realism runs still cluster around chance-level target-style accuracy in the realism supervisor, which suggests the codec branch is learning shallow edits rather than convincing remasters."
        )
    if not diffusion_df.empty:
        best_diff = diffusion_df.sort_values("realism_fad_mert", ascending=True, na_position="last").iloc[0].to_dict()
        diagnosis["best_diffusion_realism_run"] = best_diff["run"]
        diagnosis["notes"].append(
            "The diffusion branch reaches stronger target-style realism than the codec branch, but long-form drift and checkpoint sensitivity remain unresolved."
        )

    diagnosis["notes"].extend(
        [
            "The current label space is mostly dataset-bucket genre, which is easier to classify than to render as a convincing musical remaster.",
            "Lab 4 is downstream of Lab 3 quality; long-form heuristics cannot fix a short-form generator that never truly enters the target style manifold.",
            "A clean-slate rerun should be notebook-driven and hypothesis-driven: source-balanced data, generated-audio style supervision, and a diffusion-first comparison path.",
        ]
    )
    return diagnosis


def write_audit(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    codec_df = summarize_codec_runs()
    diffusion_df = summarize_diffusion_runs()
    codec_csv = output_dir / "codec_runs.csv"
    diffusion_csv = output_dir / "diffusion_runs.csv"
    codec_df.to_csv(codec_csv, index=False)
    diffusion_df.to_csv(diffusion_csv, index=False)
    diagnosis = build_diagnosis(codec_df=codec_df, diffusion_df=diffusion_df)
    diagnosis["lab_paths"] = summarize_lab_checkpoints()
    diagnosis_json = output_dir / "diagnosis.json"
    diagnosis_json.write_text(json.dumps(diagnosis, indent=2), encoding="utf-8")
    return {
        "codec_csv": codec_csv,
        "diffusion_csv": diffusion_csv,
        "diagnosis_json": diagnosis_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DGGR Lab 1-4 run artifacts for notebook-first pipeline audit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "lab 3.1" / "outputs" / "audit",
    )
    args = parser.parse_args()
    paths = write_audit(output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
