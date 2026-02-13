from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "saves2" / "lab3_synthesis"


def _default_panel_root() -> Path:
    return Path(__file__).resolve().parent.parent / "saves2" / "lab3_synthesis" / "human_panel"


def _load_summary(run_dir: Path, tag: str) -> pd.DataFrame:
    p = run_dir / "samples" / tag / "generation_summary.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing generation summary: {p}")
    df = pd.read_csv(p)
    required = ["sample_id", "source_genre", "target_genre", "real_wav", "fake_wav", "source_path"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"{p} missing required columns: {miss}")
    return df


def _resolve_audio_path(run_dir: Path, p: str) -> str:
    q = Path(str(p))
    if q.is_absolute():
        return str(q)
    return str((run_dir / q).resolve())


def _build_balanced_panel(df: pd.DataFrame, n_panel: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out_parts: List[pd.DataFrame] = []
    n_genres = max(1, int(df["target_genre"].nunique()))
    per_target = max(1, int(np.floor(float(n_panel) / float(n_genres))))

    for tg, gdf in df.groupby("target_genre", sort=True):
        g = gdf.copy()
        if len(g) <= per_target:
            out_parts.append(g)
            continue
        pick = rng.choice(len(g), size=per_target, replace=False)
        out_parts.append(g.iloc[np.sort(pick)].copy())

    panel = pd.concat(out_parts, ignore_index=True) if out_parts else df.head(0).copy()
    # Top up/down to exact n_panel.
    if len(panel) < n_panel:
        remain = df.loc[~df["sample_id"].isin(panel["sample_id"].tolist())].copy()
        need = min(int(n_panel - len(panel)), len(remain))
        if need > 0:
            pick = rng.choice(len(remain), size=need, replace=False)
            panel = pd.concat([panel, remain.iloc[np.sort(pick)].copy()], ignore_index=True)
    elif len(panel) > n_panel:
        pick = rng.choice(len(panel), size=int(n_panel), replace=False)
        panel = panel.iloc[np.sort(pick)].copy().reset_index(drop=True)

    panel = panel.sort_values(["target_genre", "source_genre", "sample_id"]).reset_index(drop=True)
    panel["panel_id"] = np.arange(len(panel), dtype=np.int64)
    return panel


def cmd_prepare(args: argparse.Namespace) -> None:
    runs_root = Path(args.runs_root)
    panel_root = Path(args.panel_root)
    panel_root.mkdir(parents=True, exist_ok=True)
    out_dir = panel_root / str(args.panel_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_run = str(args.baseline_run)
    compare_runs = [r.strip() for r in str(args.compare_runs).split(",") if r.strip()]
    if baseline_run not in compare_runs:
        compare_runs = [baseline_run] + compare_runs

    base_dir = runs_root / baseline_run
    base_df = _load_summary(base_dir, tag=str(args.sample_tag))
    panel = _build_balanced_panel(base_df, n_panel=int(args.panel_size), seed=int(args.seed))
    panel_core = panel[
        ["panel_id", "sample_id", "cache_row", "source_path", "source_genre", "target_genre", "mps_cosine", "style_conf_target"]
    ].copy()
    panel_core.to_csv(out_dir / "panel_spec.csv", index=False)

    listens: List[pd.DataFrame] = []
    missing_runs: Dict[str, str] = {}
    mismatch_runs: Dict[str, int] = {}

    for run_name in compare_runs:
        run_dir = runs_root / run_name
        try:
            rdf = _load_summary(run_dir, tag=str(args.sample_tag))
        except Exception as e:
            missing_runs[run_name] = str(e)
            continue
        merged = panel_core.merge(
            rdf[["sample_id", "source_genre", "target_genre", "real_wav", "fake_wav"]],
            on=["sample_id", "source_genre", "target_genre"],
            how="left",
        )
        n_miss = int(merged["fake_wav"].isna().sum())
        if n_miss > 0:
            mismatch_runs[run_name] = n_miss
        merged["run_name"] = run_name
        merged["real_wav_abs"] = merged["real_wav"].map(lambda p: _resolve_audio_path(run_dir, p) if isinstance(p, str) else "")
        merged["fake_wav_abs"] = merged["fake_wav"].map(lambda p: _resolve_audio_path(run_dir, p) if isinstance(p, str) else "")
        listens.append(merged)

    if not listens:
        raise RuntimeError("No runs were loaded. Check runs_root/run names/sample_tag.")
    listen_df = pd.concat(listens, ignore_index=True)
    listen_df.to_csv(out_dir / "listen_manifest.csv", index=False)

    rating_df = listen_df[
        ["panel_id", "run_name", "sample_id", "source_genre", "target_genre", "source_path", "real_wav_abs", "fake_wav_abs"]
    ].copy()
    rating_df["melody_clear_1to5"] = np.nan
    rating_df["style_convincing_1to5"] = np.nan
    rating_df["artifact_severity_1to5"] = np.nan  # 1=clean, 5=very artifacty
    rating_df["overall_quality_1to5"] = np.nan
    rating_df["notes"] = ""
    rating_df.to_csv(out_dir / "ratings_template.csv", index=False)

    meta = {
        "baseline_run": baseline_run,
        "compare_runs": compare_runs,
        "sample_tag": str(args.sample_tag),
        "panel_size": int(len(panel_core)),
        "seed": int(args.seed),
        "missing_runs": missing_runs,
        "mismatch_runs": mismatch_runs,
    }
    with (out_dir / "panel_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[panel] out_dir={out_dir}")
    print(f"[panel] panel_size={len(panel_core)}")
    print(f"[panel] runs_loaded={sorted(list(set(listen_df['run_name'].tolist())))}")
    if missing_runs:
        print(f"[panel] missing_runs={missing_runs}")
    if mismatch_runs:
        print(f"[panel] mismatched sample rows per run={mismatch_runs}")


def cmd_summarize(args: argparse.Namespace) -> None:
    panel_dir = Path(args.panel_dir)
    ratings_path = panel_dir / "ratings_template.csv"
    if args.ratings_csv:
        ratings_path = Path(args.ratings_csv)
    if not ratings_path.exists():
        raise FileNotFoundError(f"Ratings CSV not found: {ratings_path}")

    df = pd.read_csv(ratings_path)
    for c in ["melody_clear_1to5", "style_convincing_1to5", "artifact_severity_1to5", "overall_quality_1to5"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert artifact severity to a "higher is better" cleanliness score.
    if "artifact_severity_1to5" in df.columns:
        df["artifact_cleanliness_1to5"] = 6.0 - df["artifact_severity_1to5"]

    group_cols = ["run_name"]
    run_summary = (
        df.groupby(group_cols, as_index=False)[
            ["melody_clear_1to5", "style_convincing_1to5", "artifact_cleanliness_1to5", "overall_quality_1to5"]
        ]
        .mean()
        .rename(
            columns={
                "melody_clear_1to5": "melody_mean",
                "style_convincing_1to5": "style_mean",
                "artifact_cleanliness_1to5": "cleanliness_mean",
                "overall_quality_1to5": "overall_mean",
            }
        )
    )

    # Optional merge with run audit metrics for quick side-by-side.
    if bool(args.attach_audit_metrics):
        rows = []
        for rn in run_summary["run_name"].tolist():
            ap = Path(args.runs_root) / str(rn) / "lab3_exit_audit.json"
            if ap.exists():
                with ap.open("r", encoding="utf-8") as f:
                    a = json.load(f)
                rows.append(
                    {
                        "run_name": rn,
                        "mps": a.get("mps"),
                        "style_fidelity_conf": a.get("style_fidelity_conf"),
                        "style_acc": a.get("style_acc"),
                        "spectral_continuity": a.get("spectral_continuity"),
                        "fake_pairwise_feature_cos": a.get("fake_pairwise_feature_cos"),
                    }
                )
        if rows:
            run_summary = run_summary.merge(pd.DataFrame(rows), on="run_name", how="left")

    out_csv = panel_dir / "ratings_summary.csv"
    run_summary.to_csv(out_csv, index=False)
    print(f"[panel] summary={out_csv}")
    print(run_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab3 fixed human listening panel tooling")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_prepare = sub.add_parser("prepare", help="Create fixed panel + listening/rating templates")
    p_prepare.add_argument("--runs-root", type=Path, default=_default_runs_root())
    p_prepare.add_argument("--panel-root", type=Path, default=_default_panel_root())
    p_prepare.add_argument("--panel-name", type=str, default="panel1")
    p_prepare.add_argument("--baseline-run", type=str, default="run20")
    p_prepare.add_argument("--compare-runs", type=str, default="run20,run28,run29")
    p_prepare.add_argument("--sample-tag", type=str, default="posttrain_samples")
    p_prepare.add_argument("--panel-size", type=int, default=40)
    p_prepare.add_argument("--seed", type=int, default=328)
    p_prepare.set_defaults(func=cmd_prepare)

    p_sum = sub.add_parser("summarize", help="Summarize filled ratings CSV")
    p_sum.add_argument("--panel-dir", type=Path, required=True)
    p_sum.add_argument("--ratings-csv", type=Path, default=None)
    p_sum.add_argument("--attach-audit-metrics", action=argparse.BooleanOptionalAction, default=True)
    p_sum.add_argument("--runs-root", type=Path, default=_default_runs_root())
    p_sum.set_defaults(func=cmd_summarize)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

