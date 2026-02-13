from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "saves2" / "lab3_clip_picker"


def _pick_path_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested path column '{requested}' not found in CSV.")
        return requested
    for c in ["fake_wav", "real_wav", "path", "source_path"]:
        if c in df.columns:
            return c
    raise ValueError("No usable path column found. Pass --path-col explicitly.")


def _resolve_path(base_dir: Path, p: str) -> Path:
    q = Path(str(p))
    if q.is_absolute():
        return q
    return (base_dir / q).resolve()


def _open_clip(path: Path) -> None:
    if not path.exists():
        print(f"[warn] missing file: {path}")
        return
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception as e:
        print(f"[warn] failed to open clip: {path} ({e})")


def _load_prior_decisions(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = str(row.get("clip_key", "")).strip()
            d = str(row.get("decision", "")).strip()
            if k:
                out[k] = d
    return out


def _append_decision(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive random clip picker for fast manual triage.")
    p.add_argument("--input-csv", type=Path, required=True, help="CSV containing clip paths.")
    p.add_argument("--path-col", type=str, default="", help="Path column; auto-detected if omitted.")
    p.add_argument("--base-dir", type=Path, default=Path("."), help="Base dir for relative paths.")
    p.add_argument("--shuffle-seed", type=int, default=328)
    p.add_argument("--max-clips", type=int, default=0, help="0 means all available.")
    p.add_argument("--output-dir", type=Path, default=_default_output_dir())
    p.add_argument("--session-name", type=str, default="session1")
    p.add_argument("--auto-open", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--filter-target-genre", type=str, default="")
    p.add_argument("--filter-source-genre", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")

    df = pd.read_csv(in_csv)
    path_col = _pick_path_column(df, args.path_col.strip() or None)
    if args.filter_target_genre and "target_genre" in df.columns:
        df = df[df["target_genre"].astype(str) == str(args.filter_target_genre)].copy()
    if args.filter_source_genre and "source_genre" in df.columns:
        df = df[df["source_genre"].astype(str) == str(args.filter_source_genre)].copy()
    if len(df) == 0:
        raise RuntimeError("No rows after filtering.")

    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=int(args.shuffle_seed)).to_numpy()
    if int(args.max_clips) > 0:
        rng = rng[: int(args.max_clips)]
    df = df.iloc[rng].reset_index(drop=True)

    out_dir = Path(args.output_dir) / str(args.session_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions_csv = out_dir / "decisions.csv"
    accepted_csv = out_dir / "accepted.csv"
    rejected_csv = out_dir / "rejected.csv"

    prior = _load_prior_decisions(decisions_csv)
    n_total = len(df)
    n_seen = 0
    n_accept = 0
    n_reject = 0
    n_skip = 0

    print(f"[picker] input={in_csv}")
    print(f"[picker] path_col={path_col} rows={n_total}")
    print(f"[picker] output={out_dir}")
    print("Controls: [a]=accept [r]=reject [s]=skip [o]=open/replay [q]=quit")

    for i, row in df.iterrows():
        clip_raw = str(row[path_col])
        clip_path = _resolve_path(Path(args.base_dir), clip_raw)
        clip_key = str(clip_path)
        if clip_key in prior:
            continue

        n_seen += 1
        header = f"\n[{n_seen}/{n_total}] {clip_path.name}"
        print(header)
        if "source_genre" in row:
            print(f"  source={row.get('source_genre')} ", end="")
        if "target_genre" in row:
            print(f"target={row.get('target_genre')} ", end="")
        if "sample_id" in row:
            print(f"sample_id={row.get('sample_id')} ", end="")
        print("")
        print(f"  path={clip_path}")

        if bool(args.auto_open):
            _open_clip(clip_path)

        while True:
            inp = input("decision [a/r/s/o/q]: ").strip().lower()
            if inp == "o":
                _open_clip(clip_path)
                continue
            if inp not in {"a", "r", "s", "q"}:
                print("  invalid input")
                continue
            if inp == "q":
                print("[picker] quitting session.")
                print(f"[picker] accepted={n_accept} rejected={n_reject} skipped={n_skip}")
                return

            decision = {"a": "accept", "r": "reject", "s": "skip"}[inp]
            ts = datetime.now().isoformat(timespec="seconds")
            out_row = {
                "timestamp": ts,
                "clip_key": clip_key,
                "decision": decision,
                "input_csv": str(in_csv),
                "path_col": path_col,
                "clip_path": str(clip_path),
            }
            # Keep helpful metadata if present.
            for c in ["sample_id", "cache_row", "source_genre", "target_genre", "source_path", "style_conf_target", "mps_cosine"]:
                if c in row.index:
                    out_row[c] = row.get(c)
            _append_decision(decisions_csv, out_row)
            prior[clip_key] = decision

            if decision == "accept":
                _append_decision(accepted_csv, out_row)
                n_accept += 1
            elif decision == "reject":
                _append_decision(rejected_csv, out_row)
                n_reject += 1
            else:
                n_skip += 1
            break

    print("\n[picker] session complete.")
    print(f"[picker] accepted={n_accept} rejected={n_reject} skipped={n_skip}")
    print(f"[picker] decisions={decisions_csv}")
    print(f"[picker] accepted={accepted_csv}")
    print(f"[picker] rejected={rejected_csv}")


if __name__ == "__main__":
    main()

