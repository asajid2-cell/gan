from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREV_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "rerender_block_suite_prevcontext_pack.py"
ADAPTIVE_SCRIPT = REPO_ROOT / "lab 3.4" / "scripts" / "repair_block_suite_adaptive_local_skip.py"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the current best seam-first block-suite pack.")
    ap.add_argument("--suite-dir", type=Path, required=True)
    args = ap.parse_args()

    suite_dir = Path(args.suite_dir)
    _run(
        [
            sys.executable,
            str(PREV_SCRIPT),
            "--suite-dir",
            str(suite_dir),
            "--prev-mode",
            "blend35",
        ]
    )
    _run(
        [
            sys.executable,
            str(ADAPTIVE_SCRIPT),
            "--suite-dir",
            str(suite_dir),
            "--input-pack",
            str(suite_dir / "combined_pack_prev_blend35"),
            "--candidates-ms",
            "0",
            "80",
            "120",
            "160",
            "240",
        ]
    )


if __name__ == "__main__":
    main()
