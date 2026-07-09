from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _tail_lines(path: Path, max_lines: int = 80) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()[-max_lines:]


def _proc_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    cmd = ["tasklist", "/FI", f"PID eq {int(pid)}"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return str(pid) in (out.stdout or "")


def _latest_progress_line(stdout_log: Path) -> str:
    for line in reversed(_tail_lines(stdout_log, 120)):
        s = line.strip()
        if s.startswith("[epoch ") or s.startswith("epoch=") or "resumed_from=" in s:
            return s
    return ""


def _latest_err_line(stderr_log: Path) -> str:
    for line in reversed(_tail_lines(stderr_log, 80)):
        s = line.strip()
        if s:
            return s
    return ""


def _history_epochs(run_dir: Path) -> int:
    history_path = run_dir / "v2_history.json"
    if not history_path.exists():
        return 0
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _epoch_sample_count(run_dir: Path) -> int:
    root = run_dir / "epoch_samples"
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.wav"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Background monitor for a training run.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--stdout-log", type=Path, required=True)
    ap.add_argument("--stderr-log", type=Path, required=True)
    ap.add_argument("--pid", type=int, default=0)
    ap.add_argument("--interval-sec", type=int, default=1500)
    ap.add_argument("--monitor-log", type=Path, required=True)
    args = ap.parse_args()

    args.monitor_log.parent.mkdir(parents=True, exist_ok=True)
    last_progress = ""
    last_err = ""

    while True:
        alive = _proc_alive(int(args.pid)) if int(args.pid) > 0 else False
        progress = _latest_progress_line(args.stdout_log)
        err = _latest_err_line(args.stderr_log)
        epochs_done = _history_epochs(args.run_dir)
        sample_wavs = _epoch_sample_count(args.run_dir)
        summary_exists = (args.run_dir / "summary.json").exists()

        payload = {
            "ts": _now(),
            "alive": alive,
            "epochs_done": epochs_done,
            "sample_wavs": sample_wavs,
            "summary_exists": summary_exists,
            "progress": progress,
            "stderr": err,
        }
        with args.monitor_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        if progress and progress != last_progress:
            last_progress = progress
        if err and err != last_err:
            last_err = err

        if not alive:
            break
        time.sleep(max(30, int(args.interval_sec)))


if __name__ == "__main__":
    main()
