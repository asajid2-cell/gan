from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _pid_exists_windows(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    except Exception:
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait-pid", type=int, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stdout-log", type=Path, required=True)
    p.add_argument("--stderr-log", type=Path, required=True)
    args = p.parse_args()

    while _pid_exists_windows(int(args.wait_pid)):
        time.sleep(10.0)

    repo_root = Path(args.repo_root)
    stdout_log = Path(args.stdout_log)
    stderr_log = Path(args.stderr_log)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(repo_root / "lab 3.1" / "scripts" / "diffusion_longform_retool_train.py"),
        "--cache-dir",
        "saves2\\lab3_diffusion\\run_d001\\cache",
        "--out-dir",
        str(Path(args.out_dir)),
        "--bootstrap-checkpoint",
        "saves2\\lab3_diffusion\\run_d002\\checkpoints\\epoch_006.pt",
        "--epochs",
        "6",
        "--batch-size",
        "1",
        "--grad-accum",
        "1",
        "--max-frames",
        "432",
        "--lr",
        "5e-5",
        "--cfg-dropout-p",
        "0.08",
        "--identity-weight",
        "1.0",
        "--style-weight",
        "1.6",
        "--anchor-weight",
        "0.40",
        "--envelope-weight",
        "0.20",
        "--continuity-weight",
        "0.80",
        "--hf-penalty-weight",
        "0.18",
        "--vocal-weight",
        "0.50",
        "--crackle-weight",
        "0.45",
        "--anchor-bins",
        "40",
        "--hf-start-bin",
        "56",
        "--vocal-start-bin",
        "10",
        "--vocal-end-bin",
        "42",
        "--overlap-frames",
        "64",
        "--hf-margin",
        "0.04",
        "--crackle-margin",
        "0.008",
        "--style-probe-frames",
        "256",
        "--style-every-steps",
        "2",
        "--style-batch-splits",
        "1",
        "--max-batches-per-epoch",
        "3500",
        "--monitor-steps",
        "25",
        "--save-every-steps",
        "100",
        "--epoch-train-samples",
        "3",
        "--epoch-download-samples",
        "4",
        "--epoch-sample-ddim-steps",
        "50",
        "--epoch-sample-t-start",
        "230",
        "--epoch-sample-guidance-scale",
        "1.8",
        "--epoch-sample-style-strength",
        "0.55",
        "--source-aug-prob",
        "0.80",
        "--source-noise-std",
        "0.020",
        "--source-cond-noise-std",
        "0.015",
        "--source-global-offset-std",
        "0.060",
        "--source-hf-tilt-std",
        "0.085",
        "--source-time-mask-prob",
        "0.35",
        "--source-time-mask-frames",
        "28",
        "--resume",
        "--device",
        "auto",
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONIOENCODING"] = "utf-8"

    with stdout_log.open("ab") as out_f, stderr_log.open("ab") as err_f:
        subprocess.Popen(cmd, cwd=str(repo_root), stdout=out_f, stderr=err_f, env=env)


if __name__ == "__main__":
    main()
