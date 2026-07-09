from __future__ import annotations

import json
from pathlib import Path

from diffusion_longform_settings_sweep import DiffusionSettingsSweepConfig, run_settings_sweep
from run_bestpt_targeted_downloads_sweep import targeted_bestpt_settings_panel


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    cfg = DiffusionSettingsSweepConfig(
        tag="20260328_102330",
        output_root=REPO_ROOT / "lab 3.1" / "outputs" / "diffusion_bestpt_targeted_sweep",
        run_dir=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002",
        checkpoint_path=REPO_ROOT / "saves2" / "lab3_diffusion" / "run_d002" / "checkpoints" / "best.pt",
        n_songs=6,
        targets_per_song=1,
        source_seconds=36.0,
        chunk_seconds=3.0,
        overlap_seconds=0.5,
        n_frames=256,
        ddim_steps=50,
        assemble_domain="mel",
        device="auto",
        seed=328,
    ).materialize()
    summary = run_settings_sweep(cfg, targeted_bestpt_settings_panel())
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
