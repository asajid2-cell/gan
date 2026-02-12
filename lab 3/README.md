# Lab 3 - Reconstruction Decoder (Analysis -> Synthesis)

Lab 3 turns the frozen analysis stack into a generative synthesis stack.

## Objective

Build a conditional Reconstruction Decoder that takes:

- `z_content` (Lab 1 structural skeleton, 128-D)
- `V_target` (Lab 2 genre target vector)

and synthesizes a Log-Mel spectrogram that preserves melody while shifting style.

## Two-Stage Training

1. Stage 1 - Self-Reconstruction Baseline
- Condition on the sample's own genre target vector.
- Reconstruct the original Log-Mel target.
- Validates decoder capacity and conditioning path.

2. Stage 2 - Genre-Shift Synthesis
- Condition on a different genre target vector.
- Supports conditioning modes: `centroid`, `exemplar`, `mix` (default `mix`).
- Preserve `z_content` while steering style toward target genre.
- Uses content/style consistency + adversarial + spectral continuity regularization.

## Data Generalization Controls

- Multi-chunk sampling per track during cache build:
  - `--chunks-per-track` (default `4`)
  - `--chunk-sampling` (`uniform` or `random`)
  - `--min-start-sec`, `--max-start-sec`
- Grouped split to avoid track leakage:
  - `--split-by-track` (default on)
  - uses `track_id` in cache index to ensure train/val track disjointness.

## Exit Metrics

- `MPS` (Melodic Preservation Score): cosine(`z_content`, `z_content'`) >= `0.90`
- `SF` (Stylistic Fidelity): classifier confidence in target genre >= `0.85`
- `Spectral Continuity`: multi-resolution STFT continuity score reported (lower is better)

## Save/Resume

Run artifacts are written to:

`../saves2/lab3_synthesis/runN/`

with:

- `run_state.json`
- `checkpoints/stage1_latest.pt`
- `checkpoints/stage2_latest.pt`
- `history.csv`
- `lab3_exit_audit.json`

Resume by passing `--mode resume --resume-dir <run_dir>`.

By default, strict run naming is enforced (`run1`, `run2`, ...), and each completed run
auto-exports a standardized post-train sample pack to:

`../saves2/lab3_synthesis/runN/samples/posttrain_samples/`

## Quick Start

```powershell
cd "lab 3"
python run_lab3.py --smoke
```

Notebook runner:

`lab 3/lab3_reconstruction_decoder.ipynb`

Full run example:

```powershell
cd "lab 3"
python run_lab3.py `
  --per-genre-samples 800 `
  --stage1-epochs 20 `
  --stage2-epochs 20
```
