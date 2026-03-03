# Quickstart: Long-Form Genre Remastering

Goal: run end-to-end long-form generation to evaluate coherence and style transfer qualitatively.

## Prereqs

- Python environment with the repo dependencies (see `requirements.txt`).
- Existing diffusion cache and checkpoint (recommended defaults in this repo):
  - Cache: `saves2/lab3_diffusion/run_d001/cache`
  - Checkpoint: `saves2/lab3_diffusion/run_d002/checkpoints/epoch_006.pt` (chosen for best perceived quality)

## Run

From repo root:

```powershell
python "lab 3/run_lab4_longform_coherence.py" `
  --cache-dir "saves2/lab3_diffusion/run_d001/cache" `
  --checkpoint "saves2/lab3_diffusion/run_d002/checkpoints/epoch_006.pt" `
  --out-dir "saves2/lab4_longform_coherence/quickstart" `
  --input "PATH_TO_AUDIO_FILE"
```

Outputs:
- `longform_coherent.wav`
- per-chunk wav exports
- `coherence_metrics.json` (boundary diagnostics)

## Knobs (most important)

- `--t-start` / `--t-start-end`: SDEdit-style source anchoring strength (higher = more edit freedom).
- `--prefix-blend`: overlap locking strength (higher = more continuity).
- `--reanchor-every` / `--reanchor-t-start`: periodically reset drift.
- `--style-strength`: mix between source style and target style embedding.
- `--source-mel-blend` / `--hf-source-blend`: anti-warble stabilization (especially for long tracks).

