# CLI Reference

This page lists the main runnable scripts in the repo and what they do.

Conventions:
- Run commands from the repo root unless noted.
- Large artifacts are not committed (`saves/`, `saves2/` are gitignored).
- Prefer setting `DGGR_MANIFESTS_ROOT` and `DGGR_DATA_ROOT` for portability (see `docs/reference/env_vars.md`).

## Lab 2

### `lab 2/run_lab2.py`

Purpose:
- Harvest Lab 1 embeddings (`z_content`, `z_style`) for curated manifests.
- Build 160D target vectors and validate target-space separability.

Key flags:
- `--checkpoint`: frozen Lab 1 checkpoint (`saves/.../latest.pt`).
- `--manifests-root`: path to cleaned manifests (defaults via `DGGR_MANIFESTS_ROOT` fallback).
- `--manifest-files`: list of CSVs to include.
- `--per-genre-samples`: sample cap per genre.
- `--output-dir`: where to write artifacts (`saves/lab2_calibration/...`).

Outputs:
- `validation_summary.json`
- centroid CSVs / JSON exports (depends on run mode)

## Lab 3 (codec-latent transfer)

### `lab 3/run_lab3_codec.py`

Purpose:
- Style transfer by translating EnCodec embeddings `q_src -> q_hat`.
- Content preservation enforced via Lab 1 `z_content` cosine similarity.
- Style control driven by a conditioning embedding (Lab 1 / codec judge / MERT probe).

Key flags:
- `--style-cond-source`: where the style embedding comes from.
  - `mert_probe_embed` is the best-performing conditioning in current results.
- `--style-loss-mode`: how style loss is computed.
- `--translator-direct-output`: removes the residual leash; enabled in the best run (`run1055`).
- `--gate-multi-pass`: apply translator multiple times at eval for stronger style shift (optional).

Outputs:
- `codec_gate_eval.json` (MPS, style confidence, style accuracy, collapse proxy)
- optional exported sample WAVs

## Lab 3 (diffusion)

### `lab 3/run_lab3_diffusion_v2.py`

Purpose:
- Train diffusion V2 (v-prediction UNet) in mel space with EMA + CFG dropout.

Key flags:
- `--cache-dir`: diffusion cache (mel/chroma/onset + z_content/z_style).
- `--out-dir`: run folder for checkpoints + history.
- `--epochs`, `--lr`, `--ema-decay`, `--cfg-dropout-p`

Outputs:
- `v2_config.json`, `v2_history.json`
- `checkpoints/epoch_*.pt`
- `epoch_samples/` (if enabled)

### `lab 3/run_lab3_diffusion_v3.py`

Purpose:
- Fine-tune diffusion V2 with a mel discriminator (hinge GAN + feature matching).

Notes:
- In current runs, V3 did not beat V2 on validation, but the scaffold exists for future tuning.

## Lab 4 (long-form coherence)

### `lab 4/run_lab4_longform_coherence.py`

Purpose:
- Long-form chunked generation with *coherence-first* constraints:
  - SDEdit-style anchoring to the source mel at timestep `t_start`.
  - Prefix-locking overlap frames at every DDIM step to keep seams consistent.
  - Drift controls (re-anchoring, mel smoothing, HF source blend).

Key flags (highest impact):
- `--t-start`, `--t-start-end`: how far to diffuse the source before denoising (edit magnitude).
- `--prefix-blend`: overlap locking strength (cohesion).
- `--reanchor-every`, `--reanchor-t-start`: periodically reset drift.
- `--style-strength`: mix between source and target style embeddings.
- `--source-mel-blend`, `--hf-source-blend`: anti-warble stabilization.
- `--assemble-domain`: `mel` (one-shot vocoding) vs `audio` (per-chunk vocoding + crossfade).

Outputs:
- `longform_coherent.wav`
- per-chunk WAVs
- `coherence_metrics.json` (boundary diagnostics)

## Evaluation helpers

### `lab 3/quick_eval_diffusion.py`
Quickly vocodes a small number of diffusion samples at different guidance scales.

### `lab 3/quick_eval_sdedit.py` and `lab 3/quick_eval_sdedit_v2.py`
Quickly tests SDEdit-style transfer and envelope-transfer variants for timbral shift.
