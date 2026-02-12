# Lab 3 Plan - Reconstruction Decoder

## Goal

Synthesize genre-remastered Log-Mel outputs from:

- structural content (`z_content`)
- Lab 2 target vectors (`V_target`)

while preserving melody and enforcing target style.

## Phase A - Data and Latent Cache

- Load manifests and assign Lab 2-compatible genres.
- Materialize balanced sample subset.
- Extract and cache:
  - fixed-size Log-Mel targets
  - `z_content`
  - `z_style`
  - genre labels and paths

Done criteria:
- `cache/cache_index.csv` and `cache/cache_arrays.npz` exist.

## Phase B - Stage 1 Training (Self Reconstruction)

- Train conditional decoder + discriminator.
- Condition with same-genre target vectors.
- Optimize reconstruction + adversarial + latent consistency + MR-STFT continuity.

Done criteria:
- `checkpoints/stage1_latest.pt` exists.
- Stage 1 losses trend down and output quality is stable.

## Phase C - Stage 2 Training (Genre Shift)

- Shift condition to non-matching target genre vectors.
- Preserve `z_content` and steer toward target style centroid.
- Continue adversarial + spectral continuity regularization.

Done criteria:
- `checkpoints/stage2_latest.pt` exists.

## Phase D - Exit Audit

- MPS: cosine(`z_content`, `z_content'`)
- SF: target-genre confidence from external classifier
- Spectral continuity (MR-STFT score)

Done criteria:
- `lab3_exit_audit.json` written with pass/fail by metric.

## Resume Strategy

- `run_state.json` tracks stage completion and config.
- Safe resume from `stage1_latest.pt` / `stage2_latest.pt`.
- Cached arrays avoid repeated latent extraction.
