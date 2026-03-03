# Lab 4: Long-Form Coherence (Explanation)

## Goal

Generate minutes of audio that remain structurally consistent and seam-free, without drift compounding across chunk boundaries.

## Core approach (implemented)

We generate overlapping chunks and enforce continuity in two ways:

1. SDEdit-style source anchoring:
   - forward diffuse the *source* mel to timestep `t_start`
   - denoise toward a target style embedding

2. Prefix-lock overlap frames during sampling:
   - at every DDIM step, overwrite the overlap prefix with a noise-matched version of the previous chunk’s tail overlap

## Stabilization controls (implemented)

Practical knobs that reduce warble/static accumulation:
- periodic re-anchoring (`--reanchor-every`)
- mel smoothing (time/frequency)
- blending toward source mel globally (`--source-mel-blend`)
- extra blending for high mel bins (`--hf-source-blend`)

## Diagnostics

We record boundary discontinuity metrics (mel MSE and dB proxy) to quantify seam quality across chunks.
These diagnostics are not a full perceptual metric, but they provide a reliable tuning signal.

