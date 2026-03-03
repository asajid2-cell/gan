# Architecture Overview

DGGR is implemented as a staged pipeline:

1. **Lab 1: Deconstruction Encoder**
   - Learns disentangled latents: `z_content` (style-suppressed) and `z_style` (style-bearing).
   - Uses adversarial training (GRL) to remove style from `z_content`.
   - Adds a music gate head to reject non-music segments downstream.

2. **Lab 2: Target Vector Space**
   - Builds a 160D style target vector per sample:
     - `z_style` (128D) plus `descriptor32` (32D; summary stats over mel bands).
   - Calibrates genre centroids with inlier filtering and validates separability.

3. **Lab 3: Reconstruction (Codec and Diffusion branches)**
   - Codec branch: translate EnCodec embeddings with FiLM conditioning (best short-form style metrics: run1055).
   - Diffusion branch: generate mel spectrograms with v-prediction DDIM + CFG; vocode with BigVGAN.

4. **Lab 4: Long-Form Coherence**
   - Chunked generation with overlap locking at every DDIM step.
   - SDEdit-style source anchoring for structural preservation.
   - Drift controls: re-anchoring, mel stabilization, HF blending, style-strength mixing.

See `docs/reference/metrics.md` for the evaluation criteria and how they are computed in code.

