# Long-Form Coherence (Explanation)

The central long-form problem is *error accumulation*: small deviations in rhythm/timbre/pitch that are tolerable in a 3–5 second clip can compound over minutes.

DGGR’s current approach combines three ideas:

## 1) SDEdit-style source anchoring

Instead of sampling each chunk from pure noise, we:
- forward-diffuse a source mel to timestep `t-start`
- denoise with target style conditioning

This preserves the source’s macro-structure while allowing controlled timbral edits.

## 2) Prefix overlap locking at every reverse step

We enforce that the overlap region at the start of the current chunk matches the tail overlap region of the previous chunk, at the *same noise level* during DDIM steps.

This converts “stitching” from a post-hoc crossfade into a constraint baked into sampling.

## 3) Drift controls and stabilization

Practical additions that help long tracks:
- periodic re-anchoring (`--reanchor-every`)
- mel smoothing
- blending toward source mel globally or in high-frequency bins

The resulting system is coherence-first: it prioritizes continuity and structural stability, then increases style magnitude only as far as stability allows.

