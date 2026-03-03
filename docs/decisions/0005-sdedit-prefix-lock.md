# 0005: Long-Form Coherence via SDEdit Anchoring + Prefix Lock

Date: 2026-03-02

## Context

Chunked generation accumulates drift: small timing/timbre differences compound across minutes and produce
warble/static and seam artifacts.

## Decision

In long-form diffusion generation:
- Use SDEdit-style anchoring: start each chunk by forward-diffusing the *source* mel to timestep `t_start`.
- Enforce overlap coherence by prefix-locking overlap frames at every DDIM step using the previous chunk tail.

## Consequences

Positive:
- Reduces boundary discontinuities and improves perceived cohesion.
- Provides explicit knobs to trade style magnitude vs coherence.

Negative:
- Over-constraint can reduce stylistic diversity.
- Still requires stabilization (re-anchoring, mel blending) for very long tracks.

