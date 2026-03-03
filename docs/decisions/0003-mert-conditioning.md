# 0003: Use MERT-Probe Embeddings for Style Conditioning

Date: 2026-03-02

## Context

Conditioning on raw Lab 1 `z_style` centroids collapsed in practice (centroid cosine similarity near 1.0),
which limited controllability and reduced style confidence.

## Decision

Use MERT as the style feature backbone and train a probe that produces a well-separated embedding space.
Use the probe embedding as `z_style_tgt` for FiLM conditioning.

## Consequences

Positive:
- Better-separated target style vectors (measurable via centroid diagnostics).
- Enabled style confidence to exceed the project target in Lab 3 (run1055).

Negative:
- Adds dependency on a large pretrained model (MERT) and its inference cost (mitigated by caching).

