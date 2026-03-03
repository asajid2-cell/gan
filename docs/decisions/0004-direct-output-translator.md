# 0004: Remove Residual Leash (Direct-Output Translator)

Date: 2026-03-02

## Context

Residual translation of codec latents (e.g., `q_hat = q_src + s*tanh(raw)`) tends to cap style shift magnitude.
We observed a style ceiling when the residual path dominated.

## Decision

Enable direct-output mode in the codec translator so the model can rewrite `q_hat` without an identity shortcut,
while preserving content through explicit content loss.

## Consequences

Positive:
- Large jump in achieved style confidence and accuracy without content regression (run1055).

Negative:
- Higher risk of content drift if content loss is weakened or mis-specified.

