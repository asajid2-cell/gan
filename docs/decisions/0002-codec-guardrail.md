# 0002: Use EnCodec as a Decoder Guardrail for Lab 3

Date: 2026-03-02

## Context

Early raw waveform / spectrogram GAN attempts were unstable (phase artifacts, screeching, brittle training).
We needed a reconstruction approach that preserves acoustic realism while still allowing genre edits.

## Decision

Perform style transfer in a pretrained neural codec latent space:
- Encode audio using EnCodec to latents `q_src`.
- Learn a translator to produce `q_hat`.
- Decode with EnCodec’s pretrained decoder to waveform.

## Consequences

Positive:
- High baseline audio quality (decoder acts as a prior).
- Lower risk of catastrophic waveform artifacts.

Negative:
- Latent bottleneck can limit perceptual magnitude of remastering.
- Strong edits can still drift without explicit long-form constraints (motivating Lab 4).

