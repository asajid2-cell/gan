# Lab 3: Codec Latent Translation (Explanation)

## Goal

Perform style transfer with high acoustic realism by operating in a pretrained codec latent space rather than raw waveform generation.

## Data representation

- EnCodec encodes waveform -> embeddings `q_src` (shape roughly `[B, 128, T]`).
- Translator predicts `q_hat` conditioned on:
  - `z_content` (Lab 1)
  - `z_style_tgt` (style conditioning embedding)
  - injected noise (for stochasticity and mode seeking)
- EnCodec decodes `q_hat` -> waveform.

## Translator architecture (implemented)

- Conv1D in-projection
- Stack of FiLM-conditioned residual blocks with dilations
- GroupNorm + SiLU + Conv1D out-projection

Two output modes:
- residual: `q_hat = q_src + s*tanh(raw)`
- direct-output: `q_hat = raw` (best run uses this)

## Objective (implemented)

Generator loss is a weighted sum:
- adversarial hinge loss + feature matching (multi-scale waveform discriminator)
- latent L1 and latent continuity (smoothness)
- MR-STFT waveform reconstruction proxy
- content cosine preservation (`z_content`)
- style loss (judge CE or probe CE depending on mode)
- push loss discouraging source genre probability
- optional mode-seeking loss

## Best configuration and why it worked

The best run (`run1055`) used:
- MERT-probe embeddings as the style conditioning space
- direct-output translator mode (no residual leash)

This combination improved style separability in the conditioning space and removed an architectural cap on edit magnitude, while content preservation remained enforced through explicit content loss.

See `docs/explanation/results.md` for the achieved metrics.

