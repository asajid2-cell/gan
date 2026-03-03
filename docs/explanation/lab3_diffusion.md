# Lab 3/4: Diffusion Branch (Explanation)

## Goal

Enable larger perceptual remastering than codec translation by generating mel spectrograms directly, while retaining controllability via content/style conditioning.

## Representation

- Mel spectrograms are extracted using BigVGAN-compatible settings (80 mel bins).
- Conditioning features include chroma, onset envelope, and beat grid (cached).
- Content/style embeddings (`z_content`, `z_style`) come from the frozen Lab 1 encoder.

## Diffusion model (V2)

Key choices:
- v-prediction objective (stability vs epsilon)
- EMA for sampling stability
- classifier-free guidance (CFG) via dropout at training and guidance at inference
- StyleAdaIN blocks separate style modulation from time/content FiLM

## Diffusion model (V3)

Extension:
- add a mel discriminator with hinge loss + feature matching

Observation:
- In current runs, V3 did not beat V2 on validation; its primary purpose is to provide a scaffold to reduce over-smoothing (mode averaging) in future tuning.

## Checkpoints

We selected epoch 6 from `run_d002` as the best subjective checkpoint for generation even though later epochs improved numeric validation loss.
This is consistent with diffusion training sometimes optimizing toward smoother “average” outputs over time.

