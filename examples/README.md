# Examples

This folder contains **small, curated outputs** that are safe to commit and useful for quick qualitative sanity checks.

What is included:
- short generated `.wav` clips (no large checkpoints)
- metadata describing the producing run/checkpoint

What is not included:
- training datasets
- large caches (`saves/`, `saves2/`)
- large model checkpoints (GitHub size limits). For checkpoints, see `docs/howto/reproduce_best_runs.md`.

## Audio

`examples/audio/` includes:
- diffusion mel-generation samples (generated clips only; no ground-truth clips)
- codec-latent transfer samples (translated clips only; source clips are not included)

