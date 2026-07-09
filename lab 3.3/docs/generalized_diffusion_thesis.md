# Generalized Diffusion Thesis

This run is not trying to invent a brand new generator from scratch. The base diffusion checkpoint `run_d002/epoch_006.pt` already has the strongest short-form sound in the project. The failure mode is that long-form rollout and arbitrary user songs expose a narrow training distribution: the model sounds better on cached training-like material than on real songs from `Downloads`.

The thesis for this run is:

1. Keep the strong local generator.
2. Preserve the longer 5-second context.
3. Broaden the source distribution the model sees during training by perturbing the source side in ways that mimic real-world mastering and recording variation.
4. Keep epoch-by-epoch samples on both cache clips and `Downloads` clips so the run is judged on out-of-distribution behavior, not just loss curves.

The practical architecture is still the longform-aware conditional diffusion U-Net, but the training regime changes:

- bootstrap from `run_d002 epoch_006`
- 5-second chunks (`max_frames=432`)
- shared source perturbations across adjacent chunks
- stronger crackle and vocal protections
- fixed checkpointing and mid-epoch resume
- epoch sample generations from both cache and arbitrary songs

The goal is not maximum style at any cost. The goal is a more general source-conditioned remastering model that survives arbitrary songs well enough that long-form conditioning becomes useful rather than destructive.
