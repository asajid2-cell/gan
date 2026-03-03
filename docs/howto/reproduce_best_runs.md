# Reproduce Best Runs (How-To)

This page collects copy-paste commands that reproduce the project’s best observed checkpoints and settings.

Assumptions:
- You have the required caches/checkpoints present locally (or you ran the cache builders).
- You run from the repo root.

## 1) Codec transfer (best short-form style metrics)

Best run so far:
- `saves2/lab3_codec_transfer/run1055`
- Key config: `style_cond_source=mert_probe_embed`, `translator_direct_output=true`

To train a new run with the same recipe:

```powershell
python \"lab 3/run_lab3_codec.py\" `
  --mode fresh `
  --style-cond-source mert_probe_embed `
  --style-loss-mode mert_probe_ce `
  --translator-direct-output `
  --per-genre-samples 600
```

To evaluate a trained run:
- Check `codec_gate_eval.json` in the run folder.

## 2) Diffusion V2 (best perceived quality checkpoint)

Current best subjective checkpoint:
- `saves2/lab3_diffusion/run_d002/checkpoints/epoch_006.pt`

To continue training V2:

```powershell
python \"lab 3/run_lab3_diffusion_v2.py\" `
  --cache-dir \"saves2/lab3_diffusion/run_d001/cache\" `
  --out-dir \"saves2/lab3_diffusion/run_d002\" `
  --epochs 60
```

## 3) Long-form coherence (full-song test)

Run long-form coherence with the V2 epoch 6 checkpoint:

```powershell
python \"lab 3/run_lab4_longform_coherence.py\" `
  --cache-dir \"saves2/lab3_diffusion/run_d001/cache\" `
  --checkpoint \"saves2/lab3_diffusion/run_d002/checkpoints/epoch_006.pt\" `
  --out-dir \"saves2/lab4_longform_coherence/repro\" `
  --source-audio \"PATH_TO_AUDIO_FILE\" `
  --source-genre hiphop_xtc `
  --target-genre baroque_classical `
  --t-start 350 `
  --prefix-blend 1.0 `
  --style-strength 0.75
```

If you hear accumulating warble/static:
- Reduce `--t-start`
- Increase `--source-mel-blend` and/or `--hf-source-blend`
- Use `--reanchor-every` (e.g., 8–16)

