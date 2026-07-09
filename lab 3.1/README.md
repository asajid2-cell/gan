# Lab 3.1 - Clean-Slate Pipeline Reset

This workspace is the notebook-first reset for DGGR after the original Lab 3/4 runs exposed a mismatch between:

- what the project wants conceptually: strong genre remastering
- what the codec branch often optimized in practice: source-preserving latent edits with shallow style cues

The purpose of `lab 3.1` is to stop extending inherited runs and instead:

1. audit the full Lab 1 -> Lab 4 pipeline
2. restate the actual research bottleneck
3. rerun the critical stages from scratch in notebooks
4. make promotion decisions based on generated-audio realism and target-style movement, not only internal training gates

## Folder layout

- `notebooks/`
  - `00_pipeline_audit.ipynb`: loads the saved runs and diagnosis tables
  - `01_lab1_lab2_audit.ipynb`: verifies whether Lab 1/2 are really the bottleneck
  - `02_codec_from_scratch.ipynb`: notebook-first fresh codec training path
  - `03_diffusion_from_scratch.ipynb`: notebook-first fresh diffusion training path
  - `04_lab4_longform_validation.ipynb`: long-form validation after short-form quality is good enough
  - `05_overnight_full_pipeline.ipynb`: single overnight controller for fresh codec + diffusion V2 + diffusion V3 + long-form evaluation
  - `06_diffusion_downloads_batch_inference.ipynb`: batch inference on random `Downloads/` songs across a panel of diffusion checkpoints (`latest`, `best`, `epoch_006`, and random epochs)
  - `07_diffusion_longform_checkpoint_compare.ipynb`: long-form comparison on random `Downloads/` songs across selected diffusion checkpoints such as `best` and `epoch_005`
  - `08_diffusion_longform_retool.ipynb`: longform-aware diffusion fine-tune that adds target-style swap training and adjacent-chunk stability losses before running the usual sweep/batch/longform checks
  - `09_codec_longform_checkpoint_compare.ipynb`: long-form comparison on random `Downloads/` songs for codec checkpoints such as `stage3_latest` and `stage2_latest`
  - `10_diffusion_best_longform_panel.ipynb`: curated long-form diffusion notebook that preloads the strongest saved checkpoints and runs random `Downloads/` songs with style-forward but stability-aware settings
  - `11_run_d002_shortform_random50.ipynb`: fixed `run_d002` short-form audition notebook that generates 50 random `Downloads/` clips to isolate checkpoint quality from long-form compounding
  - `12_diffusion_vocal_crackle_retool.ipynb`: notebook-first diffusion fine-tune focused on vocal stability and anti-crackle regularization with practical capped epochs
  - `13_epoch1_longform_settings_sweep.ipynb`: keeps an early checkpoint fixed and sweeps 12 long-form coherence presets on the same random `Downloads/` songs to expose how the dials change output quality
  - `14_epoch123_vocal_noise_settings_sweep.ipynb`: runs epochs `001` to `003` across 20 low-noise, vocal-isolation, and HF-guard long-form settings on the same 5 random `Downloads/` songs
- `docs/`
  - `pipeline_diagnosis.md`: holistic explanation of the current pipeline shortcomings
  - `clean_slate_experiment_plan.md`: concrete hypothesis-driven reset plan
- `scripts/`
  - `pipeline_audit.py`: generates cross-run summary tables used by the notebooks
  - `diffusion_downloads_batch.py`: resolves the latest diffusion checkpoint and batch-generates random `Downloads/` clips
  - `diffusion_longform_compare.py`: runs the Lab 4 coherence pipeline across a checkpoint panel on random `Downloads/` songs
  - `diffusion_longform_settings_sweep.py`: runs the same fixed long-form jobs across a panel of 12 coherence-setting presets for one checkpoint
  - `diffusion_longform_retool_train.py`: retooled diffusion fine-tune for stronger style transfer and longform stability
  - `codec_longform_compare.py`: runs chunked codec long-form generation across a checkpoint panel on random `Downloads/` songs
- `outputs/`
  - generated audit CSV/JSON tables

## Core diagnosis

Current evidence suggests:

- Lab 1 and Lab 2 are not the primary bottleneck.
- The Lab 3 codec branch has an objective/architecture mismatch:
  - internal style gates can look strong
  - generated-audio realism/style metrics stay weak or contradictory
- The diffusion branch is conceptually closer to "from-the-ground-up remastering" than the codec translator.
- Lab 4 coherence heuristics cannot rescue a short-form model that never enters the correct target-style manifold.

## Notebook-first workflow

1. Run `00_pipeline_audit.ipynb`.
2. Confirm the diagnosis with `01_lab1_lab2_audit.ipynb`.
3. Train fresh short-form baselines in:
   - `02_codec_from_scratch.ipynb`
   - `03_diffusion_from_scratch.ipynb`
4. Only once short-form realism and movement are both good enough, move to:
   - `04_lab4_longform_validation.ipynb`

## Audit tables

To refresh the saved-run summaries:

```powershell
python "lab 3.1/scripts/pipeline_audit.py"
```

Outputs land under:

`lab 3.1/outputs/audit/`

## Reset reading order

1. `docs/pipeline_diagnosis.md`
2. `docs/clean_slate_experiment_plan.md`
3. `notebooks/00_pipeline_audit.ipynb`
4. `notebooks/05_overnight_full_pipeline.ipynb`
