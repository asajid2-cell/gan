# Lab 3 - Reconstruction Decoder (Analysis -> Synthesis)

Lab 3 turns the frozen analysis stack into a generative synthesis stack.

If you are starting fresh after the earlier codec/diffusion tuning cycle, use the reset workspace in
`lab 3.1/` first. That path is notebook-first and is intended to audit the old runs before launching
new from-scratch training.

## Objective

Build a conditional Reconstruction Decoder that takes:

- `z_content` (Lab 1 structural skeleton, 128-D)
- `V_target` (Lab 2 genre target vector)

and synthesizes a Log-Mel spectrogram that preserves melody while shifting style.

## Two-Stage Training

1. Stage 1 - Self-Reconstruction Baseline
- Condition on the sample's own genre target vector.
- Reconstruct the original Log-Mel target.
- Validates decoder capacity and conditioning path.

2. Stage 2 - Genre-Shift Synthesis
- Condition on a different genre target vector.
- Supports conditioning modes: `centroid`, `exemplar`, `mix` (default `mix`).
- Preserve `z_content` while steering style toward target genre.
- Uses content/style consistency + adversarial + spectral continuity regularization.

## Data Generalization Controls

- Multi-chunk sampling per track during cache build:
  - `--chunks-per-track` (default `4`)
  - `--chunk-sampling` (`uniform` or `random`)
  - `--min-start-sec`, `--max-start-sec`
- Grouped split to avoid track leakage:
  - `--split-by-track` (default on)
  - uses `track_id` in cache index to ensure train/val track disjointness.

## Exit Metrics

- `MPS` (Melodic Preservation Score): cosine(`z_content`, `z_content'`) >= `0.90`
- `SF` (Stylistic Fidelity): classifier confidence in target genre >= `0.85`
- `Spectral Continuity`: multi-resolution STFT continuity score reported (lower is better)

## Save/Resume

Run artifacts are written to:

`../saves2/lab3_synthesis/runN/`

with:

- `run_state.json`
- `checkpoints/stage1_latest.pt`
- `checkpoints/stage2_latest.pt`
- `history.csv`
- `lab3_exit_audit.json`

Resume by passing `--mode resume --resume-dir <run_dir>`.

By default, strict run naming is enforced (`run1`, `run2`, ...), and each completed run
auto-exports a standardized post-train sample pack to:

`../saves2/lab3_synthesis/runN/samples/posttrain_samples/`

## Quick Start

```powershell
cd "lab 3"
python run_lab3.py --smoke
```

Notebook runner:

`lab 3/notebooks/04_lab3_reconstruction_decoder.ipynb`

Full run example:

```powershell
cd "lab 3"
python run_lab3.py `
  --per-genre-samples 800 `
  --stage1-epochs 20 `
  --stage2-epochs 20
```

## Codec-Latent Transfer Track (Fresh Architecture)

This track replaces mel-target generation with frozen EnCodec latents and waveform decoding:

- source waveform -> frozen EnCodec encoder -> quantized latent embedding
- translator conditions on `z_content` (Lab1) + target style exemplars
- frozen EnCodec decoder outputs waveform (no Griffin-Lim in training path)

Entry point:

```powershell
cd "lab 3"
python run_lab3_codec.py --smoke
```

## Realism-First Checkpoint Supervision

Once a run is already preserving melody and moving style, the next bottleneck is realism.
Use the realism supervisor to rank checkpoints by naturalness instead of relying on style/content metrics alone.

It evaluates checkpoints on a fixed validation transfer plan and reports:

- `fad_mert` (true Fréchet distance over pretrained MERT embeddings)
- `target_centroid_mae_norm`
- `target_hf_mae`
- `target_lf_mae`
- `target_dynamic_range_mae_db`
- plus light transfer checks like `mps` and `style_target_acc`

Codec example:

```powershell
python "lab 3/run_lab3_realism_sweep.py" codec `
  --run-dir "saves2/lab3_codec_transfer/run1055" `
  --n-samples 24
```

Diffusion example:

```powershell
python "lab 3/run_lab3_realism_sweep.py" diffusion `
  --run-dir "saves2/lab3_diffusion/run_d002" `
  --checkpoints epoch_006.pt best.pt `
  --n-samples 12
```

See `lab 3/docs/05_realism_supervisor.md` for the full workflow and gating options.

## Late-Stage Rebuild Tuning

The current bottleneck is no longer basic melody preservation. It is making late-stage codec checkpoints
move far enough into the target genre without collapsing into robotic artifacts.

Two repo changes now support that workflow directly:

- late-stage generated-audio MERT supervision:
  - `--stage2-generated-mert-weight`
  - `--stage3-generated-mert-weight`
  - `--stage2-generated-mert-align-weight`
  - `--stage3-generated-mert-align-weight`
  - `--stage2-generated-mert-every`
  - `--stage3-generated-mert-every`
- bootstrap-from-stage1 runs:
  - `--bootstrap-ckpt`
  - use this with `--skip-stage1` to tune only stage 2 and stage 3 from a known-good reconstruction checkpoint

Recommended workflow:

1. Reuse a stable stage 1 checkpoint.
2. Run a short late-stage probe.
3. Sweep the resulting checkpoints with the realism supervisor.
4. Promote only a probe config that improves both realism and target-style movement.
5. Then launch the full late-stage run.

Launchers:

```powershell
# short late-stage probe from a frozen stage1 checkpoint
powershell -ExecutionPolicy Bypass -File "lab 3/scripts/start_codec_rebuild_probe.ps1"

# full late-stage rebuild run once a probe config is approved
powershell -ExecutionPolicy Bypass -File "lab 3/scripts/start_codec_rebuild_full.ps1"
```

## Strong Schema (Unpaired-Validity Runs)

If your "genre" labels are coupled to dataset source (common in this project), the model and/or judge can learn
source fingerprints instead of transferable style. The strongest fix you can do *without new data* is to:

- remap labels into multi-source buckets (`--genre-schema binary_acoustic_beats`)
- balance sources within each bucket (`--balance-sources-within-genre`)
- require each bucket to have >=2 sources (`--require-min-sources-per-genre 2`)
- optionally filter to music-only (`--require-is-music`)

Helpers (recommended):

```powershell
# quick sanity run (cache + judge + style bank + tiny training)
./lab 3/scripts/run_codec_strong_schema_smoke.ps1

# full strong-schema run
./lab 3/scripts/run_codec_strong_schema_full.ps1

# audit the latest run for source leakage
./lab 3/scripts/run_codec_audit_latest.ps1
```

## Auto-Genre (Unpaired Labeling)

If you want "genres" that are not just dataset-source buckets, you need labels derived from audio content.
Two unpaired options are provided:

1. CLAP zero-shot prompts (semantic, external model):

```powershell
cd "lab 3"
python run_lab3_auto_genre.py --manifests-root "%DGGR_MANIFESTS_ROOT%" --out-csv "auto_genre_4way.csv" --labels hiphop lofi classical electronic
```

End-to-end helper (CLAP label + train + audit):

```powershell
powershell -ExecutionPolicy Bypass -File "lab 3/scripts/run_codec_clap_labels_full.ps1"
```

2. Lab2-style clustering (internal, no text model):
Clusters `target160 = [z_style, descriptor32]` into `K` style buckets and writes `genre=cluster_i`.

```powershell
cd "lab 3"
python run_lab3_auto_genre_lab2cluster.py --manifests-root "%DGGR_MANIFESTS_ROOT%" --out-csv "auto_cluster_k4.csv" --n-clusters 4
```

Recommended fresh 3-run sequence:

```powershell
cd "lab 3"

# run1: identity-only sanity (stage1 focus)
python run_lab3_codec.py `
  --run-name run1 `
  --stage1-epochs 8 `
  --stage2-epochs 0 `
  --stage3-epochs 0

# run2: cross-style transfer
python run_lab3_codec.py `
  --run-name run2 `
  --stage1-epochs 8 `
  --stage2-epochs 16 `
  --stage3-epochs 0

# run3: transfer + diversity pressure
python run_lab3_codec.py `
  --run-name run3 `
  --stage1-epochs 8 `
  --stage2-epochs 16 `
  --stage3-epochs 8 `
  --stage3-style-dropout-p 0.25 `
  --mode-seeking-weight 1.0
```

## Fast Manual Clip Triage

Use the interactive picker to quickly audition random clips and accept/reject them into lists:

```powershell
cd "lab 3"
python run_lab3_clip_picker.py `
  --input-csv "../saves2/lab3_synthesis/run20/samples/posttrain_samples/generation_summary.csv" `
  --path-col fake_wav `
  --base-dir ".." `
  --session-name run20_fake_triage `
  --max-clips 200 `
  --auto-open
```

Controls:

- `a` accept
- `r` reject
- `s` skip
- `o` reopen/replay
- `q` quit

## Folder Notes

- Notebooks now live under `lab 3/notebooks/`.
- Helper PowerShell launchers now live under `lab 3/scripts/`.
- Lab 3 explanation and how-to notes now live under `lab 3/docs/`.
