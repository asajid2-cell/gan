# Lab 3 - Reconstruction Decoder (Analysis -> Synthesis)

Lab 3 turns the frozen analysis stack into a generative synthesis stack.

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

`lab 3/lab3_reconstruction_decoder.ipynb`

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
./scripts/run_codec_strong_schema_smoke.ps1

# full strong-schema run
./scripts/run_codec_strong_schema_full.ps1

# audit the latest run for source leakage
./scripts/run_codec_audit_latest.ps1
```

## Auto-Genre (Unpaired Labeling)

If you want "genres" that are not just dataset-source buckets, you need labels derived from audio content.
Two unpaired options are provided:

1. CLAP zero-shot prompts (semantic, external model):

```powershell
cd "lab 3"
python run_lab3_auto_genre.py --out-csv "Z:/DataSets/_lab1_manifests/auto_genre_4way.csv" --labels hiphop lofi classical electronic
```

End-to-end helper (CLAP label + train + audit):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_codec_clap_labels_full.ps1
```

2. Lab2-style clustering (internal, no text model):
Clusters `target160 = [z_style, descriptor32]` into `K` style buckets and writes `genre=cluster_i`.

```powershell
cd "lab 3"
python run_lab3_auto_genre_lab2cluster.py --out-csv "Z:/DataSets/_lab1_manifests/auto_cluster_k4.csv" --n-clusters 4
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
