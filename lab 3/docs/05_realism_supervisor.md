# Lab 3 Realism Supervisor

This is the realism-first checkpoint selection path for Lab 3.

It exists because the main training metrics already tell us whether:

- melody/content survived
- target style moved at all

but they do not reliably tell us whether the audio sounds natural.

## What it measures

`lab 3/run_lab3_realism_sweep.py` promotes realism to a first-class checkpoint criterion.

For each checkpoint it:

1. Builds a fixed source/target transfer plan from the validation split.
2. Generates the same set of transfers for every checkpoint in the sweep.
3. Computes true Fréchet distance over pretrained MERT embeddings (`fad_mert`).
4. Measures target-reference spectral alignment:
   - `target_centroid_mae_norm`
   - `target_hf_mae`
   - `target_lf_mae`
   - `target_dynamic_range_mae_db`
5. Keeps light transfer checks in the report:
   - `mps`
   - `style_target_acc`
   - `style_target_cos`
   - `style_target_cos`

The script writes a ranked CSV plus a `*_realism_best.json` summary.

## Why this is different from the old audit

The older audit in `lab 3/run_lab3_quality_audit.py` included `fad_mfcc_proxy`, which is only a proxy.

The realism supervisor replaces that proxy with a true Fréchet computation over pretrained audio embeddings.
In this repo the default backbone is MERT because it is already used elsewhere in the pipeline and is music-aware.

## Codec sweep

Example:

```powershell
python "lab 3/run_lab3_realism_sweep.py" codec `
  --run-dir "saves2/lab3_codec_transfer/run1055" `
  --n-samples 24 `
  --write-audio-count 4
```

Outputs:

- `saves2/lab3_codec_transfer/run1055/realism_supervisor/codec_realism_sweep.csv`
- `saves2/lab3_codec_transfer/run1055/realism_supervisor/codec_realism_best.json`
- `saves2/lab3_codec_transfer/run1055/realism_supervisor/transfer_plan.csv`

Notes:

- If the run used `mert_probe_embed`, the supervisor will rebuild and cache the probe used to form the 128-D style bank.
- Checkpoint comparisons are fair because every checkpoint sees the same transfer plan.

## Diffusion sweep

Example:

```powershell
python "lab 3/run_lab3_realism_sweep.py" diffusion `
  --run-dir "saves2/lab3_diffusion/run_d002" `
  --checkpoints epoch_006.pt best.pt `
  --n-samples 12 `
  --write-audio-count 2
```

Outputs:

- `saves2/lab3_diffusion/run_d002/realism_supervisor/diffusion_realism_sweep.csv`
- `saves2/lab3_diffusion/run_d002/realism_supervisor/diffusion_realism_best.json`
- `saves2/lab3_diffusion/run_d002/realism_supervisor/transfer_plan.csv`

Notes:

- The supervisor skips corrupted checkpoints instead of aborting the whole sweep.
- Longer sweeps are intentionally batch-oriented. Run them overnight if you want stable rankings.

## Hard gating

You can turn ranking into hard gating by supplying thresholds:

```powershell
python "lab 3/run_lab3_realism_sweep.py" codec `
  --run-dir "saves2/lab3_codec_transfer/run1055" `
  --max-fad-mert 35 `
  --max-target-hf-mae 0.12 `
  --max-target-dynamic-range-mae-db 12 `
  --min-mps 0.94 `
  --min-style-target-acc 0.50 `
  --min-style-target-cos 0.05
```

If a threshold is provided, the CSV will include `pass_*` columns and `pass_all`.

The supervisor now treats "clean but unchanged" checkpoints as a failure mode.
Ranking still prioritizes realism, but it also penalizes checkpoints whose target-style accuracy/cosine stay too low.

## Intended workflow

1. Train or reuse a stable stage 1 checkpoint.
2. Run a short stage 2 / stage 3 probe with fixed-source epoch samples.
3. Sweep the probe checkpoints with realism plus transformation thresholds.
4. Listen only to the top-ranked checkpoints that pass both realism and movement gates.
5. Launch the full run with the winning late-stage config.
6. Use the human listening panel after the realism supervisor has filtered the search space.
