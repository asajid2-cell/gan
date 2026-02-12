# Lab 2 - Genre Target Vector Space

Lab 2 turns the frozen Lab 1 Deconstruction Encoder into a style-harvesting system.

## Objective

Build a high-resolution genre blueprint space (160-D vectors) that can guide downstream
generative reconstruction:

- Preserve content/style disentanglement from Lab 1.
- Harvest style markers from broad public-domain datasets.
- Compute genre centroids (target vectors).
- Validate separability and target fidelity with quantitative audits.

## What This Folder Contains

- `run_lab2.py`: CLI entry point for Lab 2 pipeline.
- `lab2_genre_target_vector_space.ipynb`: notebook-first Lab 2 workflow.
- `PLAN.md`: staged execution plan and success gates.
- `src/lab2_encoder_bridge.py`: loads frozen Lab 1 checkpoint and performs inference.
- `src/lab2_data.py`: dataset/manifests and genre-spec materialization.
- `src/lab2_pipeline.py`: harvesting, centroid building, and validation.
- `requirements.txt`: extra dependencies.

## Default Data Inputs

Uses manifests in `Z:/DataSets/_lab1_manifests`:

- `xtc_audio_clean.csv`
- `hh_lfbb_audio_clean.csv`
- `cc0_audio_clean.csv`
- `phase1_symbolic_audio_manifest.csv` (optional fallback for classical/baroque pool)

Default Lab 2 genre buckets:

- `baroque_classical`
- `hiphop_xtc`
- `lofi_hh_lfbb`
- `cc0_other`

## Default Frozen Encoder

By default the CLI uses:

`../saves/lab1_run_combo_af_gate_exit_v2/latest.pt`

You can override via `--checkpoint`.

## 160-D Target Vector Definition

For each audio sample:

- 128-D: `z_style` from frozen Lab 1 encoder.
- 32-D: handcrafted descriptor from log-mel summary statistics.

Concatenation yields a 160-D style target vector.

## Validation Outputs

The pipeline writes:

- `validation_summary.json`
- `centroids_160d.csv`
- `centroid_distances.csv`
- `genre_samples.csv`
- `embeddings.npz`
- `embeddings_index.csv`
- `target_centroids.json`
- `vector_variance_report.csv`
- `inter_centroid_separation.csv`
- `centroid_stability_trials.csv`
- `centroid_stability_report.json`
- `neighbor_audit.csv`
- `neighbor_audit_summary.csv`
- `global_genre_map_tsne.csv`
- `global_genre_map_tsne.png`
- `lab2_exit_checklist.json`

Core metrics:

- Linear probe accuracy (genre separability).
- Nearest-centroid accuracy (target fidelity proxy).
- Silhouette score (cluster separation).
- Pairwise centroid distances.
- Inter-centroid `>= 3 sigma` separation pass/fail.
- Neighbor `5/5` per genre pass/fail.

## Quick Start

```powershell
cd "lab 2"
python run_lab2.py --smoke
```

Notebook path:

`lab 2/lab2_genre_target_vector_space.ipynb`

Full run example:

```powershell
cd "lab 2"
python run_lab2.py --per-genre-samples 1200
```

Threshold controls:

```powershell
python run_lab2.py `
  --silhouette-threshold 0.45 `
  --sigma-multiplier 3.0 `
  --neighbor-top-k 5
```

Post-harvest supervised warp (reuse previous embeddings, no re-harvest):

```powershell
python run_lab2.py `
  --reuse-artifacts-dir "../saves/lab2_calibration/lab2_20260211_015118" `
  --projection lda `
  --zstyle-weight 2.0 `
  --descriptor-weight 1.0 `
  --centroid-inlier-fraction 0.5 `
  --audit-inlier-fraction 0.5 `
  --neighbor-min-mean-hits 4.0 `
  --output-dir "../saves/lab2_calibration/lab2_20260211_015118_lda"
```

Default output root:

`../saves/lab2_calibration/<timestamp>/`

## Notes

- `FPR@0.5` from Lab 1 is a threshold calibration issue, not an embedding-rank issue.
- Lab 2 focuses on style-space quality and centroid robustness for Lab 3 conditioning.
- If runtime fails on `numpy/pandas` import, repair environment first:
  `python -m pip install --upgrade --force-reinstall numpy pandas`.
