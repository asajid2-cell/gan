# Lab 2 Plan - Genre Target Vector Space

## Goal
Build a stable 160-D target vector space from frozen Lab 1 style latents, then verify that genre blueprints are separable and decoder-ready.

## Phase A - Data and Label Materialization
- Load cleaned manifests from `Z:/DataSets/_lab1_manifests`.
- Assign each sample to a Lab 2 genre bucket:
  - `baroque_classical`
  - `hiphop_xtc`
  - `lofi_hh_lfbb`
  - `cc0_other`
- Materialize balanced subsets with `per_genre_samples` cap.

Done criteria:
- No missing path failures in sampled pool.
- Each active genre has enough items for split-based validation.

## Phase B - Frozen Encoder Harvest
- Load frozen Lab 1 checkpoint.
- For each sample, extract:
  - `z_content` (128-D)
  - `z_style` (128-D)
  - handcrafted descriptor (32-D)
  - `target160 = concat(z_style, descriptor32)`

Done criteria:
- `embeddings_index.csv` and `embeddings.npz` are written.
- Vector shapes are consistent and complete.

## Phase C - Centroid Construction
- Compute normalized mean vector per genre in 160-D space.
- Export centroid table and pairwise distance matrix.

Done criteria:
- `centroids_160d.csv` and `centroid_distances.csv` written.
- Distances show non-trivial inter-genre separation.
- `target_centroids.json` exported for Lab 3 conditioning.

## Phase D - Validation Audit
- Run linear probe on `target160`.
- Run nearest-centroid target-fidelity proxy.
- Compute silhouette score (cosine).
- Run inter-centroid separation test (`distance >= 3 * sigma`).
- Run nearest-neighbor blueprint audit (`top-5 all hits` per genre).
- Run centroid stability test (20% subset trials).
- Export global t-SNE map.

Done criteria:
- `validation_summary.json` contains:
  - `linear_probe_acc`
  - `nearest_centroid_acc`
  - `target_fidelity_metric`
  - `silhouette`
- `lab2_exit_checklist.json` contains all exit pass/fail gates.

## Run Ladder
1. Smoke: `python run_lab2.py --smoke --per-genre-samples 20`
2. Mid: `python run_lab2.py --per-genre-samples 300`
3. Full: `python run_lab2.py --per-genre-samples 1200`

## Exit Thresholds
- Separability: `silhouette >= 0.45`
- Resolution: inter-centroid distance `>= 3 sigma`
- Blueprint accuracy: `5/5` nearest-neighbor hits per genre
- Readiness: `target_centroids.json` exists
