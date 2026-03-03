# Data Formats and Artifacts

This repo intentionally does not commit datasets or large caches. This page documents
the expected *schemas* so collaborators can reproduce runs on their own machines.

## Manifest CSVs (cleaned)

Manifests live under `DGGR_MANIFESTS_ROOT` and are used by Lab 2 and Lab 3 scripts.

Minimum expected columns (varies slightly by script):
- `path`: absolute or relative path to an audio file
- `source`: dataset source identifier (e.g. `xtc_audio_clean`)
- `genre`: genre bucket label (one of the current schema labels)
- Optional: `track_id` (used for split-by-track behavior)

Common patterns:
- Code filters out missing paths at runtime (`Path.exists()`).
- Code samples multiple chunks per track for training caches.

## Lab 2 artifacts

Lab 2 writes artifacts under `saves/lab2_calibration/<run>/` (gitignored).

Typical artifacts:
- `validation_summary.json`: metrics summary (probe acc, silhouette, etc.)
- `centroids_160d.csv`: centroid vectors (160D)
- `centroid_distances.csv`: centroid-to-centroid distances
- `embeddings_index.csv` and `embeddings.npz`: raw harvested vectors

## Lab 3 codec cache (EnCodec)

Lab 3 codec transfer builds a cache containing:
- EnCodec latents for source chunks
- Lab 1 `z_content`, `z_style` for source chunks
- Optional MERT features (depending on run configuration)

The cache lives under a run directory in `saves2/lab3_codec_transfer/<run>/cache/`.

## Lab 3 diffusion cache

Diffusion uses BigVGAN-compatible log-mels + conditioning features stored as NumPy arrays.
Cache folder example: `saves2/lab3_diffusion/run_d001/cache/`

Key files:
- `diff_mel.npy`: normalized mel `[-1, 1]`
- `diff_chroma.npy`, `diff_onset.npy`, `diff_beat.npy`: conditioning features
- `diff_z_content.npy`, `diff_z_style.npy`: frozen Lab 1 embeddings
- `diff_genre_idx.npy` and `diff_genre_to_idx.json`: genre labels
- `diff_index.csv`: row index mapping and metadata
- `diff_meta.json`: mel scale and extraction constants

## Checkpoints

Diffusion checkpoints are saved under:
- `saves2/lab3_diffusion/<run>/checkpoints/epoch_*.pt`

Codec transfer checkpoints and run state are saved under:
- `saves2/lab3_codec_transfer/<run>/`

## Long-form outputs

Long-form coherence runner typically produces:
- `longform_coherent.wav`
- `chunk_*.wav`
- `coherence_metrics.json`

