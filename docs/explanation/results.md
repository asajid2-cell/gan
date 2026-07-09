# Observed Results (Labs 1-4)

This page is the compact historical metrics summary.

For the current end-to-end project state, including:

- what happened after Lab 4
- which experimental branches worked
- which newer model families failed
- what the current production baseline is
- where the major post-lab artifacts now live

see:

- [project_state_20260401.md](/z:/328/CMPUT328-A2/codexworks/301/414-pl1/docs/explanation/project_state_20260401.md)

This page summarizes the best quantitative outcomes observed during the original Lab 1-4 arc.
Raw historical run artifacts mainly lived under `saves/` and `saves2/`.

## Lab 1 (Deconstruction Encoder)

Checkpoint: `saves/lab1_run_combo_af_gate_exit_v2/latest.pt`

- `style_probe_accuracy`: **0.9417** (threshold >= 0.85)
- `content_leakage_above_baseline`: **0.1083** (threshold <= 0.15)
- `gate_roc_auc`: **0.9299** (threshold >= 0.90)

Interpretation:

- `z_style` is strongly style-informative.
- `z_content` is substantially style-suppressed.
- The music gate ranks music vs non-music reliably.

## Lab 2 (Target Vector Space)

Run: `saves/lab2_calibration/lab2_20260211_015118_lda_cleanup_v2/validation_summary.json`

- `silhouette` (cosine): **0.4939** (threshold >= 0.45)
- `linear_probe_acc`: **0.8554**
- `nearest_centroid_acc`: **0.8514**

Interpretation:

- the 160D target space is meaningfully separable by genre
- it is suitable as a conditioning blueprint

## Lab 3 (Codec Latent Translation)

Best run: `saves2/lab3_codec_transfer/run1055`

Gate metrics (`codec_gate_eval.json`):

- `mps`: **0.9565**
- `style_conf`: **0.8940**
- `style_acc`: **0.9492**

Interpretation:

- the codec branch met the original project target style threshold while preserving content

## Lab 3/4 (Diffusion Branch)

Diffusion V2 run: `saves2/lab3_diffusion/run_d002`

- best validation loss (epoch 18): **0.0386**
- selected quality checkpoint (epoch 6): **0.0442**

Interpretation:

- later epochs improved numeric loss
- perceived audio quality peaked earlier
- this became the core realism anchor for the later hybrid/long-form work

## Lab 4 (Long-Form Coherence Diagnostics)

From `saves2/lab4_longform_coherence/fullsong_test/coherence_metrics.json`:

- `boundary_mel_mse_mean`: **0.0018347**
- `boundary_disc_db_mean`: **2.8691**
- `n_chunks`: **64** over **160s**

Interpretation:

- boundary metrics gave the repo a quantitative handle for coherence tuning
- the remaining issues were mainly perceptual warble/static accumulation rather than hard seam breaks
