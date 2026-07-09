# Metrics Reference

This page defines the project metrics used across Labs 1–4.

When possible, prefer reporting:
- the metric definition (what it measures),
- the measurement procedure (how it is computed in code),
- and the acceptance criteria (what threshold means “pass”).

## Lab 1 (Disentanglement + gate)

- `style_probe_accuracy`:
  - What: accuracy of a probe predicting style/source from `z_style`.
  - Why: confirms `z_style` retains genre signal.
  - Where: Lab 1 audits under `saves/.../audits_confidence/`.

- `content_leakage_above_baseline`:
  - What: (probe accuracy predicting style from `z_content`) minus chance baseline.
  - Why: lower means `z_content` is more style-invariant.
  - Interpretation: 0.0 means “chance”; 0.15 means “still substantially style-recoverable”.

- `gate_roc_auc`:
  - What: AUC for music-vs-nonmusic gate.
  - Why: threshold-independent separation quality.
  - Note: FPR at a fixed threshold can still be poor even if AUC is strong (calibration issue).

## Lab 2 (Target space calibration)

- `silhouette` (cosine):
  - What: cluster separability in target vector space.
  - Why: higher means genres are geometrically distinct.

- `nearest_centroid_acc`:
  - What: assign each sample to closest centroid by cosine similarity.
  - Why: sanity check that centroids represent their class well.

- `linear_probe_acc`:
  - What: logistic regression genre classifier accuracy on target vectors.
  - Why: measures linear separability of the target space.

## Lab 3 codec gate

- `MPS` (melodic preservation score):
  - What: cosine similarity between source and generated `z_content`.
  - Why: content preservation proxy.
  - Where: computed during codec gate eval (see `lab 3/run_lab3_codec.py`).

- `style_conf`:
  - What: mean probability assigned to the target genre by the gate judge.
  - Why: confidence of style transfer (not just argmax accuracy).
  - Note: can reveal partial transfer even when `style_acc` is similar.

- `style_acc`:
  - What: mean hit-rate of predicting target genre.
  - Why: discrete correctness.

## Lab 4 coherence diagnostics

- `boundary_mel_mse_*`:
  - What: mel-space mismatch at chunk boundaries.
  - Why: direct proxy for seam artifacts.

- `boundary_disc_db_*`:
  - What: dB-scale boundary discontinuity proxy.
  - Why: correlates with perceived clicks/jumps between chunks.

## Realism supervisor

- `fad_mert`:
  - What: true Fréchet distance between generated and real-audio embedding distributions using pretrained MERT embeddings.
  - Why: primary realism metric for checkpoint ranking once melody/style are already acceptable.
  - Note: lower is better.

- `target_centroid_mae_norm`:
  - What: normalized error between generated spectral centroid and the target-reference centroid profile.
  - Why: catches tonal balance drift.

- `target_hf_mae`:
  - What: absolute error in the high-frequency energy ratio versus target-reference audio.
  - Why: catches brittle, screechy, over-bright generations.

- `target_lf_mae`:
  - What: absolute error in the low-frequency energy ratio versus target-reference audio.
  - Why: catches thin or hollow generations.

- `target_dynamic_range_mae_db`:
  - What: absolute error in mel-domain dynamic range versus target-reference audio.
  - Why: catches over-compressed or overly flat outputs that sound synthetic.
