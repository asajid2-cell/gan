# Lab 1: Deconstruction Encoder (Explanation)

## Goal

Learn disentangled representations of short audio chunks:
- `z_content`: invariant musical structure (melody/rhythm/harmony proxy)
- `z_style`: genre-dependent style markers (instrumentation/timbre/performance cues)

Additionally learn a `music_gate` head to reject non-music segments.

## Architecture (implemented)

Input:
- log-mel spectrogram (96 mel bins) for a fixed-length chunk (typically 5 seconds at 22.05 kHz).

Model:
- Conv2D backbone (3 blocks, stride-2) -> global average pooling -> shared linear layer.
- Two heads:
  - `z_content` head (128D, L2-normalized)
  - `z_style` head (128D, L2-normalized)
- Classifiers:
  - `style_cls(z_style)` for source/genre supervision
  - `content_style_adv(GRL(z_content))` to adversarially remove style from content
  - `music_head(shared)` for music vs non-music gating

## Losses (high level)

Let `a` and `b` be two augmented views of the same chunk.

- Content invariance:
  - `L_content = MSE(z_c(a), z_c(b))`
- Style classification:
  - `L_style = CE(style_cls(z_s(a)), y_source)`
- Adversarial style leakage removal:
  - `L_adv = CE(content_style_adv(GRL(z_c(a))), y_source)`
- Gate:
  - `L_gate = BCEWithLogits(music_logit(a), y_music)`
- Optional teacher anchor in phase-3 sharpening:
  - `L_anchor = MSE(z_c(a), z_c_teacher(a))` (and same for `b`)

Total is a weighted sum with phase-specific schedules.

## Curriculum training (why it matters)

We use a 3-phase curriculum to prevent early collapse:
1. Phase 1: content-focused separation
2. Phase 2: increase adversarial pressure for disentanglement
3. Phase 3: sharpen music gate and optionally stabilize with a teacher anchor

## Audit criteria

Lab 1 is considered complete when it passes:
- style accuracy threshold (style-bearing head works)
- content leakage threshold (content head is style-suppressed)
- gate AUC threshold (music discrimination works)

See `docs/explanation/results.md` for the achieved values.

