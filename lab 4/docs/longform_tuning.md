# Long-Form Coherence Tuning (How-To)

This page explains the knobs in `lab 4/run_lab4_longform_coherence.py` in practical terms.

## Primary controls

### `--t-start` (SDEdit anchoring)

Higher `t-start`:
- more freedom to change timbre/texture
- more risk of drift/warble as duration increases

Lower `t-start`:
- stronger structural preservation
- smaller style change magnitude

### `--prefix-blend` (overlap locking)

Controls how strongly overlap frames are locked at every reverse step.

Higher:
- better seam coherence
- can “over-constrain” and reduce stylistic variation

Lower:
- more freedom per chunk
- more boundary artifacts

### `--reanchor-every` / `--reanchor-t-start`

Every N chunks, disable overlap lock and re-anchor to the source at a different `t-start`.
This prevents slow drift in long sequences.

## Anti-warble stabilization

### `--source-mel-blend`

Blends generated mel toward the source mel (global).
Useful when voice/instruments become permanently “gained” or phasey.

### `--hf-source-blend` and `--hf-start-bin`

Only blends high mel bins toward the source.
Useful when hiss/static accumulates but you still want low/mid-band style shift.

### `--mel-time-smooth` / `--mel-freq-smooth`

Applies simple smoothing to reduce jitter that tends to present as warble.

## Assembly mode

### `--assemble-domain mel`

Assembles the long mel first and runs BigVGAN once.
This can reduce per-chunk vocoder mismatch artifacts.

### `--assemble-domain audio`

Vocodes each chunk then crossfades audio.
Can be faster for iteration but may produce seam differences due to vocoder randomness/conditioning.
