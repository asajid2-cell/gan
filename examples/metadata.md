# Example Metadata

## Diffusion examples

Files:
- `examples/audio/diffusion_v2_run_d002_epoch006_00_gen.wav`
- `examples/audio/diffusion_v2_run_d002_epoch006_01_gen.wav`
- `examples/audio/diffusion_v2_run_d002_epoch006_02_gen.wav`
- `examples/audio/diffusion_v2_run_d002_epoch006_03_gen.wav`

Provenance:
- Model family: diffusion V2 (v-prediction) + BigVGAN vocoding.
- Run: `saves2/lab3_diffusion/run_d002`
- Epoch: 6 (selected as best perceived quality checkpoint in practice).

Notes:
- These are **generated outputs only** (no ground-truth clips are included).
- These clips are intended as quick “does it sound plausible?” checks for collaborators.

## Codec transfer examples

Files:
- `examples/audio/codec_run1055_sample0000_src1_tgt3.wav`
- `examples/audio/codec_run1055_sample0004_src2_tgt1.wav`
- `examples/audio/codec_run1055_sample0008_src3_tgt0.wav`

Provenance:
- Model family: EnCodec latent translation + EnCodec decoder.
- Run: `saves2/lab3_codec_transfer/run1055`

Genre index convention:
- 0: `baroque_classical`
- 1: `cc0_other`
- 2: `hiphop_xtc`
- 3: `lofi_hh_lfbb`

Filename convention:
- `srcX_tgtY` indicates source genre index `X` and target genre index `Y`.

