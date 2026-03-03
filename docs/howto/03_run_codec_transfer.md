# Run Lab 3 Codec Transfer (How-To)

Goal: perform short-form style transfer in EnCodec latent space with content preservation.

Entry point:
- `lab 3/run_lab3_codec.py`

Key concepts:
- Conditioning can use centroid/exemplar banks.
- Best observed configuration uses MERT probe embeddings for conditioning and direct-output translation.

Outputs:
- `codec_gate_eval.json` (MPS, style metrics)
- exported sample WAVs (if enabled)

