# Codec vs Diffusion (Explanation)

DGGR currently has two reconstruction approaches.

## Codec latent translation (EnCodec)

What it does:
- Encodes audio into EnCodec embeddings and learns a translator `q_src -> q_hat`.

Why it works well:
- The pretrained EnCodec decoder acts as a high-quality “physics prior”, preventing many forms of raw-audio instability.
- Style transfer becomes a constrained latent editing problem, which is easier than waveform synthesis from scratch.

Limitations:
- The codec representation can bottleneck the magnitude of perceptual change.
- Very strong edits can induce “identity leakage” or introduce characteristic codec artifacts.

## Diffusion in mel space

What it does:
- Generates BigVGAN-compatible mel spectrograms conditioned on content and style embeddings.
- Uses CFG and SDEdit-style anchoring for controlled edits.

Why it is valuable:
- It can enable stronger perceptual remastering than codec translation (in principle).
- Conditioning and sampling controls can explicitly trade off content preservation vs style change.

Limitations:
- Long-form generation is challenging; drift/warble/static can accumulate across chunks.
- Requires careful vocoder integration (BigVGAN) and coherence constraints (prefix locking, re-anchoring).

