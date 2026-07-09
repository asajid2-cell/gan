# General Field Presentation Outline

## Core thesis

Move the talk away from "a survey of old audio generators" and toward a clearer systems story:

- weak systems repaint timbre
- stronger systems separate content from style
- modern systems need controllable latent representations and high-fidelity synthesis
- true genre remastering still depends on long-form coherence and honest evaluation

## Slide sequence

1. Title: from coat-of-paint transfer to genre remastering
2. Talk map: five-part structure and speaker handoff
3. Why the problem is hard
4. Content vs style vs genre
5. What success looks like: content, style, realism, coherence
6. Representation ladder: waveform to spectral to symbolic to learned latent
7. Historical arc: WaveNet, WaveGAN, GANSynth, disentanglement, RAVE, diffusion
8. Disentanglement slide: core equation and why factorization matters
9. Disentanglement examples: Hung, Yang, and MoVE
10. RAVE slide: why modern latent audio matters
11. Diffusion slide: DDPM, CFG, SDEdit, AudioLDM
12. Field-level design rules: what strong systems now need
13. DGGR bridge: how our pipeline maps onto the field
14. DGGR diagnosis: codec branch vs diffusion branch
15. Closing: what we should focus on next

## Speaker split

- Kelsey: slides 1 to 5
- Sahara: slides 6 to 9
- Ahmed: slides 10 to 14

## Design intent

- fewer slides, stronger transitions
- vector-style diagrams instead of dense paragraphs
- equations only where they clarify the main idea
- each slide answers one question clearly
