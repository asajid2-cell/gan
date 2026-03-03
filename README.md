# Deep Generative Genre Remastering (DGGR)

This repository contains a multi-stage music style transfer system designed to move beyond superficial audio-to-audio "repainting" and toward **genre remastering**: deconstruct a track into content vs style, calibrate a target style space, and reconstruct audio conditioned on the target genre.

Repo structure (collaboration-friendly):
- `dggr/`: canonical Python package (Labs 2–4 code lives here).
- `lab 2/` and `lab 3/`: original lab folders (kept as runnable entrypoints).
- `notebooks/`: cleaned notebooks (outputs stripped for git).
- `docs/`: comprehensive documentation (organized using the Diataxis framework).

Quick links:
- Notebooks index: `notebooks/README.md`
- Documentation home: `docs/README.md`

## Running

Most training/inference entrypoints are in the original lab folders:
- Lab 2: `lab 2/run_lab2.py`
- Lab 3 codec transfer: `lab 3/run_lab3_codec.py`
- Diffusion training: `lab 3/run_lab3_diffusion_v2.py`
- Long-form coherence: `lab 3/run_lab4_longform_coherence.py`

See `docs/tutorials/` for copy-paste commands and recommended presets.
