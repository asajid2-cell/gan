# Lab 1 - Deconstruction Encoder

Lab 1 focuses on disentangling content from style and learning a reliable music gate.

## What This Folder Contains

- `notebooks/01_lab1_deconstruction_encoder.ipynb`: main Lab 1 training and audit notebook.
- `scripts/render_phase1_symbolic.py`: symbolic-data rendering helper used for Phase 1 corpus preparation.
- `docs/lab1_deconstruction_encoder.md`: Lab 1 explanation and success criteria.
- `docs/lab1_results_demo.pdf`: compact Lab 1 quantitative results summary for the demo.

## Purpose

Lab 1 is the representation-learning stage of the project. Its job is to produce:
- `z_content`: structure-preserving content representation
- `z_style`: style-bearing latent representation
- a music gate for separating music from non-music negatives

Downstream labs depend on this stage being auditable and stable, so Lab 1 is organized around both training and validation.
