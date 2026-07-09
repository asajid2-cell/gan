# Train Diffusion V2/V3 (How-To)

Goal: train mel-domain diffusion for higher-magnitude perceptual changes and long-form control.

Entrypoints:
- V2: `lab 3/run_lab3_diffusion_v2.py`
- V3: `lab 3/run_lab3_diffusion_v3.py` (adversarial fine-tuning)

Notes:
- V2 trains with v-prediction MSE and EMA.
- V3 adds a mel discriminator (hinge + feature matching).

