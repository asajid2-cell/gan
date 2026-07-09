# Deep Generative Genre Remastering (DGGR)

This repository contains a multi-stage music style transfer system designed to move beyond superficial audio-to-audio "repainting" and toward genre remastering: deconstruct a track into content vs style, calibrate a target style space, and reconstruct audio conditioned on the target genre.

Repo structure:
- `dggr/`: canonical shared Python package used across the later labs.
- `lab 1/` through `lab 5/`: lab-segmented folders for notebooks, run scripts, and lab-specific docs.
- `lab 3.1/`: notebook-first clean-slate reset workspace for the next Lab 3/4 cycle.
- `docs/`: shared documentation, reference material, and architecture notes.
- `gui/`: local inference GUI and generated GUI output bundles.
- `presentation/`: final presentation deliverables.

Quick links:
- Lab notebook index: `notebooks/README.md`
- Clean-slate reset workspace: `lab 3.1/README.md`
- Shared documentation home: `docs/README.md`
- Final presentation deck: `presentation/dggr_lecture_deck.pptx`
- Final presentation PDF: `presentation/dggr_lecture_deck.pdf`
- Final speaker script PDF: `presentation/dggr_speaker_script.pdf`
- GUI guide: `gui/README.md`

## Running

Most training and inference entrypoints are now organized by lab:
- Lab 1 symbolic/render prep: `lab 1/scripts/render_phase1_symbolic.py`
- Lab 2: `lab 2/run_lab2.py`
- Lab 3 codec transfer: `lab 3/run_lab3_codec.py`
- Lab 3 diffusion training: `lab 3/run_lab3_diffusion_v2.py`
- Lab 4 long-form coherence: `lab 4/run_lab4_longform_coherence.py`

See the per-lab folders plus `docs/` for copy-paste commands and shared reference material.

## Clean-Slate Reset

The repo's current diagnosis is that Labs 1 and 2 are mostly healthy, while Lab 3 is where the
main mismatch appears: internal style/content gates can look strong even when generated-audio
realism and target-style movement are weak.

The notebook-first reset path for that work now lives in:

- `lab 3.1/README.md`
- `lab 3.1/notebooks/00_pipeline_audit.ipynb`
- `lab 3.1/docs/pipeline_diagnosis.md`

To refresh the cross-run audit tables used by those notebooks:

```powershell
python "lab 3.1/scripts/pipeline_audit.py"
```

For the current late-stage codec tuning workflow, start with:

```powershell
powershell -ExecutionPolicy Bypass -File "lab 3/scripts/start_codec_rebuild_probe.ps1"
```

That launcher reuses a stable stage 1 checkpoint, tunes only stage 2/3, exports fixed-song epoch samples, and runs the realism supervisor before you commit to a longer run.

## Local GUI

To launch the local DGGR inference GUI:

```powershell
python run_gui.py
```

or on Windows:

```powershell
.\run_gui.ps1
```

The GUI supports:
- uploading a song or selecting an example clip
- side-by-side codec vs diffusion comparison
- codec-only inference
- diffusion-only inference
- Lab 4 long-form coherence inference
- run browsing and model-cache management
- a bottom terminal panel for device and progress logs during long runs
