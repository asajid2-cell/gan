# DGGR Inference Studio

`DGGR Inference Studio` is the local GUI for running short-form and long-form inference from the saved DGGR checkpoints.

## Launch

From the repo root:

```powershell
python run_gui.py
```

Windows shortcut:

```powershell
.\run_gui.ps1
```

The launcher opens a local Gradio app in your browser.

## What it supports

- Upload a song or use one of the example clips.
- Compare short-form codec transfer and short-form diffusion transfer side by side.
- Run codec-only short-form inference.
- Run diffusion-only short-form inference.
- Run codec long-form chunked transfer directly in the GUI.
- Run diffusion long-form coherence through the Lab 4 runner with full continuity controls.
- Compare codec long-form and diffusion long-form on the same source excerpt.
- Browse available saved runs and clear loaded model caches.
- Watch a live terminal panel at the bottom of the app for device selection, model loading, and run progress.
- Choose explicit checkpoints/epochs within each saved run instead of being limited to one default checkpoint.
- Select from both legacy `saves2` runs and newer `lab 3.1` overnight/reset runs.

## Notes

- The GUI now defaults to the newest discovered codec and diffusion runs so overnight/reset experiments appear immediately.
- Output bundles are written to `gui/outputs/` and zipped automatically for easy inspection.
- The first diffusion run will load BigVGAN and the saved diffusion checkpoint, so it may take noticeably longer than subsequent runs.
- The app is usable on CPU, but the intended path is `cuda` when a GPU is available.
