@echo off
setlocal

set "REPO=Z:\328\CMPUT328-A2\codexworks\301\414-pl1"
set "OUTDIR=Z:\328\CMPUT328-A2\codexworks\301\414-pl1\lab 3.3\outputs\diffusion_generalized_5s_ood\run_queued_after_compare_20260327"
set "STDOUTLOG=Z:\328\CMPUT328-A2\codexworks\301\414-pl1\lab 3.3\outputs\diffusion_generalized_5s_ood\logs\train_queued_after_compare_20260327.out.log"
set "STDERRLOG=Z:\328\CMPUT328-A2\codexworks\301\414-pl1\lab 3.3\outputs\diffusion_generalized_5s_ood\logs\train_queued_after_compare_20260327.err.log"

cd /d "%REPO%"
if not exist "%OUTDIR%" mkdir "%OUTDIR%"
if not exist "%REPO%\lab 3.3\outputs\diffusion_generalized_5s_ood\logs" mkdir "%REPO%\lab 3.3\outputs\diffusion_generalized_5s_ood\logs"

set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONIOENCODING=utf-8"

python -u "%REPO%\lab 3.1\scripts\diffusion_longform_retool_train.py" ^
  --cache-dir "saves2\lab3_diffusion\run_d001\cache" ^
  --out-dir "%OUTDIR%" ^
  --bootstrap-checkpoint "saves2\lab3_diffusion\run_d002\checkpoints\epoch_006.pt" ^
  --epochs 6 ^
  --batch-size 1 ^
  --grad-accum 1 ^
  --max-frames 432 ^
  --lr 5e-5 ^
  --cfg-dropout-p 0.08 ^
  --identity-weight 1.0 ^
  --style-weight 1.6 ^
  --anchor-weight 0.40 ^
  --envelope-weight 0.20 ^
  --continuity-weight 0.80 ^
  --hf-penalty-weight 0.18 ^
  --vocal-weight 0.50 ^
  --crackle-weight 0.45 ^
  --anchor-bins 40 ^
  --hf-start-bin 56 ^
  --vocal-start-bin 10 ^
  --vocal-end-bin 42 ^
  --overlap-frames 64 ^
  --hf-margin 0.04 ^
  --crackle-margin 0.008 ^
  --style-probe-frames 256 ^
  --style-every-steps 2 ^
  --style-batch-splits 1 ^
  --max-batches-per-epoch 3500 ^
  --monitor-steps 25 ^
  --save-every-steps 100 ^
  --epoch-train-samples 3 ^
  --epoch-download-samples 4 ^
  --epoch-sample-ddim-steps 50 ^
  --epoch-sample-t-start 230 ^
  --epoch-sample-guidance-scale 1.8 ^
  --epoch-sample-style-strength 0.55 ^
  --source-aug-prob 0.80 ^
  --source-noise-std 0.020 ^
  --source-cond-noise-std 0.015 ^
  --source-global-offset-std 0.060 ^
  --source-hf-tilt-std 0.085 ^
  --source-time-mask-prob 0.35 ^
  --source-time-mask-frames 28 ^
  --resume ^
  --device auto 1>> "%STDOUTLOG%" 2>> "%STDERRLOG%"

endlocal
