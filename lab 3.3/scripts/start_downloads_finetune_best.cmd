@echo off
setlocal EnableExtensions

set "REPO_ROOT=%~dp0..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
cd /d "%REPO_ROOT%"

if "%~1"=="" (
  for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TAG=%%I"
) else (
  set "TAG=%~1"
)

set "OUT_DIR=%REPO_ROOT%\lab 3.3\outputs\diffusion_downloads_best_finetune\run_%TAG%"
set "LOG_DIR=%REPO_ROOT%\lab 3.3\outputs\diffusion_downloads_best_finetune\logs"
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "STDOUT_LOG=%LOG_DIR%\train_%TAG%.out.log"
set "STDERR_LOG=%LOG_DIR%\train_%TAG%.err.log"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "PYTHONIOENCODING=utf-8"

echo {^
  "tag": "%TAG%",^
  "out_dir": "%OUT_DIR:\=\\%",^
  "stdout_log": "%STDOUT_LOG:\=\\%",^
  "stderr_log": "%STDERR_LOG:\=\\%",^
  "bootstrap_checkpoint": "saves2\\lab3_diffusion\\run_d002\\checkpoints\\best.pt",^
  "source_mode": "downloads"^
} > "%OUT_DIR%\launch_meta.json"

start "" /b cmd /c python -u "%REPO_ROOT%\lab 3.1\scripts\diffusion_longform_retool_train.py" ^
  --cache-dir "saves2\lab3_diffusion\run_d001\cache" ^
  --out-dir "%OUT_DIR%" ^
  --bootstrap-checkpoint "saves2\lab3_diffusion\run_d002\checkpoints\best.pt" ^
  --epochs 4 ^
  --batch-size 1 ^
  --grad-accum 1 ^
  --max-frames 432 ^
  --lr 5e-5 ^
  --identity-weight 1.0 ^
  --style-weight 1.6 ^
  --anchor-weight 0.45 ^
  --envelope-weight 0.22 ^
  --continuity-weight 0.72 ^
  --hf-penalty-weight 0.22 ^
  --vocal-weight 0.52 ^
  --crackle-weight 0.36 ^
  --anchor-bins 40 ^
  --hf-start-bin 56 ^
  --vocal-start-bin 10 ^
  --vocal-end-bin 42 ^
  --overlap-frames 64 ^
  --hf-margin 0.05 ^
  --crackle-margin 0.010 ^
  --style-probe-frames 256 ^
  --style-every-steps 2 ^
  --style-batch-splits 1 ^
  --max-batches-per-epoch 600 ^
  --monitor-steps 25 ^
  --save-every-steps 50 ^
  --epoch-train-samples 0 ^
  --epoch-download-samples 5 ^
  --epoch-sample-ddim-steps 50 ^
  --epoch-sample-t-start 225 ^
  --epoch-sample-guidance-scale 1.8 ^
  --epoch-sample-style-strength 0.58 ^
  --source-mode downloads ^
  --downloads-source-samples-per-epoch 900 ^
  --source-aug-prob 0.20 ^
  --source-noise-std 0.003 ^
  --source-cond-noise-std 0.003 ^
  --source-global-offset-std 0.015 ^
  --source-hf-tilt-std 0.020 ^
  --source-time-mask-prob 0.10 ^
  --source-time-mask-frames 12 ^
  --resume ^
  --device auto ^
  1>"%STDOUT_LOG%" 2>"%STDERR_LOG%"

echo Started downloads fine-tune
echo Tag: %TAG%
echo OutDir: %OUT_DIR%
echo Stdout: %STDOUT_LOG%
echo Stderr: %STDERR_LOG%
