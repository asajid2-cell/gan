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

set "OUT_DIR=%REPO_ROOT%\lab 3.3\outputs\diffusion_stage1_warble_mixed\run_%TAG%"
set "LOG_DIR=%REPO_ROOT%\lab 3.3\outputs\diffusion_stage1_warble_mixed\logs"
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
  "source_mode": "mixed",^
  "downloads_mix_ratio": 0.30^
} > "%OUT_DIR%\launch_meta.json"

start "" /b cmd /c python -u "%REPO_ROOT%\lab 3.1\scripts\diffusion_longform_retool_train.py" ^
  --cache-dir "saves2\lab3_diffusion\run_d001\cache" ^
  --out-dir "%OUT_DIR%" ^
  --bootstrap-checkpoint "saves2\lab3_diffusion\run_d002\checkpoints\best.pt" ^
  --epochs 2 ^
  --batch-size 1 ^
  --grad-accum 1 ^
  --max-frames 384 ^
  --lr 2.5e-5 ^
  --identity-weight 1.0 ^
  --style-weight 1.45 ^
  --anchor-weight 0.40 ^
  --envelope-weight 0.18 ^
  --continuity-weight 0.62 ^
  --hf-penalty-weight 0.14 ^
  --vocal-weight 0.34 ^
  --crackle-weight 0.18 ^
  --anchor-bins 40 ^
  --hf-start-bin 56 ^
  --vocal-start-bin 10 ^
  --vocal-end-bin 42 ^
  --overlap-frames 64 ^
  --hf-margin 0.05 ^
  --crackle-margin 0.010 ^
  --style-probe-frames 128 ^
  --style-every-steps 8 ^
  --style-batch-splits 1 ^
  --max-batches-per-epoch 300 ^
  --monitor-steps 25 ^
  --save-every-steps 50 ^
  --epoch-train-samples 2 ^
  --epoch-download-samples 4 ^
  --epoch-sample-ddim-steps 40 ^
  --epoch-sample-t-start 220 ^
  --epoch-sample-guidance-scale 1.78 ^
  --epoch-sample-style-strength 0.54 ^
  --source-mode mixed ^
  --downloads-source-samples-per-epoch 700 ^
  --mixed-source-samples-per-epoch 1200 ^
  --downloads-mix-ratio 0.30 ^
  --source-aug-prob 0.08 ^
  --source-noise-std 0.0015 ^
  --source-cond-noise-std 0.0015 ^
  --source-global-offset-std 0.010 ^
  --source-hf-tilt-std 0.012 ^
  --source-time-mask-prob 0.05 ^
  --source-time-mask-frames 8 ^
  --resume ^
  --device auto ^
  1>"%STDOUT_LOG%" 2>"%STDERR_LOG%"

echo Started stage1 warble mixed fine-tune
echo Tag: %TAG%
echo OutDir: %OUT_DIR%
echo Stdout: %STDOUT_LOG%
echo Stderr: %STDERR_LOG%
