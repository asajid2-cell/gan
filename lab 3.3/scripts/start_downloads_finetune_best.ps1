param(
    [string]$Tag = $(Get-Date -Format "yyyyMMdd_HHmmss"),
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
$trainScript = Join-Path $repoRoot "lab 3.1\scripts\diffusion_longform_retool_train.py"

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:PYTHONIOENCODING = "utf-8"

$outDir = Join-Path $repoRoot ("lab 3.3\outputs\diffusion_downloads_best_finetune\run_{0}" -f $Tag)
$logDir = Join-Path $repoRoot "lab 3.3\outputs\diffusion_downloads_best_finetune\logs"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdoutLog = Join-Path $logDir ("train_{0}.out.log" -f $Tag)
$stderrLog = Join-Path $logDir ("train_{0}.err.log" -f $Tag)

$cmdLine = @(
    'python -u "{0}"' -f $trainScript,
    '--cache-dir "saves2\lab3_diffusion\run_d001\cache"',
    '--out-dir "{0}"' -f $outDir,
    '--bootstrap-checkpoint "saves2\lab3_diffusion\run_d002\checkpoints\best.pt"',
    '--epochs 4',
    '--batch-size 1',
    '--grad-accum 1',
    '--max-frames 432',
    '--lr 5e-5',
    '--identity-weight 1.0',
    '--style-weight 1.6',
    '--anchor-weight 0.45',
    '--envelope-weight 0.22',
    '--continuity-weight 0.72',
    '--hf-penalty-weight 0.22',
    '--vocal-weight 0.52',
    '--crackle-weight 0.36',
    '--anchor-bins 40',
    '--hf-start-bin 56',
    '--vocal-start-bin 10',
    '--vocal-end-bin 42',
    '--overlap-frames 64',
    '--hf-margin 0.05',
    '--crackle-margin 0.010',
    '--style-probe-frames 256',
    '--style-every-steps 2',
    '--style-batch-splits 1',
    '--max-batches-per-epoch 600',
    '--monitor-steps 25',
    '--save-every-steps 50',
    '--epoch-train-samples 0',
    '--epoch-download-samples 5',
    '--epoch-sample-ddim-steps 50',
    '--epoch-sample-t-start 225',
    '--epoch-sample-guidance-scale 1.8',
    '--epoch-sample-style-strength 0.58',
    '--source-mode downloads',
    '--downloads-source-samples-per-epoch 900',
    '--source-aug-prob 0.20',
    '--source-noise-std 0.003',
    '--source-cond-noise-std 0.003',
    '--source-global-offset-std 0.015',
    '--source-hf-tilt-std 0.020',
    '--source-time-mask-prob 0.10',
    '--source-time-mask-frames 12',
    '--resume',
    '--device auto',
    '1>"{0}"' -f $stdoutLog,
    '2>"{0}"' -f $stderrLog
) -join ' '

$meta = [ordered]@{
    tag = $Tag
    out_dir = $outDir
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
    bootstrap_checkpoint = "saves2\lab3_diffusion\run_d002\checkpoints\best.pt"
    source_mode = "downloads"
}
($meta | ConvertTo-Json -Depth 4) | Set-Content -Encoding UTF8 (Join-Path $outDir "launch_meta.json")

$proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmdLine) -WorkingDirectory $repoRoot -PassThru

Write-Output ("Started downloads fine-tune. PID={0}" -f $proc.Id)
Write-Output ("OutDir: {0}" -f $outDir)
Write-Output ("Stdout: {0}" -f $stdoutLog)
Write-Output ("Stderr: {0}" -f $stderrLog)
