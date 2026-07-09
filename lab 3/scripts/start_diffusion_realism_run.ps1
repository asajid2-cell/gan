param(
    [string]$RunName = "run_d004",
    [int]$SweepSamples = 8,
    [int]$SweepWriteAudioCount = 2
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

function Quote-CmdArg([string]$Value) {
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '\"') + '"'
    }
    return $Value
}

$outDir = Join-Path $repoRoot ("saves2\lab3_diffusion\" + $RunName)
$logDir = Join-Path $outDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trainLog = Join-Path $logDir "train_diffusion_realism.log"
$sweepLog = Join-Path $logDir "realism_sweep.log"

if (Test-Path (Join-Path $outDir "v3_config.json")) {
    throw "Refusing to overwrite existing diffusion run: $RunName"
}

Write-Host "[diffusion] starting realism-biased fine-tune for $RunName"
Write-Host "[diffusion] logs: $trainLog"

$trainArgs = @(
    "python", "-u", "lab 3/run_lab3_diffusion_v3.py",
    "--out-dir", ("saves2/lab3_diffusion/" + $RunName),
    "--v2-checkpoint", "saves2/lab3_diffusion/run_d002/checkpoints/epoch_006.pt",
    "--restart",
    "--epochs", "6",
    "--lr", "5e-5",
    "--disc-lr", "1e-4",
    "--batch-size", "4",
    "--grad-accum", "4",
    "--ema-decay", "0.999",
    "--cfg-dropout-p", "0.12",
    "--disc-warmup-steps", "1200",
    "--adv-weight", "0.05",
    "--fm-weight", "1.00",
    "--epoch-samples", "6",
    "--ddim-steps", "50",
    "--guidance-scale", "2.0"
)
$trainCmd = (($trainArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $trainLog) + " 2>&1"
Add-Content -Path $trainLog -Value ("[diffusion train] " + $trainCmd)
cmd /c $trainCmd

if ($LASTEXITCODE -ne 0) {
    throw "Diffusion training failed for $RunName. See $trainLog"
}

Write-Host "[diffusion] training finished, starting realism sweep"
Write-Host "[diffusion] sweep log: $sweepLog"

$sweepArgs = @(
    "python", "-u", "lab 3/run_lab3_realism_sweep.py", "diffusion",
    "--run-dir", ("saves2/lab3_diffusion/" + $RunName),
    "--include-all-epochs",
    "--n-samples", "$SweepSamples",
    "--write-audio-count", "$SweepWriteAudioCount"
)
$sweepCmd = (($sweepArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $sweepLog) + " 2>&1"
Add-Content -Path $sweepLog -Value ("[diffusion sweep] " + $sweepCmd)
cmd /c $sweepCmd

if ($LASTEXITCODE -ne 0) {
    throw "Diffusion realism sweep failed for $RunName. See $sweepLog"
}

Write-Host "[diffusion] finished: $RunName"
