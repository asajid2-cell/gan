param(
    [string]$RunName = "run1056",
    [int]$SweepSamples = 12,
    [int]$SweepWriteAudioCount = 3,
    [string]$EpochSampleSourceFile = "C:\Users\Ahmed\Downloads\Milky & Mall Grab - Just The Way You Are.flac",
    [double]$EpochSampleOffsetSec = 0.0
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

$outDir = Join-Path $repoRoot ("saves2\lab3_codec_transfer\" + $RunName)
$logDir = Join-Path $outDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trainLog = Join-Path $logDir "train_codec_realism.log"
$sweepLog = Join-Path $logDir "realism_sweep.log"

$isResume = Test-Path (Join-Path $outDir "run_state.json")

Write-Host ("[codec] " + ($(if ($isResume) { "resuming" } else { "starting" })) + " realism-biased training for $RunName")
Write-Host "[codec] logs: $trainLog"

$trainArgs = @(
    "python", "-u", "lab 3/run_lab3_codec.py",
    "--reuse-cache-dir", "saves2/lab3_codec_transfer/run1051/cache",
    "--style-cond-source", "mert_probe_embed",
    "--style-loss-mode", "mert_probe_ce",
    "--batch-size", "6",
    "--stage1-epochs", "8",
    "--stage2-epochs", "16",
    "--stage3-epochs", "10",
    "--stage2-cond-mode", "exemplar",
    "--stage3-cond-mode", "exemplar",
    "--stage2-cond-alpha-start", "0.75",
    "--stage2-cond-alpha-end", "0.45",
    "--stage3-cond-alpha-start", "0.45",
    "--stage3-cond-alpha-end", "0.25",
    "--stage2-adv-weight", "0.40",
    "--stage3-adv-weight", "0.45",
    "--stage2-style-weight", "7.0",
    "--stage3-style-weight", "8.0",
    "--stage2-content-weight", "3.0",
    "--stage3-content-weight", "2.5",
    "--stage2-mrstft-weight", "0.80",
    "--stage3-mrstft-weight", "0.50",
    "--stage2-latent-l1-weight", "0.60",
    "--stage3-latent-l1-weight", "0.30",
    "--stage2-delta-budget", "0.10",
    "--stage3-delta-budget", "0.08",
    "--stage2-delta-budget-weight", "2.0",
    "--stage3-delta-budget-weight", "3.0",
    "--stage2-style-dropout-p", "0.00",
    "--stage3-style-dropout-p", "0.10",
    "--stage2-style-jitter-std", "0.02",
    "--stage3-style-jitter-std", "0.04",
    "--stage2-exemplar-noise-std", "0.02",
    "--stage3-exemplar-noise-std", "0.03",
    "--stage2-style-embed-align-weight", "0.40",
    "--stage3-style-embed-align-weight", "0.50",
    "--stage3-mode-seeking-weight", "0.02",
    "--stage3-mode-seeking-target", "0.02",
    "--sample-count", "16",
    "--sample-export-tag", "realism_samples",
    "--epoch-sample-source-file", $EpochSampleSourceFile,
    "--epoch-sample-offset-sec", "$EpochSampleOffsetSec",
    "--epoch-sample-every", "1",
    "--epoch-sample-tag", "epoch_samples"
)
if ($isResume) {
    $trainArgs += @("--mode", "resume", "--resume-dir", ("saves2/lab3_codec_transfer/" + $RunName))
} else {
    $trainArgs += @("--run-name", $RunName)
}
$trainCmd = (($trainArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $trainLog) + " 2>&1"
Add-Content -Path $trainLog -Value ("[codec train] " + $trainCmd)
cmd /c $trainCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec training failed for $RunName. See $trainLog"
}

Write-Host "[codec] training finished, starting realism sweep"
Write-Host "[codec] sweep log: $sweepLog"

$sweepArgs = @(
    "python", "-u", "lab 3/run_lab3_realism_sweep.py", "codec",
    "--run-dir", ("saves2/lab3_codec_transfer/" + $RunName),
    "--n-samples", "$SweepSamples",
    "--write-audio-count", "$SweepWriteAudioCount"
)
$sweepCmd = (($sweepArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $sweepLog) + " 2>&1"
Add-Content -Path $sweepLog -Value ("[codec sweep] " + $sweepCmd)
cmd /c $sweepCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec realism sweep failed for $RunName. See $sweepLog"
}

Write-Host "[codec] finished: $RunName"
