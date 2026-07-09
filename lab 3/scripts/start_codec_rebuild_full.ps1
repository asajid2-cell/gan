param(
    [string]$RunName = "run1058_rebuild",
    [string]$BootstrapCkpt = "saves2/lab3_codec_transfer/run1056/checkpoints/stage1_latest.pt",
    [string]$EpochSampleSourceFile = "C:\Users\Ahmed\Downloads\Milky & Mall Grab - Just The Way You Are.flac",
    [double]$EpochSampleOffsetSec = 0.0,
    [int]$Stage2Epochs = 10,
    [int]$Stage3Epochs = 8,
    [int]$SweepSamples = 16,
    [int]$SweepWriteAudioCount = 4
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

$trainLog = Join-Path $logDir "train_codec_rebuild_full.log"
$sweepLog = Join-Path $logDir "realism_sweep_full.log"

Write-Host "[codec-full] starting late-stage rebuild run for $RunName"
Write-Host "[codec-full] bootstrap checkpoint: $BootstrapCkpt"

$trainArgs = @(
    "python", "-u", "lab 3/run_lab3_codec.py",
    "--run-name", $RunName,
    "--force-custom-run-name",
    "--reuse-cache-dir", "saves2/lab3_codec_transfer/run1051/cache",
    "--bootstrap-ckpt", $BootstrapCkpt,
    "--skip-stage1",
    "--style-cond-source", "mert_probe_embed",
    "--style-loss-mode", "mert_probe_ce",
    "--translator-direct-output",
    "--translator-direct-mix", "0.45",
    "--batch-size", "6",
    "--stage2-epochs", "$Stage2Epochs",
    "--stage3-epochs", "$Stage3Epochs",
    "--stage2-cond-mode", "exemplar",
    "--stage3-cond-mode", "exemplar",
    "--stage2-cond-alpha-start", "0.55",
    "--stage2-cond-alpha-end", "0.18",
    "--stage3-cond-alpha-start", "0.18",
    "--stage3-cond-alpha-end", "0.03",
    "--stage2-adv-weight", "0.52",
    "--stage3-adv-weight", "0.68",
    "--stage2-style-weight", "9.5",
    "--stage3-style-weight", "11.5",
    "--stage2-content-weight", "2.4",
    "--stage3-content-weight", "1.9",
    "--stage2-mrstft-weight", "0.10",
    "--stage3-mrstft-weight", "0.02",
    "--stage2-latent-l1-weight", "0.12",
    "--stage3-latent-l1-weight", "0.02",
    "--stage2-delta-budget", "0.18",
    "--stage3-delta-budget", "0.22",
    "--stage2-delta-budget-weight", "0.0",
    "--stage3-delta-budget-weight", "0.25",
    "--stage2-style-dropout-p", "0.08",
    "--stage3-style-dropout-p", "0.20",
    "--stage2-style-jitter-std", "0.04",
    "--stage3-style-jitter-std", "0.08",
    "--stage2-exemplar-noise-std", "0.03",
    "--stage3-exemplar-noise-std", "0.05",
    "--stage2-style-embed-align-weight", "0.35",
    "--stage3-style-embed-align-weight", "0.60",
    "--stage2-generated-mert-weight", "0.55",
    "--stage3-generated-mert-weight", "0.75",
    "--stage2-generated-mert-align-weight", "0.20",
    "--stage3-generated-mert-align-weight", "0.30",
    "--stage2-generated-mert-every", "4",
    "--stage3-generated-mert-every", "2",
    "--stage3-mode-seeking-weight", "0.05",
    "--stage3-mode-seeking-target", "0.03",
    "--sample-count", "20",
    "--sample-export-tag", "rebuild_full_samples",
    "--epoch-sample-source-file", $EpochSampleSourceFile,
    "--epoch-sample-offset-sec", "$EpochSampleOffsetSec",
    "--epoch-sample-every", "1",
    "--epoch-sample-tag", "epoch_samples"
)

$trainCmd = (($trainArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $trainLog) + " 2>&1"
Add-Content -Path $trainLog -Value ("[codec full train] " + $trainCmd)
cmd /c $trainCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec rebuild full run failed for $RunName. See $trainLog"
}

Write-Host "[codec-full] training finished, starting realism sweep"

$sweepArgs = @(
    "python", "-u", "lab 3/run_lab3_realism_sweep.py", "codec",
    "--run-dir", ("saves2/lab3_codec_transfer/" + $RunName),
    "--n-samples", "$SweepSamples",
    "--write-audio-count", "$SweepWriteAudioCount",
    "--max-fad-mert", "28",
    "--min-mps", "0.94",
    "--min-style-target-acc", "0.18",
    "--min-style-target-cos", "0.02"
)
$sweepCmd = (($sweepArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $sweepLog) + " 2>&1"
Add-Content -Path $sweepLog -Value ("[codec full sweep] " + $sweepCmd)
cmd /c $sweepCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec rebuild full realism sweep failed for $RunName. See $sweepLog"
}

Write-Host "[codec-full] finished: $RunName"
