param(
    [string]$RunName = "run1057_probe",
    [string]$BootstrapCkpt = "saves2/lab3_codec_transfer/run1056/checkpoints/stage1_latest.pt",
    [string]$EpochSampleSourceFile = "C:\Users\Ahmed\Downloads\Milky & Mall Grab - Just The Way You Are.flac",
    [double]$EpochSampleOffsetSec = 0.0,
    [int]$Stage2Epochs = 3,
    [int]$Stage3Epochs = 3,
    [int]$MaxBatchesPerEpoch = 96,
    [int]$SweepSamples = 12,
    [int]$SweepWriteAudioCount = 4,
    [double]$TranslatorDirectMix = 0.45,
    [double]$Stage2AdvWeight = 0.52,
    [double]$Stage3AdvWeight = 0.68,
    [double]$Stage2StyleWeight = 9.5,
    [double]$Stage3StyleWeight = 11.5,
    [double]$Stage2GeneratedMertWeight = 0.55,
    [double]$Stage3GeneratedMertWeight = 0.75,
    [double]$Stage2GeneratedMertAlignWeight = 0.20,
    [double]$Stage3GeneratedMertAlignWeight = 0.30,
    [double]$Stage2LatentL1Weight = 0.12,
    [double]$Stage3LatentL1Weight = 0.02,
    [double]$Stage2MrstftWeight = 0.10,
    [double]$Stage3MrstftWeight = 0.02,
    [double]$Stage2ContentWeight = 2.4,
    [double]$Stage3ContentWeight = 1.9,
    [double]$Stage2DeltaBudgetWeight = 0.0,
    [double]$Stage3DeltaBudgetWeight = 0.25,
    [double]$Stage2DeltaBudget = 0.18,
    [double]$Stage3DeltaBudget = 0.22,
    [double]$Stage2StyleDropoutP = 0.08,
    [double]$Stage3StyleDropoutP = 0.20,
    [double]$Stage2StyleJitterStd = 0.04,
    [double]$Stage3StyleJitterStd = 0.08,
    [double]$Stage2ExemplarNoiseStd = 0.03,
    [double]$Stage3ExemplarNoiseStd = 0.05,
    [double]$Stage2StyleEmbedAlignWeight = 0.35,
    [double]$Stage3StyleEmbedAlignWeight = 0.60,
    [int]$Stage2GeneratedMertEvery = 4,
    [int]$Stage3GeneratedMertEvery = 2,
    [double]$Stage2CondAlphaStart = 0.55,
    [double]$Stage2CondAlphaEnd = 0.20,
    [double]$Stage3CondAlphaStart = 0.20,
    [double]$Stage3CondAlphaEnd = 0.05,
    [double]$Stage3ModeSeekingWeight = 0.05,
    [double]$Stage3ModeSeekingTarget = 0.03
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

$trainLog = Join-Path $logDir "train_codec_rebuild_probe.log"
$sweepLog = Join-Path $logDir "realism_sweep_probe.log"

Write-Host "[codec-probe] starting late-stage rebuild probe for $RunName"
Write-Host "[codec-probe] bootstrap checkpoint: $BootstrapCkpt"

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
    "--translator-direct-mix", "$TranslatorDirectMix",
    "--batch-size", "6",
    "--stage2-epochs", "$Stage2Epochs",
    "--stage3-epochs", "$Stage3Epochs",
    "--max-batches-per-epoch", "$MaxBatchesPerEpoch",
    "--stage2-cond-mode", "exemplar",
    "--stage3-cond-mode", "exemplar",
    "--stage2-cond-alpha-start", "$Stage2CondAlphaStart",
    "--stage2-cond-alpha-end", "$Stage2CondAlphaEnd",
    "--stage3-cond-alpha-start", "$Stage3CondAlphaStart",
    "--stage3-cond-alpha-end", "$Stage3CondAlphaEnd",
    "--stage2-adv-weight", "$Stage2AdvWeight",
    "--stage3-adv-weight", "$Stage3AdvWeight",
    "--stage2-style-weight", "$Stage2StyleWeight",
    "--stage3-style-weight", "$Stage3StyleWeight",
    "--stage2-content-weight", "$Stage2ContentWeight",
    "--stage3-content-weight", "$Stage3ContentWeight",
    "--stage2-mrstft-weight", "$Stage2MrstftWeight",
    "--stage3-mrstft-weight", "$Stage3MrstftWeight",
    "--stage2-latent-l1-weight", "$Stage2LatentL1Weight",
    "--stage3-latent-l1-weight", "$Stage3LatentL1Weight",
    "--stage2-delta-budget", "$Stage2DeltaBudget",
    "--stage3-delta-budget", "$Stage3DeltaBudget",
    "--stage2-delta-budget-weight", "$Stage2DeltaBudgetWeight",
    "--stage3-delta-budget-weight", "$Stage3DeltaBudgetWeight",
    "--stage2-style-dropout-p", "$Stage2StyleDropoutP",
    "--stage3-style-dropout-p", "$Stage3StyleDropoutP",
    "--stage2-style-jitter-std", "$Stage2StyleJitterStd",
    "--stage3-style-jitter-std", "$Stage3StyleJitterStd",
    "--stage2-exemplar-noise-std", "$Stage2ExemplarNoiseStd",
    "--stage3-exemplar-noise-std", "$Stage3ExemplarNoiseStd",
    "--stage2-style-embed-align-weight", "$Stage2StyleEmbedAlignWeight",
    "--stage3-style-embed-align-weight", "$Stage3StyleEmbedAlignWeight",
    "--stage2-generated-mert-weight", "$Stage2GeneratedMertWeight",
    "--stage3-generated-mert-weight", "$Stage3GeneratedMertWeight",
    "--stage2-generated-mert-align-weight", "$Stage2GeneratedMertAlignWeight",
    "--stage3-generated-mert-align-weight", "$Stage3GeneratedMertAlignWeight",
    "--stage2-generated-mert-every", "$Stage2GeneratedMertEvery",
    "--stage3-generated-mert-every", "$Stage3GeneratedMertEvery",
    "--stage3-mode-seeking-weight", "$Stage3ModeSeekingWeight",
    "--stage3-mode-seeking-target", "$Stage3ModeSeekingTarget",
    "--sample-count", "16",
    "--sample-export-tag", "rebuild_probe_samples",
    "--epoch-sample-source-file", $EpochSampleSourceFile,
    "--epoch-sample-offset-sec", "$EpochSampleOffsetSec",
    "--epoch-sample-every", "1",
    "--epoch-sample-tag", "epoch_samples"
)

$trainCmd = (($trainArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $trainLog) + " 2>&1"
Add-Content -Path $trainLog -Value ("[codec probe train] " + $trainCmd)
cmd /c $trainCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec rebuild probe failed for $RunName. See $trainLog"
}

Write-Host "[codec-probe] training finished, starting realism sweep"

$sweepArgs = @(
    "python", "-u", "lab 3/run_lab3_realism_sweep.py", "codec",
    "--run-dir", ("saves2/lab3_codec_transfer/" + $RunName),
    "--n-samples", "$SweepSamples",
    "--write-audio-count", "$SweepWriteAudioCount",
    "--max-fad-mert", "30",
    "--min-mps", "0.94",
    "--min-style-target-acc", "0.15",
    "--min-style-target-cos", "0.00"
)
$sweepCmd = (($sweepArgs | ForEach-Object { Quote-CmdArg $_ }) -join " ") + " >> " + (Quote-CmdArg $sweepLog) + " 2>&1"
Add-Content -Path $sweepLog -Value ("[codec probe sweep] " + $sweepCmd)
cmd /c $sweepCmd

if ($LASTEXITCODE -ne 0) {
    throw "Codec rebuild probe realism sweep failed for $RunName. See $sweepLog"
}

Write-Host "[codec-probe] finished: $RunName"
