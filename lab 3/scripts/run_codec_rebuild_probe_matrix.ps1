param(
    [string]$BootstrapCkpt = "saves2/lab3_codec_transfer/run1056/checkpoints/stage1_latest.pt",
    [string]$EpochSampleSourceFile = "C:\Users\Ahmed\Downloads\Milky & Mall Grab - Just The Way You Are.flac",
    [double]$EpochSampleOffsetSec = 0.0,
    [int]$Stage2Epochs = 2,
    [int]$Stage3Epochs = 2,
    [int]$MaxBatchesPerEpoch = 8,
    [int]$SweepSamples = 8,
    [int]$SweepWriteAudioCount = 2
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

$variants = @(
    @{
        RunName = "run1057probe5"
        TranslatorDirectMix = 0.35
        Stage2AdvWeight = 0.50
        Stage3AdvWeight = 0.62
        Stage2StyleWeight = 9.0
        Stage3StyleWeight = 10.8
        Stage2GeneratedMertWeight = 0.45
        Stage3GeneratedMertWeight = 0.60
        Stage2GeneratedMertAlignWeight = 0.15
        Stage3GeneratedMertAlignWeight = 0.20
        Stage2LatentL1Weight = 0.15
        Stage3LatentL1Weight = 0.04
        Stage2MrstftWeight = 0.12
        Stage3MrstftWeight = 0.05
        Stage2ContentWeight = 2.5
        Stage3ContentWeight = 2.0
        Stage2DeltaBudgetWeight = 0.0
        Stage3DeltaBudgetWeight = 0.20
    },
    @{
        RunName = "run1057probe6"
        TranslatorDirectMix = 0.30
        Stage2AdvWeight = 0.48
        Stage3AdvWeight = 0.58
        Stage2StyleWeight = 8.8
        Stage3StyleWeight = 10.2
        Stage2GeneratedMertWeight = 0.40
        Stage3GeneratedMertWeight = 0.55
        Stage2GeneratedMertAlignWeight = 0.12
        Stage3GeneratedMertAlignWeight = 0.18
        Stage2LatentL1Weight = 0.18
        Stage3LatentL1Weight = 0.06
        Stage2MrstftWeight = 0.15
        Stage3MrstftWeight = 0.06
        Stage2ContentWeight = 2.6
        Stage3ContentWeight = 2.1
        Stage2DeltaBudgetWeight = 0.0
        Stage3DeltaBudgetWeight = 0.15
    },
    @{
        RunName = "run1057probe7"
        TranslatorDirectMix = 0.40
        Stage2AdvWeight = 0.50
        Stage3AdvWeight = 0.64
        Stage2StyleWeight = 9.2
        Stage3StyleWeight = 11.0
        Stage2GeneratedMertWeight = 0.48
        Stage3GeneratedMertWeight = 0.65
        Stage2GeneratedMertAlignWeight = 0.16
        Stage3GeneratedMertAlignWeight = 0.22
        Stage2LatentL1Weight = 0.14
        Stage3LatentL1Weight = 0.04
        Stage2MrstftWeight = 0.11
        Stage3MrstftWeight = 0.04
        Stage2ContentWeight = 2.45
        Stage3ContentWeight = 1.95
        Stage2DeltaBudgetWeight = 0.0
        Stage3DeltaBudgetWeight = 0.18
    }
)

$summaryRows = New-Object System.Collections.Generic.List[object]

foreach ($variant in $variants) {
    Write-Host "[probe-matrix] running $($variant.RunName)"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_codec_rebuild_probe.ps1") `
        -RunName $variant.RunName `
        -BootstrapCkpt $BootstrapCkpt `
        -EpochSampleSourceFile $EpochSampleSourceFile `
        -EpochSampleOffsetSec $EpochSampleOffsetSec `
        -Stage2Epochs $Stage2Epochs `
        -Stage3Epochs $Stage3Epochs `
        -MaxBatchesPerEpoch $MaxBatchesPerEpoch `
        -SweepSamples $SweepSamples `
        -SweepWriteAudioCount $SweepWriteAudioCount `
        -TranslatorDirectMix $variant.TranslatorDirectMix `
        -Stage2AdvWeight $variant.Stage2AdvWeight `
        -Stage3AdvWeight $variant.Stage3AdvWeight `
        -Stage2StyleWeight $variant.Stage2StyleWeight `
        -Stage3StyleWeight $variant.Stage3StyleWeight `
        -Stage2GeneratedMertWeight $variant.Stage2GeneratedMertWeight `
        -Stage3GeneratedMertWeight $variant.Stage3GeneratedMertWeight `
        -Stage2GeneratedMertAlignWeight $variant.Stage2GeneratedMertAlignWeight `
        -Stage3GeneratedMertAlignWeight $variant.Stage3GeneratedMertAlignWeight `
        -Stage2LatentL1Weight $variant.Stage2LatentL1Weight `
        -Stage3LatentL1Weight $variant.Stage3LatentL1Weight `
        -Stage2MrstftWeight $variant.Stage2MrstftWeight `
        -Stage3MrstftWeight $variant.Stage3MrstftWeight `
        -Stage2ContentWeight $variant.Stage2ContentWeight `
        -Stage3ContentWeight $variant.Stage3ContentWeight `
        -Stage2DeltaBudgetWeight $variant.Stage2DeltaBudgetWeight `
        -Stage3DeltaBudgetWeight $variant.Stage3DeltaBudgetWeight

    $bestJson = Join-Path $repoRoot ("saves2\lab3_codec_transfer\" + $variant.RunName + "\realism_supervisor\codec_realism_best.json")
    if (Test-Path $bestJson) {
        $payload = Get-Content $bestJson -Raw | ConvertFrom-Json
        $best = $payload.best
        $summaryRows.Add([pscustomobject]@{
            run_name = $variant.RunName
            direct_mix = $variant.TranslatorDirectMix
            fad_mert = [double]$best.fad_mert
            mps = [double]$best.mps
            style_target_acc = [double]$best.style_target_acc
            style_target_cos = [double]$best.style_target_cos
            hf_mae = [double]$best.target_hf_mae
            dyn_range_mae = [double]$best.target_dynamic_range_mae_db
            realism_score = [double]$best.realism_score
        }) | Out-Null
    }
}

$summaryDir = Join-Path $repoRoot "saves2\lab3_codec_transfer\probe_matrix"
New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
$summaryCsv = Join-Path $summaryDir "latest_probe_matrix_summary.csv"
$summaryRows | Sort-Object realism_score | Export-Csv -Path $summaryCsv -NoTypeInformation

Write-Host "[probe-matrix] summary: $summaryCsv"
$summaryRows | Sort-Object realism_score | Format-Table -AutoSize
